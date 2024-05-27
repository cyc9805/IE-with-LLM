import os
import sys
from typing import Callable, Dict, Optional, List, Tuple, Union, Any
from dataclasses import dataclass, field
from datasets import Dataset
from tqdm import tqdm
import json
import logging
import torch
from transformers import Seq2SeqTrainer
from transformers.data.data_collator import DataCollator
from transformers.modeling_utils import PreTrainedModel
from transformers.trainer_callback import TrainerCallback
from transformers import Seq2SeqTrainingArguments, TrainingArguments
from transformers.trainer_utils import EvalPrediction
from transformers.training_args import TrainingArguments
from transformers.tokenization_utils import PreTrainedTokenizer
from data import get_prefix_position
from utils import analyze_raw_data, set_seed

CUR_DIR=os.path.dirname(os.path.abspath(__file__))
sys.path.append(f"{CUR_DIR}/..")

from ie_llm.data import DataCollatorWithPaddingForIeLLM

@dataclass
class llamaArgument(Seq2SeqTrainingArguments):
    task_name: str = field(default="ie", metadata={"help": "The name of the task to train."})
    generate_intermediate_output: bool=field(default=False, metadata={"help": "Whether to generate final answer via intermediate output."})
    max_new_tokens: int = field(default=100, metadata={"help": "The maximum numbers of tokens to generate, ignoring the number of tokens in the prompt."})
    num_beams: int = field(default=3, metadata={"help": "The number of beams to use for beam search."})
    sample_for_evaluation: bool = field(default=False, metadata={"help": "Whether to sample examples for evaluation."})
    num_samples: int = field(default=0, metadata={"help": "The number of samples to evaluate."})
    prefix_lm_mode: List[int] = field(default=None, metadata={"help": "Mode for prefix LM. 'all' means all input values attend each other. 'only_input_text' means only input text attend each other."})


class ModelTrainer(Seq2SeqTrainer):
    def __init__(
        self,
        model: PreTrainedModel=None,
        args: TrainingArguments=None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        train_dataset: Optional[Dataset] = None,
        data_collator: Optional[DataCollator] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        model_init: Optional[Callable[[], "PreTrainedModel"]] = None,
        # compute_metrics: Optional[Callable[["EvalPrediction"], Dict]] = None,
        callbacks: Optional[List["TrainerCallback"]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        terminators: List[PreTrainedTokenizer] = None,
    ):
        
        super().__init__(
                model=model,
                args=args,
                data_collator=data_collator,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                tokenizer=tokenizer,
                model_init=model_init,
                # compute_metrics=compute_metrics,
                callbacks=callbacks,
                optimizers=optimizers,
                preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            )
        self.terminators = terminators
        self.set_eval_dataset(eval_dataset)
    

    def set_eval_dataset(self, eval_dataset):
        self.eval_dataset = eval_dataset
        sample_for_evaluation = self.args.sample_for_evaluation
        num_samples = self.args.num_samples
        if sample_for_evaluation:
            assert num_samples > 0, "num_samples must be bigger than 0"
            if self.args.task_name == 'conversation_ie':
                logging.warning("Conversation IE does not support random sampling for evaluation.")
                self.sampler = None
            else:
                self.sampler = torch.utils.data.RandomSampler(eval_dataset, num_samples=num_samples)
        else:
            self.sampler = None
            if num_samples > 0:
                logging.warning("num_samples is bigger than 0 but use_sampler is False. num_samples will be ignored.")
        

    def evaluate(
        self, 
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        **gen_kwargs,
    ):
        # memory metrics - must set up as early as possible
        self._memory_tracker.start()
        metrics = self._evaluation_loop(dataset=self.eval_dataset)
        
        metrics = {f"{metric_key_prefix}_{k}": v for k, v in metrics.items()}
        
        self.log(metrics)
        
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)

        self._memory_tracker.stop_and_update_metrics(metrics)
        
        return metrics
    
    def _segment_answer(self, tokenizer: PreTrainedTokenizer, input_ids: torch.Tensor, predicted_ids: torch.Tensor):
        answers = list()
        for i, input_id in zip(range(input_ids.shape[0]), input_ids):
            response = tokenizer.decode(predicted_ids[i][input_id.shape[-1]:], skip_special_tokens=True)
            answers.append(response)
        return answers

    def _evaluation_loop(self, dataset)->Dict[str, float]:
        model = self.model
        tokenizer = self.tokenizer
        global_step = self.state.global_step
        terminators = self.terminators
        task_name = self.args.task_name
        num_samples = self.args.num_samples
        output_dir = f"{self.args.output_dir}/eval_step{global_step}"
        os.makedirs(output_dir, exist_ok=True)

        batch_size = self.args.per_device_eval_batch_size
        
        self.data_collator.set_generation_mode(True)

        set_seed(self.args.seed)
        dataloader = torch.utils.data.DataLoader(dataset,
                                                sampler=self.sampler,
                                                shuffle=False,
                                                collate_fn=self.data_collator,
                                                batch_size=batch_size,
                                                num_workers=self.args.dataloader_num_workers,
                                                pin_memory=True,
                                                drop_last=False)
        
        model.eval()
        inputs, preds, refs, total_intermediate_outputs = list(), list(), list(), list()

        generation_config = {
                "do_sample": False,
                "eos_token_id": terminators,
                "max_new_tokens": self.args.max_new_tokens,
                "num_beams": self.args.num_beams,
                "use_cache": True
                # temperature=self.args.temperature,
                # top_p=self.args.top_p,
                # repetition_penalty=self.args.reptition_penalty,
                # no_repeat_ngram_size=self.args.no_repeat_ngram_size
            }
        
        if task_name == 'conversation_ie':
            initial_num_samples = num_samples
            extract_prev_dialog_index = True

        for batch in tqdm(dataloader):
            # Send batch to same device as model
            batch = self._prepare_inputs(batch)
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"] 
            prefix_positions = batch["prefix_positions"]
            labels = batch["labels"]
            if isinstance(labels[0], str):
                labels = [json.loads(label) for label in labels]

            if task_name == 'conversation_ie' and self.args.sample_for_evaluation:
                if num_samples == 0:
                    break

                dialog_indexes = [x['dialog_index'] for x in labels]
                if extract_prev_dialog_index:
                    prev_dialog_index = labels[0]['dialog_index']
                    extract_prev_dialog_index = False
                # dialog_indexes = [x[0]['dialog_index']-labels[0][0]['dialog_index'] for x in labels]
                for i, dialog_index in enumerate(dialog_indexes):
                    if prev_dialog_index != dialog_index:
                        prev_dialog_index = dialog_index
                        num_samples -= 1
                        logging.info(f"Processed samples: {initial_num_samples-num_samples}/{initial_num_samples}")
                    
                    if num_samples == 0:
                        input_ids = input_ids[:i]
                        attention_mask = attention_mask[:i]
                        labels = labels[:i]
                        if prefix_positions:
                            prefix_positions = prefix_positions[:i]
                        break
                    
            generation_config["prefix_positions"] = prefix_positions
            generation_config["attention_mask"] = attention_mask
            if self.args.generate_intermediate_output:
                predicted_ids = model.generate(
                                        input_ids=input_ids,
                                        **generation_config
                                        )
                
                intermediate_outputs = self._segment_answer(tokenizer=tokenizer, input_ids=input_ids, predicted_ids=predicted_ids)                
                total_intermediate_outputs.extend(intermediate_outputs)
                posterior_input_messages = batch['posterior_input_messages']
                questions = batch['questions']
                input_dialogs = batch['input_dialogs']
                input_ids = {"input_ids":[]}
                for input_dialog, intermediate_output, posterior_input_message, question in zip(input_dialogs, intermediate_outputs, posterior_input_messages, questions):
                    user_input = input_dialog.format(intermediate_output) + question
                    posterior_input_message.append({"role": "user", "content": user_input})
                    input_ids['input_ids'].append(tokenizer.apply_chat_template(posterior_input_message, add_generation_prompt=True))

                tokenizer.padding_side = "left"
                input_ids = tokenizer.pad(input_ids, return_tensors="pt")
                attention_mask = input_ids["attention_mask"]
                prefix_positions = [get_prefix_position(x, self.args.prefix_lm_mode) for x in input_ids['input_ids']]
                input_ids = self._prepare_inputs(input_ids['input_ids'])

                generation_config["prefix_positions"] = prefix_positions
                generation_config["attention_mask"] = attention_mask

            # if self.args.task_name == 'conversation_ie':
            #     labels = batch['labels']

            predicted_ids = model.generate(
                                        input_ids=input_ids,
                                        **generation_config                       
                                        )
            
            refs += labels
            preds.extend(self._segment_answer(tokenizer, input_ids, predicted_ids))
            inputs += tokenizer.batch_decode(input_ids, skip_special_tokens=True)


        # Calculate metrics
        logging.info("Compute f1 score")
        analyzed_result = analyze_raw_data(preds=preds, refs=refs, task_name=task_name, dataset_name=dataset.config_name)

        # Individual metric score
        contents = dict(
                inputs=inputs,
                pred=preds,
            )
        
        # Average metric score
        output = {
            "total_precision": analyzed_result['total_precision'],
            "total_recall": analyzed_result['total_recall']
        }

        if task_name == 'conversation_ie':
            output["f1c_score"] = analyzed_result['f1c_score']
            preprocessed_refs = list()
            for ref in refs:
                ref = [{"relation": relation, "trigger_word":trigger_word} for relation, trigger_word in zip(ref['relations'], ref['trigger_words'])]
                preprocessed_refs.append(ref)
            contents["refs"] = preprocessed_refs
        else:
            contents.update({
                "refs": refs,
                "precisions": analyzed_result['precision'],
                "recalls": analyzed_result['recall'],
                "f1_score": analyzed_result['f1_score']
            })
            output["micro_f1_score"] = analyzed_result['micro_f1_score']

        if self.args.generate_intermediate_output:
           contents['intermediate_outputs'] = total_intermediate_outputs
        pred_dataset = Dataset.from_dict(contents)
        df = pred_dataset.to_pandas()
        pred_result_path = f"{output_dir}/prediction.csv"
        df.to_csv(pred_result_path)
        logging.info(f"Prediction result saved in {pred_result_path}")
        json.dump(output, open(f"{output_dir}/metrics.json", "w"))
        # # logging.info(metrics)

        self.data_collator.set_generation_mode(False)
        return output
    

