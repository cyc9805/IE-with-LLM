import os
import sys
import json
import logging
import torch

from typing import Callable, Dict, Optional, List, Tuple, Union
from dataclasses import dataclass, field
from datasets import Dataset
from tqdm import tqdm
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, TrainingArguments
from transformers.data.data_collator import DataCollator
from transformers.modeling_utils import PreTrainedModel
from transformers.trainer_callback import TrainerCallback
from transformers.training_args import TrainingArguments
from transformers.tokenization_utils import PreTrainedTokenizer
from data import get_prefix_position, IGNORE_INDEX
from evaluate import analyze_raw_data
from utils import set_seed

CUR_DIR=os.path.dirname(os.path.abspath(__file__))
sys.path.append(f"{CUR_DIR}/..")


@dataclass
class llamaArgument(Seq2SeqTrainingArguments):
    task_name: str = field(default="ie", metadata={"help": "The name of the task to train."})
    max_new_tokens: int = field(default=100, metadata={"help": "The maximum numbers of tokens to generate, ignoring the number of tokens in the prompt."})
    num_beams: int = field(default=3, metadata={"help": "The number of beams to use for beam search."})
    sample_for_evaluation: bool = field(default=False, metadata={"help": "Whether to sample examples for evaluation."})
    num_samples: int = field(default=0, metadata={"help": "The number of samples to evaluate."})
    prefix_lm_mode: List[int] = field(default=None, metadata={"help": "Mode for prefix LM. 'all' means all input values attend each other. 'only_input_text' means only input text attend each other."})
    evaluation_metrics: List[str] = field(default=None, metadata={"help": "The evaluation metrics to use. Available metrics are micro_f1_score, f1c_score and perplexity."}) 

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
            self.sampler = torch.utils.data.RandomSampler(eval_dataset, num_samples=num_samples)
        else:
            self.sampler = None
            if num_samples > 0:
                logging.warning("num_samples is bigger than 0 but use_sampler is set to False. num_samples will be ignored.")
        

    def evaluate(
        self, 
        metric_key_prefix: str = "eval",
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
        evaluation_metrics = self.args.evaluation_metrics
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
        inputs, preds, refs, total_loss = list(), list(), list(), list()

        generation_config = {
                "do_sample": False,
                "eos_token_id": terminators,
                "max_new_tokens": self.args.max_new_tokens,
                "num_beams": self.args.num_beams,
                "use_cache": True
            }

        for batch in tqdm(dataloader):
            # Send batch to same device as model
            batch = self._prepare_inputs(batch)
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"] 
            prefix_positions = batch["prefix_positions"]
            labels = batch["labels"]
            generation_config["prefix_positions"] = prefix_positions
            generation_config["attention_mask"] = attention_mask

            if 'f1' in evaluation_metrics:
                if isinstance(labels[0], str):
                    labels = [json.loads(label) for label in labels]
                    
                predicted_ids = model.generate(
                                            input_ids=input_ids,
                                            **generation_config                       
                                            )
                input = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
                pred = self._segment_answer(tokenizer, input_ids, predicted_ids)
                refs += labels   

            if 'perplexity' in evaluation_metrics:
                if isinstance(labels[0], dict):
                    labels = [json.dumps(label, ensure_ascii=False) for label in labels]

                concat_input_ids = []
                for input_id in input_ids:
                    while input_id[-1] == tokenizer.pad_token:
                        input_id = input_id[:-1]
                    concat_input_ids.append(input_id.tolist())

                padded_concat_input_ids = []
                padded_concat_labels = []

                for x, y in zip(concat_input_ids, labels):
                    input_id = x + self.tokenizer(y)['input_ids'][1:]
                    label = [IGNORE_INDEX]*len(x) + self.tokenizer(y)['input_ids'][1:]
                    label = label[1:] + [IGNORE_INDEX]
                    padded_concat_input_ids.append({"input_ids": input_id})
                    padded_concat_labels.append({"input_ids": label})

                self.tokenizer.padding_side = "left"
                padded_concat_input_ids = self.tokenizer.pad(padded_concat_input_ids, return_tensors="pt")
                padded_concat_labels = self.tokenizer.pad(padded_concat_labels, return_tensors="pt")

                attention_mask = padded_concat_input_ids["attention_mask"]
                prefix_positions = [get_prefix_position(x, task_name, self.args.prefix_lm_mode, tokenizer) for x in padded_concat_input_ids['input_ids']]

                padded_concat_input_ids = self._prepare_inputs(padded_concat_input_ids['input_ids'])
                padded_concat_labels = self._prepare_inputs(padded_concat_labels['input_ids'])

                with torch.no_grad():
                    output = model(padded_concat_input_ids, labels=padded_concat_labels, attention_mask=attention_mask, prefix_positions=prefix_positions)
                total_loss.append(output.loss.item())
                
                if len(evaluation_metrics) == 1:
                    input = tokenizer.batch_decode(padded_concat_input_ids, skip_special_tokens=True)
                    pred = ['No prediction'] * len(labels)

            inputs += input
            preds.extend(pred)

        # Calculate metrics
        analyzed_result = dict()
        for evaluation_metric in evaluation_metrics:
            logging.info(f"Compute {evaluation_metric} score")
            
            if evaluation_metric == 'f1':
                f1_result = analyze_raw_data(preds=preds, refs=refs, task_name=task_name)
                analyzed_result.update(f1_result)
                
            if evaluation_metric == 'perplexity':
                perplexity = torch.exp(torch.tensor(total_loss).mean()).item()
                analyzed_result["perplexity"] = perplexity
                
        # Individual metric score
        contents = dict(
                inputs=inputs,
                pred=preds,
            )
        
        # Average metric score
        output = dict()

        if 'f1' in evaluation_metrics:
            output["total_precision"] = analyzed_result['total_precision']
            output["total_recall"] = analyzed_result['total_recall']
            output["micro_f1_score"] = analyzed_result['micro_f1_score']
            contents.update({
                "refs": refs,
                "precisions": analyzed_result['precision'],
                "recalls": analyzed_result['recall'],
                "f1_score": analyzed_result['f1_score']
            })

        if 'perplexity' in evaluation_metrics:
            output["perplexity"] = analyzed_result["perplexity"]
            
        pred_dataset = Dataset.from_dict(contents)
        df = pred_dataset.to_pandas()
        pred_result_path = f"{output_dir}/prediction.csv"
        df.to_csv(pred_result_path)
        logging.info(f"Prediction result saved in {pred_result_path}")
        json.dump(output, open(f"{output_dir}/metrics.json", "w"))
        self.data_collator.set_generation_mode(False)
        
        return output
    

