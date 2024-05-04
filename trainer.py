import os
import sys
from typing import Callable, Dict, Optional, List, Tuple, Union, Any
from dataclasses import dataclass, field
from datasets import Dataset
from tqdm import tqdm
import json
import logging
import torch
from transformers import Seq2SeqTrainer, LlamaForCausalLM
from transformers.data.data_collator import DataCollator
from transformers.modeling_utils import PreTrainedModel
from transformers.trainer_callback import TrainerCallback
from transformers import Seq2SeqTrainingArguments, TrainingArguments
from transformers.trainer_utils import EvalPrediction
from transformers.training_args import TrainingArguments
from transformers.tokenization_utils import PreTrainedTokenizer
from utils import analyze_raw_data

CUR_DIR=os.path.dirname(os.path.abspath(__file__))
sys.path.append(f"{CUR_DIR}/..")

from ie_llm.data import DataCollatorWithPadding

@dataclass
class llamaArgument(Seq2SeqTrainingArguments):
    max_new_tokens: int = field(default=100, metadata={"help": "The maximum numbers of tokens to generate, ignoring the number of tokens in the prompt."})
    num_beams: int = field(default=3, metadata={"help": "The number of beams to use for beam search."})
    sample_for_evaluation: bool = field(default=False, metadata={"help": "Whether to sample examples for evaluation."})
    num_samples: int = field(default=0, metadata={"help": "The number of samples to evaluate."})


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
        if not sample_for_evaluation and num_samples > 0:
            logging.warning("num_samples is set but use_sampler is False. num_samples will be ignored.")
        

    def evaluate(
        self, 
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        **gen_kwargs,
    ):
        # memory metrics - must set up as early as possible
        self._memory_tracker.start()
        metrics = self._evaluation_loop(dataset=self.eval_dataset)
        
        metrics = {f"{metric_key_prefix}/{k}": v for k, v in metrics.items()}
        
        self.log(metrics)
        
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)

        self._memory_tracker.stop_and_update_metrics(metrics)
        
        return metrics
    

    def _evaluation_loop(self, dataset)->Dict[str, float]:
        model = self.model
        tokenizer = self.tokenizer
        global_step = self.state.global_step
        terminators = self.terminators
        output_dir = f"{self.args.output_dir}/eval_step{global_step}"
        os.makedirs(output_dir, exist_ok=True)

        batch_size = self.args.per_device_eval_batch_size
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer, generation_mode=True)

        # Sample 100 examples
        dataloader = torch.utils.data.DataLoader(dataset,
                                                sampler=self.sampler,
                                                shuffle=False,
                                                collate_fn=data_collator,
                                                batch_size=batch_size,
                                                num_workers=self.args.dataloader_num_workers,
                                                pin_memory=True,
                                                drop_last=False)
        
        model.eval()
        inputs, preds, refs = list(), list(), list()

        for batch in tqdm(dataloader):
            # Send batch to same device as model
            batch = self._prepare_inputs(batch)
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"] 
            predicted_ids = model.generate(
                                        input_ids=input_ids,
                                        do_sample=False,
                                        eos_token_id=terminators,
                                        attention_mask=attention_mask,
                                        max_new_tokens=self.args.max_new_tokens,
                                        num_beams=self.args.num_beams,
                                        use_cache=True,
                                        # temperature=self.args.temperature,
                                        # top_p=self.args.top_p,
                                        # repetition_penalty=self.args.reptition_penalty,
                                        # no_repeat_ngram_size=self.args.no_repeat_ngram_size
                                        )

            for i, input_id in zip(range(input_ids.shape[0]), input_ids):
                response = tokenizer.decode(predicted_ids[i][input_id.shape[-1]:], skip_special_tokens=True)
                preds.append(response)
        
            refs += batch["labels"]
            inputs += tokenizer.batch_decode(input_ids, skip_special_tokens=True)

        # Calculate metrics
        logging.info("Compute F1 score")
        analyzed_result = analyze_raw_data(preds=preds, refs=refs, dataset_name=dataset.config_name)
        pred_dataset = Dataset.from_dict(
            dict(inputs=inputs, pred=preds, ref=refs, metric_scores=analyzed_result['f1_score'])
            )
        df = pred_dataset.to_pandas()
        pred_result_path = f"{output_dir}/prediction.csv"
        df.to_csv(pred_result_path)
        logging.info(f"Prediction result saved in {pred_result_path}")
        
        # Average metric score
        output = {"micro_f1_score": analyzed_result['micro_f1_score']}
        json.dump(output, open(f"{output_dir}/metrics.json", "w"))
        # # logging.info(metrics)

        return output
    

