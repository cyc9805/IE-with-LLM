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
from utils import DataCollatorWithPadding, analyze_raw_data

CUR_DIR=os.path.dirname(os.path.abspath(__file__))
sys.path.append(f"{CUR_DIR}/..")

@dataclass
class llamaArgument(Seq2SeqTrainingArguments):
    max_new_tokens: int = field(default=100, metadata={"help": "The maximum numbers of tokens to generate, ignoring the number of tokens in the prompt."})
    

class ModelTrainer(Seq2SeqTrainer):
    def __init__(
        self,
        model: PreTrainedModel=None,
        args: TrainingArguments=None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
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
        
    # def compute_loss(self, model, inputs, return_outputs=False):

    def evaluate(
        self, 
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        **gen_kwargs,
    ):
        # memory metrics - must set up as early as possible
        self._memory_tracker.start()
        dataset = eval_dataset if eval_dataset else self.eval_dataset
        metrics = self._evaluation_loop(dataset=dataset)
        
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
        dataloader = torch.utils.data.DataLoader(dataset,
                                                shuffle=False,
                                                collate_fn=data_collator,
                                                batch_size=batch_size,
                                                num_workers=self.args.dataloader_num_workers,
                                                pin_memory=True,
                                                drop_last=False)

        model.eval()
        preds, refs = list(), list()
        for batch in tqdm(dataloader):
            batch = self._prepare_inputs(batch)
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"] 
            predicted_ids = model.generate(
                                        input_ids=input_ids,
                                        do_sample=False,
                                        eos_token_id=terminators,
                                        attention_mask=attention_mask,
                                        max_new_tokens=self.args.max_new_tokens,
                                        # temperature=self.args.temperature,
                                        # top_p=self.args.top_p,
                                        # repetition_penalty=self.args.reptition_penalty,
                                        # no_repeat_ngram_size=self.args.no_repeat_ngram_size
                                        )

            for i, input_id in zip(range(input_ids.shape[0]), input_ids):
                response = tokenizer.decode(predicted_ids[i][input_id.shape[-1]:], skip_special_tokens=True)
                preds.append(response)
            refs += batch["labels"]

        # Calculate metrics
        logging.info("Compute F1 score")
        analyzed_result = analyze_raw_data(preds=preds, refs=refs, dataset_name=dataset.config_name)
        
        pred_dataset = Dataset.from_dict(
            dict(pred=preds, ref=refs, metric_scores=analyzed_result['metric_scores'])
            )
        df = pred_dataset.to_pandas()
        pred_result_path = f"{output_dir}/prediction.csv"
        df.to_csv(pred_result_path)
        logging.info(f"Prediction result saved in {pred_result_path}")
        
        # Average metric score
        output = {"avg_metric_score": analyzed_result['avg_metric_score']}
        json.dump(output, open(f"{output_dir}/metrics.json", "w"))
        # # logging.info(metrics)

        return output