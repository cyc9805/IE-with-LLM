import argparse
import os
import sys
import json
import numpy as np
from setproctitle import setproctitle
from datetime import datetime
from rich import print
from rich.logging import RichHandler

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f"{CUR_DIR}/..")

from ie_llm.utils import DataCollatorWithPadding
from ie_llm.trainer import ModelTrainer, llamaArgument
from ie_llm.models import load_model
from ie_llm.data import prepare_dataset

setproctitle('yongchan')

def main(cfg):
    
    # Set up logging
    now = datetime.now().strftime("%m_%d_%H_%M_%S")
    output_dir = os.path.join(cfg["output_dir"], now)

    # Load model
    model, tokenizer, terminators = load_model(cfg['model_name'])

    # Load Dataset
    dataset = prepare_dataset(cfg["dataset_name"], tokenizer)

    # Setup trainer
    training_args = llamaArgument(
        per_device_eval_batch_size=4,
        evaluation_strategy="no",
        max_new_tokens=512,
        output_dir=output_dir,
        report_to="wandb",
        # repetition_penalty=1.0,
        # no_repeat_ngram_size=3,
    )

    trainer = ModelTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        terminators=terminators,
        # compute_metrics=compute_metrics,
    )
    
    trainer.data_collator = DataCollatorWithPadding(tokenizer=tokenizer, generation_mode=True)

    if not cfg["train_mode"]:
        trainer.evaluate(dataset['test'])



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="configs/llama-3-instruct.jsonl", help="Path to the config file")
    args = parser.parse_args()
    cfg = json.load(open(args.config_path, 'r'))
    main(cfg)

