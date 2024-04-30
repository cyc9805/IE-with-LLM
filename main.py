import argparse
import os
import sys
import json
import logging
import numpy as np
from setproctitle import setproctitle
from datetime import datetime
from rich import print
from rich.logging import RichHandler
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f"{CUR_DIR}/..")

from ie_llm.trainer import ModelTrainer, llamaArgument
from ie_llm.models import load_model
from ie_llm.data import prepare_dataset, DataCollatorWithPadding

setproctitle('yongchan')

def main(cfg):
    
    # Set up logging
    now = datetime.now().strftime("%m_%d_%H_%M_%S")
    output_dir = os.path.join(cfg["output_dir"], now)

    # Load model
    model, tokenizer, terminators = load_model(cfg['model_name'])

    # Load Dataset
    dataset = prepare_dataset(cfg["dataset_name"], tokenizer, cfg["cache_file_name"])

    # Setup trainer
    training_args = llamaArgument(
        output_dir=output_dir,
        seed=cfg["seed"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 1), 
        learning_rate=cfg['learning_rate'],
        weight_decay=cfg.get("weight_decay", 0),
        warmup_steps=cfg["warmup_steps"],
        gradient_checkpointing=False,
        fp16=True,
        evaluation_strategy=cfg["evaluation_strategy"],
        dataloader_num_workers=cfg["dataloader_num_workers"],
        generation_max_length=cfg["generation_max_length"],
        per_device_eval_batch_size=cfg['per_device_eval_batch_size'],
        eval_steps=cfg["eval_steps"],
        save_steps=cfg["save_steps"],
        logging_steps=cfg["logging_steps"],
        max_new_tokens=512,
        num_train_epochs=cfg['num_train_epochs'],
        load_best_model_at_end=cfg["load_best_model_at_end"],
        greater_is_better=False,
        push_to_hub=False,
        predict_with_generate=cfg.get("predict_with_generate", False),
        remove_unused_columns=False,
        label_names=['labels'],
        auto_find_batch_size=cfg.get('auto_find_batch_size', False),
        report_to="wandb",
        # max_steps=cfg['max_steps']
        # p_value=cfg.get('p_value', 0.3),
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
    
    trainer.data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    if not cfg["train_mode"]:
        trainer.evaluate(dataset['test'])
    else:
        trainer.train_dataset = dataset['train']
        trainer.eval_dataset = dataset['validation']
        logging.info('Start training ...')
        trainer.train(resume_from_checkpoint=cfg.get("resume_from_checkpoint", None))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="configs/llama-3-instruct.jsonl", help="Path to the config file")
    args = parser.parse_args()
    cfg = json.load(open(args.config_path, 'r'))
    main(cfg)

