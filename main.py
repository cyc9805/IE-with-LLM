import argparse
import os
import sys
import json
import logging
from setproctitle import setproctitle
from datetime import datetime
from rich.logging import RichHandler

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f"{CUR_DIR}/..")

from trainer import ModelTrainer, llamaArgument
from models.builder import load_model
from data import prepare_dataset, DataCollatorWithPaddingForIeLLM
setproctitle('yongchan')

AVALIABLE_TASK = ['denoising', 'open_ie', 'closed_ie']
AVAILABLE_DATASET = ['dialog_re', 'allenai/c4', 'c4']
AVAILABLE_EVALUATION_METRICS = ['f1', 'perplexity']

def main(cfg):
    # Set up logging
    now = datetime.now().strftime("%m_%d_%H_%M_%S")
    train_mode = cfg["train_mode"]
    task = cfg["task"]
    dataset_name = cfg["dataset_name"]
    model_dtype = cfg['model_dtype']
    prefix_lm_mode = cfg['prefix_lm_mode']
    wandb_run_id = cfg.get('wandb_run_id', now)
    evaluation_metrics = cfg.get('evaluation_metrics', ['f1'])
    metric_for_best_model = cfg.get('metric_for_best_model', 'micro_f1_score')

    if task not in AVALIABLE_TASK:
        raise ValueError(f"task should be one of {AVALIABLE_TASK}, but got {task}")
    
    if dataset_name not in AVAILABLE_DATASET:
        raise ValueError(f"dataset_name should be one of {AVAILABLE_DATASET}, but got {dataset_name}")
    
    if not all([metric in AVAILABLE_EVALUATION_METRICS for metric in evaluation_metrics]):
        raise ValueError(f"evaluation_metrics should be one of {AVAILABLE_EVALUATION_METRICS}, but got {evaluation_metrics}")
    
    if task != 'denoising' and dataset_name == 'c4':
        raise ValueError(f"C4 dataset is not supported for {task} task")

    os.environ["WANDB_PROJECT"] = "IE-LLM"
    os.environ["WANDB_RUN_ID"] = f"{'train_' if train_mode else 'test_'}{task}_{wandb_run_id}_{now}"

    output_dir = os.path.join(cfg["output_dir"], 'train' if train_mode else 'test', task, now)
    os.makedirs(output_dir, exist_ok=True)
    json.dump(cfg, open(f"{output_dir}/run_config.json", "w"), indent=2)
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(f"{output_dir}/run.log"),
                  RichHandler()])

    # Load model
    logging.info("Start loading model")
    model, tokenizer, terminators = load_model(cfg['model_name'], cfg['peft_type'], cfg.get('peft_ckpt_dir', None), model_dtype)

    # Load Dataset
    logging.info("Start loading dataset")
    dataset = prepare_dataset(
        dataset_name=dataset_name, 
        tokenizer=tokenizer, 
        task=task, 
        cache_file_name=cfg["cache_file_name"],
        seed=cfg["seed"],
        )

    # Setup trainer
    training_args = llamaArgument(
        output_dir=output_dir,
        seed=cfg["seed"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"], 
        learning_rate=cfg['learning_rate'],
        weight_decay=cfg["weight_decay"],
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
        greater_is_better=cfg["greater_is_better"],
        metric_for_best_model=metric_for_best_model,
        save_total_limit=cfg["save_total_limit"],
        push_to_hub=False,
        predict_with_generate=cfg["predict_with_generate"],
        remove_unused_columns=False,
        label_names=['labels'],
        auto_find_batch_size=cfg['auto_find_batch_size'],
        report_to="wandb",
        num_beams=cfg["num_beams"],
        sample_for_evaluation=cfg["sample_for_evaluation"],
        num_samples=cfg["num_samples"],
        task_name=task,
        evaluation_metrics=evaluation_metrics,
        prefix_lm_mode=prefix_lm_mode,
    )

    trainer = ModelTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        terminators=terminators
    )
    
    trainer.data_collator = DataCollatorWithPaddingForIeLLM(
            tokenizer=tokenizer,
            generation_mode=False,
            task=task,
            prefix_lm_mode=prefix_lm_mode,
            evaluation_metrics=evaluation_metrics)


    if not train_mode:
        trainer.set_eval_dataset(dataset['test'])
        logging.info('Start evaluating ...')
        trainer.evaluate()
    else:
        trainer.train_dataset = dataset['train']
        trainer.set_eval_dataset(dataset['validation'])
        logging.info('Start training ...')
        trainer.train(resume_from_checkpoint=cfg.get("resume_from_checkpoint", None))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="configs/llama-3-instruct.jsonl", help="Path to the config file")
    args = parser.parse_args()
    cfg = json.load(open(args.config_path, 'r'))
    main(cfg)

