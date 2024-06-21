# IE with LLM

## Pre-requisites
1. python >= 3.10.0
2. Clone this repository:
```bash
git clone https://github.com/cyc9805/IE-with-LLM.git
cd ie_llm
```
3. Install python requirements:
```bash
pip install -r requirements.txt
```

## Training
There are two settings for training: instruction-tuning to IE dataset / training with denoising task

1. Before instruction-tuning to IE dataset, you can customize the following configuration file:
```bash
vi configs/train/ie_configuration.jsonl 
```
Then, you can start training:
```bash
cd run_script/train
sh ie.sh
```

2. Before training with denoising task, you can customize the following configuration file:
```bash
vi configs/train/denoising_configuration.jsonl 
```
Then, you can start trainig:
```bash
cd run_script/train
sh denoising.sh
```

## Testing
There are two settings for testing: instruction-tuning to IE dataset / training with denoising task

1. Before testing model that is instruction-tuned to IE dataset, you can customize the following configuration file:
```bash
vi configs/test/ie_configuration.jsonl 
```
Then, you can start testing:
```bash
cd run_script/test
sh ie.sh
```

2. Before testing model trained with denoising task, you can customize the following configuration file:
```bash
vi configs/test/denoising_configuration.jsonl 
```
Then, you can start testing:
```bash
cd run_script/test
sh denoising.sh
```

## Configuration
In configuration files, there are a few important configurations to take care of:
- `dataset_name`: Name of the dataset. When performing instruction-tuning to IE dataset, it should be set to `dialog_re`. Otherwise, it should be set to one of `[dialog_re, c4]`
- `model_name`: To utilize prefix attention, it should be set to `ie_llm`. Otherwise, it supports `llama3` and `llama3-instruct`.
- `train_mode`: If set to `True`, it performs training. Otherwise, it performs testing.
- `peft_type`: `lora` is the only supported PEFT method right now.
- `peft_ckpt_dir`: To resume training or testing with previously trained PEFT model, provdie the directory to the adapter checkpoint. Otherwise, set to `null`.
- `prefix_lm_mode`: The prefix attention used in IE with LLM currently has 4 different modes: 
    1. `only_input_text`: This mode enables full attention to user prompt.
    2. `only_system_prompt`: This mode enables full attention to system prompt.
    3. `all`: This mode enables full attention to both system prompt and user prompt.
    4. `null`: Disable prefix attention.
- `task`: Task should be one of `[closed_ie, open_ie, denoising]`.
- `evaluation_metrics`: This should be a list containing elements from `[f1, perplexity]`. For example, if set to `[f1]`, f1 score is used as a metric for evaluation.
- `model_dtype`: This should be on of `[bf16, fp16, default]`.
- `output_dir`: Directory to the folder where the result is saved.
- `cache_file_name`: Name of the dataset cache file.

The remaining configurations are identical to that of huggingface trainer. 