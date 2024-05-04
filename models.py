import torch
from typing import List, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaModel, LlamaConfig, LlamaForCausalLM
from peft import PeftModel, LoraConfig, get_peft_model, PromptTuningConfig, TaskType, PromptTuningInit
from ie_llm.utils import print_trainable_parameters, enable_lora_finetuning

MODELS = {
    "llama3": "meta-llama/Meta-Llama-3-8B",
    "llama3-instruct": "meta-llama/Meta-Llama-3-8B-instruct"
}

MODEL_DTYPES = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "default": torch.float32,
}

# def IeLlmModel(LlamaModel):
    

# def peft_config():
#     peft_config = PromptTuningConfig(
#         task_type=TaskType.CAUSAL_LM,
#         prompt_tuning_init=PromptTuningInit.TEXT,
#         num_virtual_tokens=8,
#         prompt_tuning_init_text="Classify if the tweet is a complaint or not:",
#         tokenizer_name_or_path=model_name_or_path,
#     )
#     return peft_config


def load_model(model_name:str, set_peft:bool, peft_ckpt_dir:str, model_dtype:str)-> Tuple[AutoModelForCausalLM, AutoTokenizer, List[str]]:
    if model_name not in MODELS:
        raise ValueError(f"Model {model_name} not found. Available models: {MODELS.keys()}")
    
    # Load two types of LLaMA3
    tokenizer = AutoTokenizer.from_pretrained(MODELS[model_name])
    model = AutoModelForCausalLM.from_pretrained(MODELS[model_name], device_map='auto', torch_dtype=MODEL_DTYPES[model_dtype])

    if model_name in ["llama3", "llama3-instruct"]:
        tokenizer.pad_token = tokenizer.eos_token
        
        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
    
    if set_peft:        
        if peft_ckpt_dir:
            model = PeftModel.from_pretrained(model, peft_ckpt_dir)
            enable_lora_finetuning(model)
        else:
            config = LoraConfig(
                    r=64,
                    lora_alpha=128,
                    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
                    lora_dropout=0.1,
                    bias="none"
                )
            model = get_peft_model(model, config)
    
    print_trainable_parameters(model)
    return model, tokenizer, terminators

