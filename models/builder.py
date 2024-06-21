import torch
from typing import List, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, LoraConfig, get_peft_model
from models.llama_model_for_ie import LlamaForIe
from utils import print_trainable_parameters, enable_peft_finetuning

MODELS = {
    "llama3": "meta-llama/Meta-Llama-3-8B",
    "llama3-instruct": "meta-llama/Meta-Llama-3-8B-instruct",
    "ie_llm" : "meta-llama/Meta-Llama-3-8B-instruct",
}

MODEL_DTYPES = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "default": torch.float32,
}

def load_model(
    model_name: str="ie_llm", 
    peft_type: str=None, 
    peft_ckpt_dir: str=None, 
    model_dtype: str=MODEL_DTYPES['default']
    )-> Tuple[AutoModelForCausalLM, AutoTokenizer, List[str]]:
    
    if model_name not in MODELS:
        raise ValueError(f"Model {model_name} not found. Available models: {MODELS.keys()}")

    # Load models
    if model_name == "ie_llm":
        model = LlamaForIe.from_pretrained(MODELS[model_name], device_map='auto', torch_dtype=MODEL_DTYPES[model_dtype])
    else:
        model = AutoModelForCausalLM.from_pretrained(MODELS[model_name], device_map='auto', torch_dtype=MODEL_DTYPES[model_dtype])

    tokenizer = AutoTokenizer.from_pretrained(MODELS[model_name])
    terminators = [tokenizer.eos_token_id]

    if model_name in ["llama3", "llama3-instruct", "ie_llm"]:
        tokenizer.pad_token = tokenizer.eos_token
        terminators.append(tokenizer.convert_tokens_to_ids("<|eot_id|>"))

    peft_type = peft_type.lower() if peft_type is not None else None

    if peft_ckpt_dir:
        model = PeftModel.from_pretrained(model, peft_ckpt_dir)
        enable_peft_finetuning(model, peft_type)

    elif peft_type is not None:
        # TODO: Add more PEFT configurations
        if peft_type == 'lora':
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
