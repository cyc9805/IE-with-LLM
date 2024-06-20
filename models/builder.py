import torch
import logging
import torch.nn as nn
from typing import List, Tuple, Optional, Union
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from peft import PeftModel, LoraConfig, get_peft_model
from ie_llm.models.llama_model_for_ie import LlamaForIe, IeLlamaConfig
from ie_llm.utils import print_trainable_parameters, enable_peft_finetuning
from ie_llm.prompt_tuning import PromptTuningConfigForIe

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

def load_model(model_name:str, peft_type:str=None, peft_ckpt_dir:str=None, model_dtype:str=MODEL_DTYPES['default'], noise_data:bool=False)-> Tuple[AutoModelForCausalLM, AutoTokenizer, List[str]]:
    if model_name not in MODELS:
        raise ValueError(f"Model {model_name} not found. Available models: {MODELS.keys()}")

    # Load models
    if model_name == "ie_llm":
        model = LlamaForIe.from_pretrained(MODELS[model_name], device_map='auto', torch_dtype=MODEL_DTYPES[model_dtype])
    else:
        model = AutoModelForCausalLM.from_pretrained(MODELS[model_name], device_map='auto', torch_dtype=MODEL_DTYPES[model_dtype])

    tokenizer = AutoTokenizer.from_pretrained(MODELS[model_name])
    terminators = [tokenizer.eos_token_id]
    
    # 이거 수정해야 함
    # if noise_data:
        # tokenizer.add_special_tokens({"<|mask_token|>":tokenizer.convert_tokens_to_ids('<|reserved_special_token_0|>')})

    if model_name in ["llama3", "llama3-instruct", "ie_llm"]:
        tokenizer.pad_token = tokenizer.eos_token
        terminators.append(tokenizer.convert_tokens_to_ids("<|eot_id|>"))

    peft_type = peft_type.lower() if peft_type is not None else None

    if peft_ckpt_dir:
        model = PeftModel.from_pretrained(model, peft_ckpt_dir)
        enable_peft_finetuning(model, peft_type)

    elif peft_type is not None:
        # if peft_type == "prompt_tuning":
            # config = PromptTuningConfigForIe(
            #     peft_type="PROMPT_TUNING",
            #     task_type="SEQ_2_SEQ_LM",
            #     num_virtual_tokens=20,
            #     token_dim=tokenizer.config.hidden_size,
            #     num_transformer_submodules=1,
            #     num_attention_heads=12,
            #     num_layers=12,
            #     prompt_tuning_init="TEXT",
            #     prompt_tuning_init_text="Predict if sentiment of this review is positive, negative or neutral",
            #     tokenizer_name_or_path=MODELS[model_name],
            #     use_encoder=use_encoder_model_for_peft,
            #     encoder_token_dim=1024,
            #     encoder_model_name=MODELS["gte-large"],
            # )

        #     encoder_model_name = 'gte-large'
        #     encoder_tokenizer = AutoTokenizer.from_pretrained(encoder_model_name)
        #     encoder_model = AutoModel.from_pretrained(encoder_model_name)
            
            
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
