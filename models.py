from typing import List, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments

MODELS = {
    "llama3": "meta-llama/Meta-Llama-3-8B",
    "llama3-instruct": "meta-llama/Meta-Llama-3-8B-instruct"
}

def load_model(model_name)-> Tuple[AutoModelForCausalLM, AutoTokenizer, List[str]]:
    if model_name not in MODELS:
        raise ValueError(f"Model {model_name} not found. Available models: {MODELS.keys()}")
        
    # Load two types of LLaMA3
    tokenizer = AutoTokenizer.from_pretrained(MODELS[model_name])
    model = AutoModelForCausalLM.from_pretrained(MODELS[model_name], device_map='auto')

    if model_name in ["llama3", "llama3-instruct"]:
        tokenizer.pad_token = tokenizer.eos_token
        
        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
    
    return model, tokenizer, terminators