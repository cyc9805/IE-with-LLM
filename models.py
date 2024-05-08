import torch
import logging
import torch.nn as nn
from typing import List, Tuple, Optional, Union
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel, LlamaModel, LlamaConfig, LlamaForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput
from peft import PeftModel, PeftConfig, LoraConfig, get_peft_model, PromptTuningConfig, PeftType, TaskType, PromptTuningInit
from ie_llm.utils import print_trainable_parameters, enable_peft_finetuning
from ie_llm.prompt_tuning import PromptTuningConfigForIe

MODELS = {
    "llama3": "meta-llama/Meta-Llama-3-8B",
    "llama3-instruct": "meta-llama/Meta-Llama-3-8B-instruct",
    "gte-large": "Alibaba-NLP/gte-large-en-v1.5"
}

MODEL_DTYPES = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "default": torch.float32,
}

# def IeLlmModel(LlamaModel):



def load_model(model_name:str, peft_type:str=None, peft_ckpt_dir:str=None, model_dtype:str=MODEL_DTYPES['default'], use_encoder_model_for_peft:bool=False)-> Tuple[AutoModelForCausalLM, AutoTokenizer, List[str]]:
    if model_name not in MODELS:
        raise ValueError(f"Model {model_name} not found. Available models: {MODELS.keys()}")

    # Load models
    tokenizer = AutoTokenizer.from_pretrained(MODELS[model_name])
    model = AutoModelForCausalLM.from_pretrained(MODELS[model_name], device_map='auto', torch_dtype=MODEL_DTYPES[model_dtype])
    terminators = [tokenizer.eos_token_id]
    
    if model_name in ["llama3", "llama3-instruct"]:
        tokenizer.pad_token = tokenizer.eos_token
        terminators.append(tokenizer.convert_tokens_to_ids("<|eot_id|>"))

    if peft_ckpt_dir:
        model = PeftModel.from_pretrained(model, peft_ckpt_dir)
        enable_peft_finetuning(model, peft_type)

    elif peft_type is not None:
        if peft_type == "prompt_tuning":
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

            encoder_model_name = 'gte-large'
            encoder_tokenizer = AutoTokenizer.from_pretrained(encoder_model_name)
            encoder_model = AutoModel.from_pretrained(encoder_model_name)
            
            
            
        elif peft_type == 'lora':
            if use_encoder_model_for_peft:
                logging.warning("LORA does not use additional encoder model")      

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


class IeLlamaConfig(LlamaConfig):
    model_type = "llama_for_ie"


class IeLlamaModel(LlamaModel):

    def __init__(self, config: LlamaConfig):
        super(IeLlamaModel, self).__init__(config)


class IeLlamaForCausalLM(LlamaForCausalLM):

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = IeLlamaModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.align_matrix = nn.Linear(config.encoder_token_dim, config.token_dim)
        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def prepare_inputs_labels():
        pass

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                image_sizes
            )

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _
            ) = self.prepare_inputs_labels(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                image_sizes=image_sizes
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs['images'] = images
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes
        return inputs

AutoConfig.register("llava_llama", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)
