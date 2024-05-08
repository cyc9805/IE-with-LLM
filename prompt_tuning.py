from peft.tuners import PromptEmbedding, PromptTuningConfig, PromptTuningInit
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaModel, LlamaConfig, LlamaForCausalLM
from dataclasses import dataclass, field
from typing import Optional, Union
import torch
import torch.nn as nn
import math

    
# class PromptTuningInitForIe(PromptTuningInit):
#     pass


@dataclass
class PromptTuningConfigForIe(PromptTuningConfig):
    use_encoder: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to use encoder model for prompt tuning."},
    )
    encoder_token_dim: Optional[int] = field(
        default=None,
        metadata={"help": "The token dimension of the encoder model."},
    )
    encoder_model_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name or path of the encoder tokenizer."},
    )
    encoder_tokenizer_kwargs: Optional[dict] = field(
        default=None,
        metadata={"help": "The keyword arguments to pass to `AutoTokenizer.from_pretrained` for the encoder model."},
    )        

 
class PromptEmbeddingForIe(nn.Module):

    def __init__(self, config, word_embeddings):
        super().__init__()
        total_virtual_tokens = config.num_virtual_tokens * config.num_transformer_submodules
        self.embedding = nn.Embedding(total_virtual_tokens, config.encoder_token_dim)
        self.config = config
        if config.use_encoder and config.prompt_tuning_init == PromptTuningInit.TEXT and not config.inference_mode:
            assert config.encoder_token_dim and config.encoder_tokenizer_name_or_path
            self.align_matrix = nn.Linear(config.encoder_token_dim, config.token_dim)
            from transformers import AutoTokenizer, AutoModel
            tokenizer_kwargs = config.encoder_tokenizer_kwargs or {}
            self.encoder_model = AutoModel.from_pretrained(config.encoder_model_name)
            encoder_tokenizer = AutoTokenizer.from_pretrained(config.encoder_model_name, **tokenizer_kwargs)
            init_text = config.prompt_tuning_init_text
            init_token_ids = encoder_tokenizer(init_text)["input_ids"]
            # Trim or iterate until num_text_tokens matches total_virtual_tokens
            num_text_tokens = len(init_token_ids)
            if num_text_tokens > total_virtual_tokens:
                init_token_ids = init_token_ids[:total_virtual_tokens]
            elif num_text_tokens < total_virtual_tokens:
                num_reps = math.ceil(total_virtual_tokens / num_text_tokens)
                init_token_ids = init_token_ids * num_reps
            init_token_ids = init_token_ids[:total_virtual_tokens]
            init_token_ids = torch.LongTensor(init_token_ids).to(word_embeddings.weight.device)
            word_embedding_weights = word_embeddings(init_token_ids).detach().clone()
            word_embedding_weights = word_embedding_weights.to(torch.float32)
            self.embedding.weight = torch.nn.Parameter(word_embedding_weights)

        elif not config.use_encoder and config.prompt_tuning_init == PromptTuningInit.TEXT and not config.inference_mode:
            if config.prompt_tuning_init == PromptTuningInit.TEXT and not config.inference_mode:
                from transformers import AutoTokenizer
                tokenizer_kwargs = config.tokenizer_kwargs or {}
                tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name_or_path, **tokenizer_kwargs)
            init_text = config.prompt_tuning_init_text
            init_token_ids = tokenizer(init_text)["input_ids"]
            # Trim or iterate until num_text_tokens matches total_virtual_tokens
            num_text_tokens = len(init_token_ids)
            if num_text_tokens > total_virtual_tokens:
                init_token_ids = init_token_ids[:total_virtual_tokens]
            elif num_text_tokens < total_virtual_tokens:
                num_reps = math.ceil(total_virtual_tokens / num_text_tokens)
                init_token_ids = init_token_ids * num_reps
            init_token_ids = init_token_ids[:total_virtual_tokens]
            init_token_ids = torch.LongTensor(init_token_ids).to(word_embeddings.weight.device)

            word_embedding_weights = word_embeddings(init_token_ids).detach().clone()
            word_embedding_weights = word_embedding_weights.to(torch.float32)
            self.embedding.weight = torch.nn.Parameter(word_embedding_weights)

    def forward(self, indices):
        # Just get embeddings
        prompt_embeddings = self.embedding(indices)
        if self.config.use_encoder:
            prompt_embeddings = self.encoder_model(prompt_embeddings)
            prompt_embeddings = self.align_matrix(prompt_embeddings)

        return prompt_embeddings
        
    

