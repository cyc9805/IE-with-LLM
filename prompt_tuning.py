from peft.tuners import PromptEmbedding, PromptTuningConfig, PromptTuningInit
# from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaModel, LlamaConfig, LlamaForCausalLM
# import torch
# import torch.nn as nn
# import math

# class PromptTuningInitForIe(PromptTuningInit):
#     pass

# class PromptTuningConfigForIe(PromptTuningConfig):
#     pass

# class PromptEmbeddingForIe(nn.Module):

#     def __init__(self, config, word_embeddings):
#         super().__init__()
#         total_virtual_tokens = config.num_virtual_tokens * config.num_transformer_submodules
#         self.align_matrix = nn.Linear(config.encoder_token_dim, config.token_dim)
#         self.embedding = nn.Embedding(total_virtual_tokens, config.encoder_token_dim)
#         self.config = config
#         if config.set_encoder and config.prompt_tuning_init == PromptTuningInit.TEXT and not config.inference_mode:
#             assert config.encoder_token_dim and config.encoder_tokenizer_name_or_path
#             from transformers import AutoTokenizer
#             tokenizer_kwargs = config.tokenizer_kwargs or {}
#             tokenizer = AutoTokenizer.from_pretrained(config.encoder_tokenizer_name_or_path, **tokenizer_kwargs)
#         elif not config.set_encoder and config.prompt_tuning_init == PromptTuningInit.TEXT and not config.inference_mode:
#             self.embedding = nn.Embedding(total_virtual_tokens, config.token_dim)
#             if config.prompt_tuning_init == PromptTuningInit.TEXT and not config.inference_mode:
#                 from transformers import AutoTokenizer
#                 tokenizer_kwargs = config.tokenizer_kwargs or {}
#                 tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name_or_path, **tokenizer_kwargs)
#             init_text = config.prompt_tuning_init_text
#             init_token_ids = tokenizer(init_text)["input_ids"]
#             # Trim or iterate until num_text_tokens matches total_virtual_tokens
#             num_text_tokens = len(init_token_ids)
#             if num_text_tokens > total_virtual_tokens:
#                 init_token_ids = init_token_ids[:total_virtual_tokens]
#             elif num_text_tokens < total_virtual_tokens:
#                 num_reps = math.ceil(total_virtual_tokens / num_text_tokens)
#                 init_token_ids = init_token_ids * num_reps
#             init_token_ids = init_token_ids[:total_virtual_tokens]
#             init_token_ids = torch.LongTensor(init_token_ids).to(word_embeddings.weight.device)

#             word_embedding_weights = word_embeddings(init_token_ids).detach().clone()
#             word_embedding_weights = word_embedding_weights.to(torch.float32)
#             self.embedding.weight = torch.nn.Parameter(word_embedding_weights)

#     def forward(self, indices):
#         # Just get embeddings
#         prompt_embeddings = self.embedding(indices)
#         if self.config.set_encoder:
            

#         return prompt_embeddings
        
    

