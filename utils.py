import numpy as np
import logging
import random
from torch import manual_seed
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from typing import Tuple, Union


def enable_peft_finetuning(model, peft_type):
    """
    Enables finetuning of the LoRA adapter.
    """
    for name, param in model.named_parameters():
        if peft_type in name:
            param.requires_grad = True    


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    logging.info(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )



def noise_data(
    input_text: Union[str, np.array],
    tokenizer: PreTrainedTokenizerBase,
    r_denoising: bool = True,
    r_probability: float = 0.25,
    r_denoising_config: Tuple[Tuple] = ((3, 0.15),),
    s_denoising: bool = True,
    s_probability: float = 0.5,
    s_denoising_config: Tuple = (0.25,),
    x_denoising: bool = True,
    x_probability: float = 0.25,
    x_denoising_config: Tuple[Tuple] = ((32, 0.5), (64, 0.2)),
    label_format: str = "json",
    mask_token: str = None,
    ):
    """
    A copy from https://github.com/theblackcat102/unify-learning-paradigms
    Add noise to the tokenized input based on parameters mu, r, and n.

    Args:
    input_text (list or str): The input sequence.
    tokenizer: The tokenizer object.
    mu (int): The mean span length.
    r (float): The corruption rate.
    n (int): The number of corrupted spans.

    Returns:
    str: The noised tokenized input.
    """

    def _create_sentinel_ids(mask_indice, mask_token, input_ids, label=False):
        """
        Sentinel ids creation given the indices that should be masked.
        The start indices of each mask are replaced by the sentinel ids in increasing
        order. Consecutive mask indices to be deleted are replaced with `-1`.
        """
        start_indice = mask_indice ^ np.roll(mask_indice, 1, axis=-1) * mask_indice     # np.roll shifts the elements of mask_indice by 1 position along the specified axis
        start_indice[0] = mask_indice[0]

        _sentinel_ids = np.where(
            start_indice != 0, np.cumsum(start_indice, axis=-1), start_indice
        )

        sentinel_ids = np.where(
            _sentinel_ids != 0, (len(tokenizer) - _sentinel_ids), 0
        )
        sentinel_ids -= mask_indice ^ start_indice

        if mask_token:
            end_indice = mask_indice ^ np.roll(mask_indice, -1, axis=-1) * mask_indice
            start_indices_to_replace = np.where(_sentinel_ids != 0)[0]
            end_indices_to_replace = np.where(end_indice != 0)[0]
            mask_ids = [tokenizer(mask_token.format(i))['input_ids'][1:] for i in range(sum(start_indice))]
            if not label:
                # Replace values at the found indices with values from c
                for i, values in enumerate(mask_ids):
                    start_idx = start_indices_to_replace[i]
                    end_idx = end_indices_to_replace[i]
                    idx_diff = end_idx-start_idx+1
                    if len(values) > idx_diff:
                        sentinel_ids = np.concatenate([sentinel_ids[:start_idx], values, sentinel_ids[end_idx+1:]])
                        input_ids = np.concatenate([input_ids[:start_idx], values, input_ids[end_idx+1:]]) 
                        start_indices_to_replace = np.array([idx+len(values)-idx_diff for idx in start_indices_to_replace], dtype=np.int32)
                        end_indices_to_replace = np.array([idx+len(values)-idx_diff for idx in end_indices_to_replace], dtype=np.int32)
                    end_idx = start_idx + len(values)
                    if end_idx > len(sentinel_ids)-1:
                        sentinel_ids = np.pad(sentinel_ids, (0, end_idx-len(sentinel_ids)), 'constant')
                    sentinel_ids[start_idx:end_idx] = values
            else:
                for i, values in enumerate(mask_ids):
                    start_idx = start_indices_to_replace[i]
                    end_idx = end_indices_to_replace[i]
                    sentinel_ids = np.concatenate([sentinel_ids[:start_idx], values, sentinel_ids[start_idx+1:]])
                    input_ids = np.concatenate([input_ids[:start_idx], values, input_ids[start_idx+1:]]) 
                    start_indices_to_replace = np.array([idx+len(values)-1 for idx in start_indices_to_replace], dtype=np.int32)
                    end_indices_to_replace = np.array([idx+len(values)-1 for idx in end_indices_to_replace], dtype=np.int32)
                    if end_idx > len(sentinel_ids)-1:
                        sentinel_ids = np.pad(sentinel_ids, (0, end_idx-len(sentinel_ids)), 'constant')

        return sentinel_ids, input_ids
    
    def _filter_input_ids(input_ids, sentinel_ids):
        """
        Puts sentinel mask on `input_ids` and fuse consecutive mask tokens into a single mask token by deleting.
        This will reduce the sequence length from `expanded_inputs_length` to `input_length`.
        """
        if len(input_ids) < len(sentinel_ids):
            input_ids = np.pad(input_ids, (0, len(sentinel_ids)-len(input_ids)), 'constant')
        input_ids_full = np.where(sentinel_ids != 0, sentinel_ids, input_ids)
        # masked tokens coming after sentinel tokens and should be removed
        collapsed_id = input_ids_full[input_ids_full >= 0]

        return collapsed_id
    
    # tokenize text if not tokenized
    if isinstance(input_text, str):
        input_ids = tokenizer(input_text, return_tensors='np')['input_ids'][0, 1:]
    else:
        input_ids = input_text

    task_types = ['r', 's', 'x']
    task_prob = []
    task_prob.append(r_probability if r_denoising else 0.0)
    task_prob.append(s_probability if s_denoising else 0.0)
    task_prob.append(x_probability if x_denoising else 0.0)
    task_type = random.choices(task_types, weights=task_prob, k=1)[0]
    input_length = input_ids.shape[-1]

    if task_type == 's':
        noise = s_denoising_config[0]
        split = int(max(input_length*noise, 2))
        filtered_input_ids = input_ids[:split] 
        filtered_labels = input_ids[split:]

        if mask_token:
            mask_id = tokenizer(mask_token.format(0), return_tensors='np')['input_ids'][0, 1:]
        else:
            mask_id = np.array([len(tokenizer)-1])

        filtered_input_ids = np.concatenate([filtered_input_ids, mask_id])
        filtered_labels = np.concatenate([mask_id, filtered_labels])
        if filtered_labels[-1] != tokenizer.eos_token_id: 
            filtered_labels[-1] = tokenizer.eos_token_id
        num_indice = 1
           
    else:
        config = r_denoising_config if task_type == 'r' else x_denoising_config
        mask_indice = None
        for (mean_span, noise) in config:
            _mask_indice = random_spans_noise_mask(input_length, mean_span, noise)
            
            if mask_indice is None:
                mask_indice = _mask_indice
            else:
                # Why does the or operation done???
                mask_indice = mask_indice | _mask_indice

        labels_mask = ~mask_indice
        input_ids_sentinel, sub_input_ids = _create_sentinel_ids(mask_indice, mask_token, input_ids, False)
        labels_sentinel, sub_labels = _create_sentinel_ids(labels_mask, mask_token, input_ids, True)
        filtered_input_ids = _filter_input_ids(sub_input_ids, input_ids_sentinel)
        filtered_labels = _filter_input_ids(sub_labels, labels_sentinel)
        start_indice = mask_indice ^ np.roll(mask_indice, 1, axis=-1) * mask_indice
        num_indice = sum(start_indice)
    
    input_text = tokenizer.decode(filtered_input_ids)
    label = tokenizer.decode(filtered_labels)

    if label_format == 'json' and mask_token:
        label = modify_str_to_json(label, mask_token, num_indice)

    return input_text, label


def modify_str_to_json(label, mask_token, num_labels):
    label_json = {}
    for i in range(1, num_labels+1):
        key = mask_token.format(num_labels-i)
        label, value = label.split(key)
        label_json[key]=value

    label_json = dict(reversed(list(label_json.items())))
    return label_json
        

def random_spans_noise_mask(length, mean_noise_span_length, noise_density):
    """
    A copy from https://github.com/EleutherAI/oslo/blob/main/oslo/transformers/tasks/data_t5_pretraining.py#L230 (inception)
    This function is copy of `random_spans_helper <https://github.com/google-research/text-to-text-transfer-transformer/blob/84f8bcc14b5f2c03de51bd3587609ba8f6bbd1cd/t5/data/preprocessors.py#L2682>`__ .
    Noise mask consisting of random spans of noise tokens.
    The number of noise tokens and the number of noise spans and non-noise spans
    are determined deterministically as follows:
    num_noise_tokens = round(length * noise_density)
    num_nonnoise_spans = num_noise_spans = round(num_noise_tokens / mean_noise_span_length)
    Spans alternate between non-noise and noise, beginning with non-noise.
    Subject to the above restrictions, all masks are equally likely.
    Args:
        length: an int32 scalar (length of the incoming token sequence)
        noise_density: a float - approximate density of output mask
        mean_noise_span_length: a number
    Returns:
        a boolean tensor with shape [length]
    """

    orig_length = length

    num_noise_tokens = int(np.round(length * noise_density))
    # avoid degeneracy by ensuring positive numbers of noise and nonnoise tokens.
    num_noise_tokens = min(max(num_noise_tokens, 1), length - 1)
    num_noise_spans = int(np.round(num_noise_tokens / mean_noise_span_length))

    # avoid degeneracy by ensuring positive number of noise spans
    num_noise_spans = max(num_noise_spans, 1)
    num_nonnoise_tokens = length - num_noise_tokens

    # pick the lengths of the noise spans and the non-noise spans
    def _random_segmentation(num_items, num_segments):
        """Partition a sequence of items randomly into non-empty segments.
        Args:
            num_items: an integer scalar > 0
            num_segments: an integer scalar in [1, num_items]
        Returns:
            a Tensor with shape [num_segments] containing positive integers that add
            up to num_items
        """
        mask_indices = np.arange(num_items - 1) < (num_segments - 1)
        np.random.shuffle(mask_indices)
        first_in_segment = np.pad(mask_indices, [[1, 0]])   # pad single 0 to leftmost position. If set to [2,2], padding is done for double 0 to the leftmost and rightmost position.
        segment_id = np.cumsum(first_in_segment)
        # count length of sub segments assuming that list is sorted
        _, segment_length = np.unique(segment_id, return_counts=True)
        return segment_length

    noise_span_lengths = _random_segmentation(num_noise_tokens, num_noise_spans)
    nonnoise_span_lengths = _random_segmentation(
        num_nonnoise_tokens, num_noise_spans
    )

    interleaved_span_lengths = np.reshape(
        np.stack([nonnoise_span_lengths, noise_span_lengths], axis=1),
        [num_noise_spans * 2],
    )
    span_starts = np.cumsum(interleaved_span_lengths)[:-1]
    span_start_indicator = np.zeros((length,), dtype=np.int8)
    span_start_indicator[span_starts] = True
    span_num = np.cumsum(span_start_indicator)
    is_noise = np.equal(span_num % 2, 1)

    return is_noise[:orig_length]

def compute_input_and_target_lengths(
        inputs_length, noise_density, mean_noise_span_length
    ):
    """
    A copy of copy from https://github.com/EleutherAI/oslo/blob/main/oslo/transformers/tasks/data_t5_pretraining.py#L76 (shits getting meta)
    This function is copy of `random_spans_helper <https://github.com/google-research/text-to-text-transfer-transformer/blob/84f8bcc14b5f2c03de51bd3587609ba8f6bbd1cd/t5/data/preprocessors.py#L2466>`__ .
    Training parameters to avoid padding with random_spans_noise_mask.
    When training a model with random_spans_noise_mask, we would like to set the other
    training hyperparmeters in a way that avoids padding.
    This function helps us compute these hyperparameters.
    We assume that each noise span in the input is replaced by extra_tokens_per_span_inputs sentinel tokens,
    and each non-noise span in the targets is replaced by extra_tokens_per_span_targets sentinel tokens.
    This function tells us the required number of tokens in the raw example (for split_tokens())
    as well as the length of the encoded targets. Note that this function assumes
    the inputs and targets will have EOS appended and includes that in the reported length.
    Args:
        inputs_length: an integer - desired length of the tokenized inputs sequence
        noise_density: a float
        mean_noise_span_length: a float
    Returns:
        tokens_length: length of original text in tokens
        targets_length: an integer - length in tokens of encoded targets sequence
    """
    def _tokens_length_to_inputs_length_targets_length(tokens_length):
        num_noise_tokens = int(round(tokens_length * noise_density))
        num_nonnoise_tokens = tokens_length - num_noise_tokens
        num_noise_spans = int(round(num_noise_tokens / mean_noise_span_length))
        # inputs contain all nonnoise tokens, sentinels for all noise spans
        # and one EOS token.
        _input_length = num_nonnoise_tokens + num_noise_spans + 1
        _output_length = num_noise_tokens + num_noise_spans + 1
        return _input_length, _output_length


    tokens_length = inputs_length

    while (
        _tokens_length_to_inputs_length_targets_length(tokens_length + 1)[0]
        <= inputs_length
    ):
        tokens_length += 1

    inputs_length, targets_length = _tokens_length_to_inputs_length_targets_length(
        tokens_length
    )

    # minor hack to get the targets length to be equal to inputs length
    # which is more likely to have been set to a nice round number.
    if noise_density == 0.5 and targets_length > inputs_length:
        tokens_length -= 1
        targets_length -= 1
    return tokens_length, targets_length


def set_seed(seed = 42):
    random.seed(seed)
    np.random.seed(seed) 
    manual_seed(seed)
    
