import json
import numpy as np
import logging
import re
import random
from collections import defaultdict
from torch import manual_seed
from typing import List, Union
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from torch.nn import functional as F
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union


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


def analyze_raw_data(
    preds: str=None,
    refs: str=None,
    task_name: str=None,
    dataset_name: str=None
):
    analyzed_result = {
        "f1_score": [], 
        "precision":[], 
        "recall": [], 
        "total_precision": 0, 
        "total_recall":0, 
        "micro_f1_score": 0
        }
    total_tp, total_len_pred, total_len_ref = 0, 0, 0

    if task_name == 'conversation_ie':
        analyzed_result = parse_and_compute_metrics_for_conversation_ie(preds, refs, dataset_name)
    
    else:
        for pred, ref in zip(preds, refs):
            metric_result = parse_and_compute_metrics(pred, ref, task_name, dataset_name)
            total_tp += metric_result['tp']
            total_len_pred += metric_result['len_pred']
            total_len_ref += metric_result['len_ref']
            analyzed_result["f1_score"].append(metric_result['f1_score'])
            analyzed_result["precision"].append(metric_result['precision'])
            analyzed_result["recall"].append(metric_result['recall'])
        
        total_metric_result = micro_f1_score(total_tp, total_len_pred, total_len_ref)
        analyzed_result["total_precision"] = total_metric_result['total_precision']
        analyzed_result["total_recall"] = total_metric_result['total_recall']
        analyzed_result['micro_f1_score'] = total_metric_result['micro_f1_score']

    return analyzed_result 


def parse_and_compute_metrics_for_conversation_ie(preds, refs, dataset_name):
    num_samples = set()
    # num_samples = [x['dialog_index'] for x in refs]

    # parsed_preds, parsed_refs = defaultdict(list), defaultdict(list)
    parsed_preds, parsed_refs = list(), list()
    # Parse differently based on dataset
    if dataset_name == 'dialog_re':
        for i, (pred, ref) in enumerate(zip(preds, refs)):
            dialog_index = ref['dialog_index']
            num_samples.add(dialog_index)
            matches = re.findall(r'\{.*?\}', pred)
            parsed_pred = []
            for match in matches:
                try:
                    match = json.loads(match)
                    relation = match['relation']
                    parsed_pred.append(relation)
                except Exception as e:
                    logging.info(f"Error parsing prediction: {e}")
                    parsed_pred.append([])
            parsed_preds.append(parsed_pred)

            if i == len(preds)-1 or ref['input_text'] is not None and refs[i+1]['dialog_index'] != dialog_index:
                input_text = ref['input_text']
                del ref['input_text']
                converted_data = []
                for i in range(len(ref['x'])):
                    entry = {
                        'x': ref['x'][i],
                        'y': ref['y'][i],
                        'r': ref['r'][i],
                        't': ref['t'][i]
                    }
                    converted_data.append(entry)
                parsed_refs.append([input_text, converted_data])

        assert len(list(num_samples)) == len(parsed_refs), f"Number of samples do not match. Expected: {num_samples}, Got: {len(parsed_refs)}"
        result = f1c_score(parsed_preds, parsed_refs)
        
    return result


def parse_and_compute_metrics(pred, ref, task_name, dataset_name):
    # Parse differently based on dataset
    parsed_pred = list()
    if dataset_name in ['dialog_re', 'c4']:
        metric_result_template =  {
            "tp":0, 
            "len_pred":0, 
            "len_ref":0, 
            "precision":0,
            "recall":0,
            "f1_score":0}
        try:
            matches = re.findall(r'\{.*?\}', pred)
            if task_name in ['open_ie', 'closed_ie']:
                for match in matches:
                    match = json.loads(match)
                    relation = match['relation']
                    if relation not in parsed_pred:
                        parsed_pred.append(relation)

            elif task_name in ['denoising']:
                ref = list(ref.values())
                for match in matches:
                    match = json.loads(match)
                    for prediction in match.values():
                        parsed_pred.append(prediction)                
            
            parsed_pred = normalize(parsed_pred, dataset_name)
            ref = normalize(ref, dataset_name)
            result = f1_score(pred=parsed_pred, ref=ref, metric_result_template=metric_result_template)

        except Exception as e:
            logging.info(f"Error parsing prediction: {e}")
            metric_result_template["len_ref"] = len(ref)
            result = metric_result_template

    return result


def normalize(answers:List[str], dataset_name:str=None):
    normalized_answers = []
    if dataset_name=='dialog_re':
        for answer in answers:
            answer = answer.strip().lower().replace('_', '').replace(' ', '')
            normalized_answers.append(answer)
    return normalized_answers


def f1_score(pred, ref, metric_result_template):
    tp, len_pred, len_ref = 0, len(pred), len(ref)
    for relation in pred:
        if relation in ref:
            tp += 1
    
    precision = tp / len_pred
    recall = tp / len_ref

    if not precision and not recall:
        f1_score = 0
    else:
        f1_score = 2 * precision * recall / (precision + recall)

    metric_result_template['tp'] = tp
    metric_result_template['len_pred'] = len_pred
    metric_result_template['len_ref'] = len_ref
    metric_result_template['precision'] = precision
    metric_result_template['recall'] = recall
    metric_result_template['f1_score'] = f1_score

    return metric_result_template


def micro_f1_score(total_tp, total_len_pred, total_len_ref):
    metric_result_template = {"total_precision":0, "total_recall":0, "micro_f1_score":0}

    precision = total_tp / total_len_pred if total_len_pred > 0 else 0
    recall = total_tp / total_len_ref if total_len_ref > 0 else 0

    if not precision and not recall:
        micro_f1_score = 0
    else:
        micro_f1_score = 2 * precision * recall / (precision + recall)

    metric_result_template["total_precision"] = precision
    metric_result_template["total_recall"] = recall
    metric_result_template["micro_f1_score"] = micro_f1_score

    return metric_result_template


def f1c_score(devp, data):
    '''
    A copy from DialogRE https://github.com/nlpdata/dialogre/blob/master/bert/evaluate.py
    '''
    metric_result_template = {"total_precision":0, "total_recall":0, "f1c_score":0}
    index = 0
    precisions = []
    recalls = []
    for i in range(len(data)):
        for j in range(len(data[i][1])):
            correct_sys, all_sys = 0, 0
            correct_gt = 0
            
            x = data[i][1][j]["x"].lower().strip()
            y = data[i][1][j]["y"].lower().strip()
            t = {}
            for k in range(len(data[i][1][j]["r"])):
                if data[i][1][j]["r"][k] != 'unanswerable':
                    t[data[i][1][j]["r"][k]] = data[i][1][j]["t"][k].lower().strip()

            l = set(data[i][1][j]["r"]) - set(['unanswerable'])

            ex, ey = False, False
            et = {}
            for r in range(36):
                et[r] = r not in l

            for k in range(len(data[i][0])):
                o = set(devp[index]) - set(['unanswerable'])
                e = set()
                if x in data[i][0][k].lower():
                    ex = True
                if y in data[i][0][k].lower():
                    ey = True
                if k == len(data[i][0])-1:
                    ex = ey = True
                    for r in range(36):
                        et[r] = True
                for r in range(36):
                    if r in t:
                        if t[r] != "" and t[r] in data[i][0][k].lower():
                            et[r] = True
                    if ex and ey and et[r]:
                        e.add(r)
                correct_sys += len(o & l & e)
                all_sys += len(o & e)
                correct_gt += len(l & e)
                index += 1

            precisions += [correct_sys/all_sys if all_sys != 0 else 1]
            recalls += [correct_sys/correct_gt if correct_gt != 0 else 0]
    
    precision = sum(precisions) / len(precisions)
    recall = sum(recalls) / len(recalls)
    f_1 = 2*precision*recall/(precision+recall) if precision+recall != 0 else 0

    metric_result_template["total_precision"] = precision
    metric_result_template["total_recall"] = recall
    metric_result_template["f1c_score"] = f_1

    return metric_result_template

    
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
            mask_id = len(tokenizer)-1

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
    
