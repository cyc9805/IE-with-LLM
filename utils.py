import json
import numpy as np
import logging
import re
import random
from torch import manual_seed
from typing import List, Union


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
    dataset_name: str=None
):
    analyzed_result = {"f1_score": [], "micro_f1_score": 0}
    total_tp, total_len_pred, total_len_ref = 0, 0, 0
    for pred, ref in zip(preds, refs):
        metric_result = parse_and_compute_metrics(pred, ref, dataset_name)
        total_tp += metric_result['tp']
        total_len_pred += metric_result['len_pred']
        total_len_ref += metric_result['len_ref']
        analyzed_result["f1_score"].append(metric_result['f1_score'])
    
    analyzed_result['micro_f1_score'] = micro_f1_score(total_tp, total_len_pred, total_len_ref)

    return analyzed_result


def parse_and_compute_metrics(pred, ref, dataset_name):
    # Parse differently based on dataset
    parsed_pred = list()
    if dataset_name == 'dialog_re':
        metric_result_template =  {"tp":0, "len_pred":0, "len_ref":0, "f1_score":0}
        try:
            matches = re.findall(r'\{.*?\}', pred)
            for match in matches:
                match = json.loads(match)['relation']
                if match not in parsed_pred:
                    parsed_pred.append(match)
            # assert all([len(pred[key])==len(pred[list(pred.keys())[0]]) for key in pred])
            parsed_pred = normalize_relation(parsed_pred, dataset_name)
            ref = normalize_relation(ref, dataset_name)
            result = f1_score(pred=parsed_pred, ref=ref, metric_result_template=metric_result_template)

        except Exception as e:
            logging.info(f"Error parsing prediction: {e}")
            metric_result_template["len_ref"] = len(ref)
            result = metric_result_template

    return result


def normalize_relation(relations:List[str], dataset_name:str=None):
    normalized_relations = []
    if dataset_name=='dialog_re':
        for relation in relations:
            relation = relation.strip().lower().replace('_', '').replace(' ', '')
            normalized_relations.append(relation)
    return normalized_relations


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
    metric_result_template['f1_score'] = f1_score

    return metric_result_template


def micro_f1_score(total_tp, total_len_pred, total_len_ref):
    precision = total_tp / total_len_pred if total_len_pred > 0 else 0
    recall = total_tp / total_len_ref if total_len_ref > 0 else 0

    if not precision and not recall:
        micro_f1_score = 0
    else:
        micro_f1_score = 2 * precision * recall / (precision + recall)

    return micro_f1_score

def set_seed(seed = 42):
    random.seed(seed)
    np.random.seed(seed) 
    manual_seed(seed)
    
