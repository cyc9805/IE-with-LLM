import json
import numpy as np
import logging
from typing import List, Union

    
def analyze_raw_data(
    preds: str=None,
    refs: str=None,
    dataset_name: str=None
):
    anlayzed_result = {"metric_scores": [], "avg_metric_score": None}
    for pred, ref in zip(preds, refs):
        anlayzed_result["metric_scores"].append(parse_and_compute_metrics(pred, ref, dataset_name))
    
    anlayzed_result["avg_metric_score"] = np.mean(anlayzed_result["metric_scores"])

    return anlayzed_result


def parse_and_compute_metrics(pred, ref, dataset_name):
    # Parse differently based on dataset
    if dataset_name == 'dialog_re':
        try:
            pred = pred[pred.find('{'):]
            pred = pred[:pred.rfind('}')+1]
            pred = json.loads(pred)
            # assert all([len(pred[key])==len(pred[list(pred.keys())[0]]) for key in pred])
            pred = normalize_relation(pred['x'], pred['y'], pred['rid'], dataset_name)
            ref = normalize_relation(ref['x'], ref['y'], ref['rid'], dataset_name)
            metric_score = f1_score(pred=pred, ref=ref)

        except Exception as e:
            logging.info(f"Error parsing prediction: {e}")
            metric_score = 0

    return metric_score


def normalize_relation(pred_x: List[str], pred_y: List[str], rid:List[Union[int, str]], dataset_name:str=None):
    normalized_relation = []
    if dataset_name=='dialog_re':
        for x, y, rids in zip(pred_x, pred_y, rid):
            x = x.strip().lower().replace(' ', '')
            y = y.strip().lower().replace(' ', '')
            for rid in rids:
                relation = str(rid)+''.join(sorted([x, y], key=lambda x: x[0]))
                normalized_relation.append(relation)
    return normalized_relation


def f1_score(pred, ref):
    tp = 0
    for relation in pred:
        if relation in ref:
            tp += 1
    
    precision = tp / len(pred)
    recall = tp / len(ref)

    if not precision and not recall:
        f1 = 0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return f1