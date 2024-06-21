import logging
import json
import re
from typing import List, Dict, Any

def analyze_raw_data(
    preds: List[str]=None,
    refs: List[str]=None,
    task_name: str=None,
):
    overall_result = {
        "f1_score": list(), 
        "precision":list(), 
        "recall": list(), 
        "total_precision": 0, 
        "total_recall":0, 
        "micro_f1_score": 0
        }
    
    total_tp, total_len_pred, total_len_ref = 0, 0, 0
    
    for pred, ref in zip(preds, refs):
        individual_result = parse_and_compute_metrics(pred, ref, task_name)   
        total_tp += individual_result['tp']
        total_len_pred += individual_result['len_pred']
        total_len_ref += individual_result['len_ref']
        
        overall_result["f1_score"].append(individual_result['f1_score'])
        overall_result["precision"].append(individual_result['precision'])
        overall_result["recall"].append(individual_result['recall'])
    
    overall_result = micro_f1_score(total_tp, total_len_pred, total_len_ref, overall_result)

    return overall_result 



def parse_and_compute_metrics(
    pred:str, 
    ref:str, 
    task_name:str
    ):
    
    def _normalize(texts:List[str])->List[str]:
        normalized_texts = []
        for text in texts:
            text = text.strip().lower().replace('_', '').replace(' ', '')
            normalized_texts.append(text)
        return normalized_texts
    
    # Parse differently based on dataset
    parsed_pred = list()
    individual_result =  {
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
        
        parsed_pred = _normalize(parsed_pred)
        ref = _normalize(ref)
        result = f1_score(pred=parsed_pred, ref=ref, individual_result=individual_result)

    except Exception as e:
        logging.info(f"Error parsing prediction: {e}")
        individual_result["len_ref"] = len(ref)
        result = individual_result

    return result


def f1_score(
    pred: List[str], 
    ref: List[str], 
    individual_result: Dict[str, int],
    )->Dict[str, int]:
    
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

    individual_result['tp'] = tp
    individual_result['len_pred'] = len_pred
    individual_result['len_ref'] = len_ref
    individual_result['precision'] = precision
    individual_result['recall'] = recall
    individual_result['f1_score'] = f1_score

    return individual_result


def micro_f1_score(
    total_tp: int, 
    total_len_pred: int, 
    total_len_ref: int, 
    overall_result: Dict[str, Any]
    )->Dict[str, Any]:
    
    precision = total_tp / total_len_pred if total_len_pred > 0 else 0
    recall = total_tp / total_len_ref if total_len_ref > 0 else 0

    if not precision and not recall:
        micro_f1_score = 0
    else:
        micro_f1_score = 2 * precision * recall / (precision + recall)

    overall_result["total_precision"] = precision
    overall_result["total_recall"] = recall
    overall_result["micro_f1_score"] = micro_f1_score

    return overall_result