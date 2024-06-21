import os
import json
import torch
import logging
import numpy as np
from transformers import PreTrainedTokenizer
from dataclasses import dataclass, field
from typing import Any, List, Dict, Tuple
from utils import noise_data, set_seed
from datasets import DatasetDict, load_dataset
from system_prompts import (
    open_dialog_re_system_prompt, 
    closed_dialog_re_system_prompt, 
    denoising_task_prompt,
)

IGNORE_INDEX = -100
SYSTEM_PROMPTS = {
    'open_ie': open_dialog_re_system_prompt,
    'closed_ie': closed_dialog_re_system_prompt,
    'denoising': denoising_task_prompt,
}

DIALOG_TEMPLATE = 'The dialogue input: {}'

QUESTION_DIALOG_TEMPLATE = '''
Subject entity: {}
Object entity: {}
'''

USER_TEMPLATE = DIALOG_TEMPLATE + QUESTION_DIALOG_TEMPLATE

@dataclass
class DenoisingArgument:
    r_denoising: bool = field(default=True, metadata={"help": "Whether to apply r_denoising setting to denoising task"})
    r_probability: float = field(default=0.445, metadata={"help": "Probability of applying r_denoising"})
    r_denoising_config: List[Tuple[int, float]] = field(default=((3, 0.15), (8, 0.15)), metadata={"help": "Configuration of r_denoising"})
    s_denoising: bool = field(default=True, metadata={"help": "Whether to apply s_denoising setting to denoising task"})
    s_probability: float = field(default=0.11, metadata={"help": "Probability of applying s_denoising"})
    s_denoising_config: List[float] = field(default=(0.25,), metadata={"help": "Configuration of s_denoising"})
    x_denoising: bool = field(default=True, metadata={"help": "Whether to apply x_denoising setting to denoising task"})
    x_probability: float = field(default=0.445, metadata={"help": "Probability of applying x_denoising"})
    x_denoising_config: List[Tuple[int, float]] = field(default=((3, 0.5), (8, 0.5), (12, 0.5), (32, 0.15),), metadata={"help": "Configuration of x_denoising"})
    label_format: str = field(default="json", metadata={"help": "Format of the label. JSON is the only supported format"})  #TODO: Add more formats
    mask_token: str = field(default=None, metadata={"help": "Mask token for denoising task. If set to None, it will be set to the last token of the tokenizer vocabulary"})


@dataclass
class DataCollatorWithPaddingForIeLLM:
    tokenizer:Any
    generation_mode:bool=False,
    task: str=None
    prefix_lm_mode: str=None
    evaluation_metrics: List[str]=None

    def __call__(self, encoded_texts):
        batch = {}
        input_ids = [{"input_ids": x["input_ids"]} for x in encoded_texts]
        
        if self.generation_mode:
            self.tokenizer.padding_side = "left"
            padded_input_ids = self.tokenizer.pad(input_ids, return_tensors="pt")
            padded_labels = [x["labels"] for x in encoded_texts]

        else:
            self.tokenizer.padding_side = "right"
            labels = [{"input_ids":x["labels"]} for x in encoded_texts]

            padded_input_ids = self.tokenizer.pad(input_ids, return_tensors="pt")
            padded_labels = self.tokenizer.pad(labels, return_tensors="pt")

            padded_labels = padded_labels["input_ids"].masked_fill(padded_labels.attention_mask.ne(1), IGNORE_INDEX)
        
        input_ids = padded_input_ids["input_ids"]

        if self.prefix_lm_mode is not None:
            # Get position of tokens where prefix attention is applied
            batch['prefix_positions'] = [get_prefix_position(x, self.task, self.prefix_lm_mode, self.tokenizer) for x in input_ids]
        else:
            batch['prefix_positions'] = None
            
        batch["input_ids"] = input_ids
        batch['attention_mask'] = padded_input_ids['attention_mask']
        batch["labels"] = padded_labels

        return batch 
    
    def set_generation_mode(self, generation_mode):
        self.generation_mode = generation_mode


def get_prefix_position(
    x:str, 
    task_name:str, 
    prefix_lm_mode:str, 
    tokenizer:PreTrainedTokenizer,
    )->List[List[int]]:
    
    def _extract_start_and_end_index(src_text, start_tgt_text, end_tgt_text, tokenizer):
        src_text_len = len(src_text)
        start_indicator = tokenizer(start_tgt_text, return_tensors='pt')['input_ids'].squeeze()[1:]
        end_indicator = tokenizer(end_tgt_text, return_tensors='pt')['input_ids'].squeeze()[1:]

        len_start_indicator = start_indicator.shape[0]
        len_end_indicator = end_indicator.shape[0]

        start_index, end_index = 0, src_text_len
        initial_start_index, initial_end_index = start_index, end_index

        for i in range(src_text_len - max(len_start_indicator, len_end_indicator) + 1):
            # Check if the sub-tensor is found
            if torch.all(src_text[i:i+len_start_indicator] == start_indicator):
                start_index = i + len_start_indicator

            if torch.all(src_text[i:i+len_end_indicator] == end_indicator):
                end_index = i
        
        if initial_start_index == start_index and initial_end_index == end_index:
            logging.warning("Prefix attends to all tokens. Unexpected result might occur.")
        
        return start_index, end_index
    
    _strip_template = lambda template: list(map(lambda x: x.strip(), template.split('{}')))        

    start_pos, end_pos = list(), list()
    if prefix_lm_mode in ['only_system_prompt', 'all']:
        start_tgt_text = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>"
        end_tgt_text = "<|eot_id|><|start_header_id|>user<|end_header_id|>"
        start_index, end_index = _extract_start_and_end_index(x, start_tgt_text, end_tgt_text, tokenizer)                    

        start_pos.append(start_index)
        end_pos.append(end_index)

    if prefix_lm_mode in ['only_input_text', 'all']:
        
        start_tgt_text = _strip_template(DIALOG_TEMPLATE)[0]

        if task_name in ['open_ie', 'closed_ie']:
            end_tgt_text = _strip_template(QUESTION_DIALOG_TEMPLATE)[0]
        elif task_name == 'denoising': 
            end_tgt_text = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"

        start_index, end_index = _extract_start_and_end_index(x, start_tgt_text, end_tgt_text, tokenizer)

        start_pos.append(start_index)
        end_pos.append(end_index)
    
    prefix_positions = [start_pos, end_pos]
    return prefix_positions
    
    
def dataset_pre_func(
    batch_data: Dict[str, List[str]],
    tokenizer: PreTrainedTokenizer, 
    task: str, 
    dataset_name: str,
    dataset_type: str,
    denoise_config: DenoisingArgument=None
    )->Dict[str, List[str]]:

    def _create_input_and_label(tokenizer, dataset_type, system_prompt, user_input, labels, preprocessed_input):
        if dataset_type == 'train':
            input_messages =  [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_input}, {"role":"assistant", "content": labels}]
            label_messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_input}]
            input_ids = tokenizer.apply_chat_template(input_messages, add_generation_prompt=False)
            labels = [IGNORE_INDEX] * len(tokenizer.apply_chat_template(label_messages, add_generation_prompt=True)) + tokenizer(labels).input_ids[1:]+[tokenizer.convert_tokens_to_ids("<|eot_id|>")]  
        else:
            input_messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_input}]
            input_ids = tokenizer.apply_chat_template(input_messages, add_generation_prompt=True)

        preprocessed_input['input_ids'].append(input_ids)
        preprocessed_input['labels'].append(labels)

    system_prompt = SYSTEM_PROMPTS[task]
    
    if task == 'denoising':
        denoise_config = denoise_config.__dict__
        mask_token = denoise_config['mask_token']
        if mask_token:
            mask_tokens = [mask_token.format(i) for i in range(4)]
        else:
            mask_tokens = [tokenizer.decode(len(tokenizer)-1-i) for i in range(4)]
        system_prompt = system_prompt.format(*mask_tokens)        

    preprocessed_input = {'input_ids': list(), 'labels': list()}

    # Create dummy inputs for labels if labels is not present in batch_data
    if 'labels' not in batch_data:
        batch_data['labels'] = [{'x': [''], 'y': [''], 'r': ['']} for _ in range(len(batch_data['input_ids']))]
    
    for input_ids, labels in zip(batch_data['input_ids'], batch_data['labels']):
        # For denoising task
        if task == 'denoising':
            if dataset_name == 'c4':
                if isinstance(input_ids, list):
                    input_text = ' \n'.join(list(map(lambda x:x.strip(), input_ids)))
                elif isinstance(input_ids, str):
                    input_text = input_ids
                
                # Split text into segments where size does not exceed split_upper_bound
                split_upper_bound = 1000
                total_input_text_len = len(input_text)
                if total_input_text_len > split_upper_bound:
                    total_splited_input_ids = input_text.split('.')
                    splited_input_ids = list()
                    input_text_len = 0
                    while input_text_len < split_upper_bound:
                        input_text = total_splited_input_ids.pop(0)
                        input_text_len += len(input_text)+1
                        splited_input_ids.append(input_text)
                    input_text = '.'.join(splited_input_ids)+'.'
                    
            # TODO: implement denoising task for DialogRE dataset
            
            iterate_num = 1     

        # For IE tasks
        elif task in ['open_ie', 'closed_ie']:
            joined_input_id = '['+', '.join(list(map(lambda x: '"'+x+'"', input_ids)))+']'
            iterate_num = len(labels['x'])
    
        if dataset_type == 'train':
            for i in range(iterate_num):
                subject = labels['x'][i].strip()
                object = labels['y'][i].strip()
                
                if task == "denoising":
                    joined_input_id, label = noise_data(input_text, tokenizer, **denoise_config)
                    user_input = DIALOG_TEMPLATE.format(joined_input_id)
                    if not isinstance(label, str):
                        label_str = json.dumps(label, ensure_ascii=False)
                    else:
                        label_str = label
                    _create_input_and_label(tokenizer, dataset_type, system_prompt, user_input, label_str, preprocessed_input)
                                    
                elif task in ['open_ie', 'closed_ie']:
                    user_input = USER_TEMPLATE.format(joined_input_id, subject, object)
                    labels_str = json.dumps([{"relation": x} for x in labels['r'][i]], ensure_ascii=False)
                    _create_input_and_label(tokenizer, dataset_type, system_prompt, user_input, labels_str, preprocessed_input)

        else:
            for i in range(iterate_num):
                subject = labels['x'][i].strip()
                object = labels['y'][i].strip()
    
                if task == "denoising":     
                    joined_input_id, label = noise_data(input_text, tokenizer, **denoise_config)
                    user_input = DIALOG_TEMPLATE.format(joined_input_id)
                    label_str = json.dumps(label, ensure_ascii=False)
                    _create_input_and_label(tokenizer, dataset_type, system_prompt, user_input, label_str, preprocessed_input)
                                    
                elif task in ['open_ie', 'closed_ie']:
                    user_input = USER_TEMPLATE.format(joined_input_id, subject, object)
                    labels_str = labels['r'][i]
                    _create_input_and_label(tokenizer, dataset_type, system_prompt, user_input, labels_str, preprocessed_input)

    return preprocessed_input


def prepare_dataset(
    dataset_name:str, 
    tokenizer:str, 
    task:str, 
    cache_file_name:str, 
    seed:int
    )->DatasetDict:
    
    if task == 'denoising':
        denoise_config = DenoisingArgument(**{
            "r_denoising": True,
            "r_probability": 0.475,
            "r_denoising_config": [[3, 0.15]], 
            "s_denoising": True,
            "s_probability": 0.05,
            "s_denoising_config": [0.1],
            "x_denoising": True,
            "x_probability": 0.475,
            "x_denoising_config": [[3, 0.5]],
            "label_format": "json",
            "mask_token": '<mask_id_{}>'}
        )
        logging.info(f"Configuration for denoising task: {denoise_config.__dict__}")
        mask_token = denoise_config.mask_token
        folder_name = mask_token if mask_token is not None else 'special_tokens_as_mask'
    else:
        denoise_config = None
        folder_name = ''
        
    if dataset_name in ['allenai/c4', 'c4']:
        dataset_configs = 'realnewslike'
    else:
        dataset_configs = None

    dataset = load_dataset(dataset_name, dataset_configs)
    cache_file_name = os.path.join(cache_file_name, dataset_name)
    
    train_cache_name = os.path.join(cache_file_name, task, folder_name, 'train.cache')
    validation_cache_name = os.path.join(cache_file_name, task, folder_name, 'validation.cache')
    test_cache_name = os.path.join(cache_file_name, task, folder_name, 'test.cache')

    os.makedirs(os.path.dirname(train_cache_name), exist_ok=True)
    os.makedirs(os.path.dirname(validation_cache_name), exist_ok=True)
    os.makedirs(os.path.dirname(test_cache_name), exist_ok=True)

    cache_file_names = dict(
        train=train_cache_name,
        validation=validation_cache_name,
        test=test_cache_name)
    
    num_proc = 10 if dataset_name == 'dialog_re' else 40
    
    if dataset_name == 'dialog_re':
        dataset = dataset.select_columns(['dialog', 'relation_data']).rename_columns(dict(dialog='input_ids', relation_data='labels'))
    
    elif dataset_name in ['allenai/c4', 'c4']:
        # Sample subset of the dataset  
        set_seed(seed)
        train_dataset = dataset['train']
        random_indices = np.random.choice(len(train_dataset), 500000, replace=True)
        train_dataset = train_dataset.select(random_indices)

        valid_dataset = dataset['validation']
        random_indices = np.random.choice(len(valid_dataset), 2000, replace=True)
        valid_dataset = valid_dataset.select(random_indices)

        dataset = DatasetDict(
            train=train_dataset,
            validation=valid_dataset
        )

        dataset = dataset.select_columns(['url', 'text']).rename_columns(dict(text='input_ids'))

    for dataset_type in dataset:
        dataset[dataset_type] = dataset[dataset_type].map(
            lambda x: dataset_pre_func(
                x, tokenizer, task, dataset_name, dataset_type, denoise_config), 
                batched=True, 
                batch_size=100, 
                num_proc=num_proc, 
                cache_file_name=cache_file_names[dataset_type]
                )

    return dataset
