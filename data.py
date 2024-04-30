import os
import json
from typing import Any
from datasets import Dataset, load_dataset, list_metrics, load_metric, concatenate_datasets
from system_prompts import dialog_re_system_prompt
from dataclasses import dataclass

IGNORE_INDEX = -100

@dataclass
class DataCollatorWithPadding:
    tokenizer:Any
    generation_mode:bool=False

    def __call__(self, encoded_texts):
        batch = {}
        # padding을 하기위해 input_ids을 추출함i
        input_ids = [{"input_ids": x["input_ids"]} for x in encoded_texts]
        
        if self.generation_mode:
            # Generation을 할때는 padding을 left로 실시함
            self.tokenizer.padding_side = "left"

            # padding 실시
            padded_input_ids = self.tokenizer.pad(input_ids, return_tensors="pt")

            padded_labels = [x["label"] for x in encoded_texts]
        else:
            # Training을 할때는 padding을 right로 실시함
            self.tokenizer.padding_side = "right"

            # padding을 실시하기 위해 label을 추출함
            labels = [{"input_ids":x["label"]} for x in encoded_texts]

            # padding 실시
            padded_input_ids = self.tokenizer.pad(input_ids, return_tensors="pt")
            padded_labels = self.tokenizer.pad(labels, return_tensors="pt")

            # Attention mask=0가 있는 부분은 padding된 부분이므로, 해당 부분을 IGNORE_INDEX로 채움
            padded_labels = padded_labels["input_ids"].masked_fill(padded_labels.attention_mask.ne(1), IGNORE_INDEX)
            
        # padding이 실행된 결과를 batch에 저장
        batch["input_ids"] = padded_input_ids["input_ids"]
        batch['attention_mask'] = padded_input_ids['attention_mask']
        batch["labels"] = padded_labels
        return batch
    

def dataset_pre_func(data, tokenizer, system_prompt, dataset_type):
    joined_input_id = ' '.join(data['input_ids'])
    if dataset_type == 'train':
        label = json.dumps(data['label'])
        input_messages =  [{"role": "system", "content": system_prompt}, {"role": "user", "content": joined_input_id}, {"role":"assistant", "content": label}]
        label_messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": joined_input_id}]
        
        input_ids = tokenizer.apply_chat_template(input_messages, add_generation_prompt=False)[:-1]
        label_ids = [IGNORE_INDEX] * (len(tokenizer.apply_chat_template(label_messages, add_generation_prompt=True))-1) + tokenizer(label).input_ids[1:]+[tokenizer.eos_token_id]

    else:
        input_messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": joined_input_id}]
        input_ids = tokenizer.apply_chat_template(input_messages, add_generation_prompt=True)
        label_ids = data['label']
    
    input = {
            'input_ids': input_ids,
            'label': label_ids}
    return input


def prepare_dataset(dataset_name, tokenizer, cache_file_name):
    """Name of the column for huggingface training should be fixed to 'input_ids' and 'label'"""
    dataset = load_dataset(dataset_name)
    cache_file_names = dict(
        train=os.path.join(cache_file_name, 'train.cache'),
        validation=os.path.join(cache_file_name, 'validation.cache'),
        test=os.path.join(cache_file_name, 'test.cache'))
    
    if dataset_name == 'dialog_re':
        dataset = dataset.select_columns(['dialog', 'relation_data']).rename_columns(dict(dialog='input_ids', relation_data='label'))
        for dataset_type in dataset:
            dataset[dataset_type] = dataset[dataset_type].map(lambda x: dataset_pre_func(x, tokenizer, dialog_re_system_prompt, dataset_type), num_proc=4, cache_file_name=cache_file_names[dataset_type])
        
        # Combine all dataset to test dataset
        # for i, dataset_type in enumerate(initial_dataset):
        #     encoded_dataset = initial_dataset[dataset_type].map(lambda x: dataset_pre_func(x, tokenizer, dialog_re_system_prompt), num_proc=4)
        #     if i == 0:
        #         dataset = encoded_dataset
        #     else:
        #         dataset = concatenate_datasets([dataset, encoded_dataset])
    return dataset