import os
import json
from typing import Any
from datasets import Dataset, load_dataset, list_metrics, load_metric, concatenate_datasets
from system_prompts import open_dialog_re_system_prompt, closed_dialog_re_system_prompt
from dataclasses import dataclass

IGNORE_INDEX = -100
SYSTEM_PROMPTS = {
    'open': open_dialog_re_system_prompt,
    'closed': closed_dialog_re_system_prompt
}
USER_TEMPLATE = """
The dialogue input: {}
Subject entity: {}
Object entity: {}
"""

@dataclass
class DataCollatorWithPadding:
    tokenizer:Any
    generation_mode:bool=False

    def __call__(self, encoded_texts):
        batch = {}
        # padding을 하기위해 input_ids을 추출함
        input_ids = [{"input_ids": x["input_ids"]} for x in encoded_texts]
        
        if self.generation_mode:
            # Generation을 할때는 padding을 left로 실시함
            self.tokenizer.padding_side = "left"

            # padding 실시
            padded_input_ids = self.tokenizer.pad(input_ids, return_tensors="pt")

            padded_labels = [x["labels"] for x in encoded_texts]
        else:
            # Training을 할때는 padding을 right로 실시함
            self.tokenizer.padding_side = "right"

            # padding을 실시하기 위해 label을 추출함
            labels = [{"input_ids":x["labels"]} for x in encoded_texts]

            # padding 실시
            padded_input_ids = self.tokenizer.pad(input_ids, return_tensors="pt")
            padded_labels = self.tokenizer.pad(labels, return_tensors="pt")

            # Attention mask=0가 있는 부분은 padding된 부분이므로, 해당 부분을 IGNORE_INDEX로 채움
            # padded_input_ids = padded_input_ids["input_ids"].masked_fill(padded_input_ids.attention_mask.ne(1), IGNORE_INDEX)
            padded_labels = padded_labels["input_ids"].masked_fill(padded_labels.attention_mask.ne(1), IGNORE_INDEX)
            
        # padding이 실행된 결과를 batch에 저장
        batch["input_ids"] = padded_input_ids["input_ids"]
        batch['attention_mask'] = padded_input_ids['attention_mask']
        batch["labels"] = padded_labels
        return batch 


def dialog_re_dataset_pre_func(batch_data, tokenizer, system_prompt, dataset_type):
    preprocessed_input = {'input_ids': [], 'labels': []}
    for input_ids, label in zip(batch_data['input_ids'], batch_data['labels']):
        data = {'input_ids': input_ids, 'labels': label}
        joined_input_id = '['+', '.join(list(map(lambda x: '"'+x+'"', data['input_ids'])))+']'
        labels = data['labels']
        # Evidence과 함께 훈련을 해야 하는데 아직 evidence 없이 훈련함
        if dataset_type == 'train':
            # labels = json.dumps(data['label'])
            for i in range(len(labels['x'])):
                question = USER_TEMPLATE.format(joined_input_id, labels['x'][i], labels['y'][i])
                label = json.dumps([{"relation": x} for x in labels['r'][i]])
                input_messages =  [{"role": "system", "content": system_prompt}, {"role": "user", "content": question}, {"role":"assistant", "content": label}]
                label_messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": question}]
                input_ids = tokenizer.apply_chat_template(input_messages, add_generation_prompt=False)
                label_ids = [IGNORE_INDEX] * len(tokenizer.apply_chat_template(label_messages, add_generation_prompt=True)) + tokenizer(label).input_ids[1:]+[tokenizer.convert_tokens_to_ids("<|eot_id|>")]
                preprocessed_input['input_ids'].append(input_ids)
                preprocessed_input['labels'].append(label_ids)
        else:
            for i in range(len(labels['x'])):
                question = USER_TEMPLATE.format(joined_input_id, labels['x'][i], labels['y'][i])
                input_messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": question}]
                input_ids = tokenizer.apply_chat_template(input_messages, add_generation_prompt=True)
                label_ids = data['labels']['r'][i]
                preprocessed_input['input_ids'].append(input_ids)
                preprocessed_input['labels'].append(label_ids)
        
    return preprocessed_input


def prepare_dataset(dataset_name, tokenizer, ie_setting, cache_file_name):
    """Name of the column for huggingface training should be fixed to 'input_ids' and 'labels'"""
    dataset = load_dataset(dataset_name)
    cache_file_names = dict(
        train=os.path.join(cache_file_name, ie_setting, 'train.cache'),
        validation=os.path.join(cache_file_name, ie_setting, 'validation.cache'),
        test=os.path.join(cache_file_name, ie_setting, 'test.cache'))
    
    if dataset_name == 'dialog_re':
        dataset = dataset.select_columns(['dialog', 'relation_data']).rename_columns(dict(dialog='input_ids', relation_data='labels'))
        system_prompt = SYSTEM_PROMPTS[ie_setting]
        for dataset_type in dataset:
            dataset[dataset_type] = dataset[dataset_type].map(lambda x: dialog_re_dataset_pre_func(x, tokenizer, system_prompt, dataset_type), batched=True, batch_size=100, num_proc=4, cache_file_name=cache_file_names[dataset_type])
        
        # Combine all dataset to test dataset
        # for i, dataset_type in enumerate(initial_dataset):
        #     encoded_dataset = initial_dataset[dataset_type].map(lambda x: dataset_pre_func(x, tokenizer, dialog_re_system_prompt), num_proc=4)
        #     if i == 0:
        #         dataset = encoded_dataset
        #     else:
        #         dataset = concatenate_datasets([dataset, encoded_dataset])
    return dataset
