import os
import json
import torch
import logging
import random
import copy
from typing import Any
from utils import noise_data
from datasets import Dataset, load_dataset
from system_prompts import (
    open_dialog_re_system_prompt, 
    closed_dialog_re_system_prompt, 
    closed_dialog_re_system_prompt_for_table_ie,
    generate_intermediate_output_prompt,
    denoising_task_prompt,
)
from dataclasses import dataclass

##### 이거 나중에 삭제하기 ######
PROMPT_NUM = 1
MERGE_DIALOG_AND_TABLE = True
###############################

IGNORE_INDEX = -100
SYSTEM_PROMPTS = {
    'open_ie': open_dialog_re_system_prompt,
    'closed_ie': closed_dialog_re_system_prompt,
    'denoising': denoising_task_prompt,
    'conversation_ie': closed_dialog_re_system_prompt,
}

DIALOG_TEMPLATE = 'The dialogue input: {}'

QUESTION_DIALOG_TEMPLATE = '''
Subject entity: {}
Object entity: {}
'''

USER_TEMPLATE = DIALOG_TEMPLATE + QUESTION_DIALOG_TEMPLATE


@dataclass
class DataCollatorWithPaddingForIeLLM:

    tokenizer:Any
    generation_mode:bool=False,
    generate_intermediate_output: bool=False
    task: str=None
    prefix_lm_mode: str=None

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

            if self.generate_intermediate_output:
                posterior_input_messages = [x["posterior_input_messages"] for x in encoded_texts]
                questions = [x['question'] for x in encoded_texts]
                input_dialogs = [x['input_dialog'] for x in encoded_texts]
                batch['posterior_input_messages'] = posterior_input_messages
                batch['questions'] = questions
                batch['input_dialogs'] = input_dialogs
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
        
        input_ids = padded_input_ids["input_ids"]
        # padding이 실행된 결과를 batch에 저장
        if self.prefix_lm_mode is not None:
            batch['prefix_positions'] = [get_prefix_position(x, self.task, self.prefix_lm_mode, self.tokenizer) for x in input_ids]
        else:
            batch['prefix_positions'] = None
        batch["input_ids"] = input_ids
        batch['attention_mask'] = padded_input_ids['attention_mask']
        batch["labels"] = padded_labels

        return batch 
    
    def set_generation_mode(self, generation_mode):
        self.generation_mode = generation_mode


def extract_start_and_end_index(src_text, start_tgt_text, end_tgt_text, tokenizer):
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

# all 모든 input 값들이 서로 attend하게 만들기
# 두번째는 주어진 context dialog에만 서로 attend하게 만들기
def get_prefix_position(x, task, prefix_lm_mode, tokenizer):

    strip_template = lambda template: list(map(lambda x: x.strip(), template.split('{}')))        

    start_pos, end_pos = list(), list()
    if prefix_lm_mode in ['only_system_prompt', 'all']:
        # sos_indicator = tokenizer.convert_tokens_to_ids("<|begin_of_text|><|start_header_id|>system<|end_header_id|>")
        # eos_indicator = tokenizer.convert_tokens_to_ids("<|eot_id|><|start_header_id|>user<|end_header_id|>")

        start_tgt_text = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>"
        end_tgt_text = "<|eot_id|><|start_header_id|>user<|end_header_id|>"
        start_index, end_index = extract_start_and_end_index(x, start_tgt_text, end_tgt_text, tokenizer)                    

        start_pos.append(start_index)
        end_pos.append(end_index)


    if prefix_lm_mode in ['only_input_text', 'all']:
        
        start_tgt_text = strip_template(DIALOG_TEMPLATE)[0]

        if task in ['open_ie', 'closed_ie', 'conversation_ie']:
            end_tgt_text = strip_template(QUESTION_DIALOG_TEMPLATE)[0]
        elif task == 'denoising': 
            end_tgt_text = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"

        start_index, end_index = extract_start_and_end_index(x, start_tgt_text, end_tgt_text, tokenizer)

        start_pos.append(start_index)
        end_pos.append(end_index)
        
    return [start_pos, end_pos]

    
def dataset_pre_func(batch_data, indexes, tokenizer, task, dataset_type, generate_intermediate_output, model_for_predefined_intermediate_output, **kwargs):

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
    preprocessed_input = {'input_ids': [], 'labels': []}

    # Create dummy inputs for labels if labels is not present in batch_data
    if 'labels' not in batch_data:
        batch_data['labels'] = [{'x': [''], 'y': [''], 'r': ['']} for _ in range(len(batch_data['input_ids']))]
        
    if generate_intermediate_output and dataset_type != 'train':
        preprocessed_input['posterior_input_messages'] = []
        preprocessed_input['question'] = []
        preprocessed_input['input_dialog'] = []
    
    for index, input_ids, labels in zip(indexes, batch_data['input_ids'], batch_data['labels']):
        data = {'input_ids': input_ids, 'labels': labels}

        # For denoising task
        if task == 'denoising':
            if isinstance(data['input_ids'], list):
                input_text = ' \n'.join(list(map(lambda x:x.strip(), data['input_ids'])))
            elif isinstance(data['input_ids'], str):
                input_text = data['input_ids']
            denoise_config = kwargs['denoise_config']
            iterate_num = 1

        # For IE tasks
        elif task in ['open_ie', 'closed_ie']:
            joined_input_id = '['+', '.join(list(map(lambda x: '"'+x+'"', data['input_ids'])))+']'
            labels = data['labels']
            iterate_num = len(labels['x'])

            ##################### using intermediate result ####################
            if model_for_predefined_intermediate_output:
                def extract_lines_after_llm_rtes(file_path):
                    with open(file_path, 'r') as file:
                        lines = file.readlines()

                    llm_rtes_section = False
                    extracted_lines = []

                    for line in lines:
                        if llm_rtes_section:
                            extracted_lines.append(line.strip())
                        if '[LLM RTEs]' in line:
                            llm_rtes_section = True

                    return extracted_lines

                file_path = '/data/yongchan/ie_llm/dataset/DialogRE/closed_ie/other_models'
                
                ################ Use setting below ################
                model_to_file_name = {'mistral': 'mistral7B', 'llama3':'llama3-8B', 'gpt3.5-turbo':'gpt', 'mistral_seq':'Mistral-7B-Instruct-v0.2'}
                auxiliary_system_prompt_for_table_ie = {
                    "IE" : "|step|predicate|subject type|subject|object type|object|",
                    1: "|Speaker #|RELATION_TYPE|Speaker #|",
                    2: "|Speaker #|relation_type|Object|",
                    3: "|PERSON|RELATION_TYPE|PERSON|REASON|"
                }
                ####################################################
                
                input_file_path = os.path.join(file_path, model_for_predefined_intermediate_output, f'dev.json_{model_to_file_name[model_for_predefined_intermediate_output]}_{PROMPT_NUM}', f'{index}.tableie.out')
                table_input_id = extract_lines_after_llm_rtes(input_file_path)

                TABLE_FORMAT = '\nSummarized table of relationship between speakers: {}'
                table_input_id = TABLE_FORMAT.format(table_input_id)

                system_prompt = closed_dialog_re_system_prompt_for_table_ie.format(auxiliary_system_prompt_for_table_ie[PROMPT_NUM])
            ################################################################################
        
        elif task == 'conversation_ie':
            joined_input_ids = []
            stacked_input_ids = []
            for input_id in data['input_ids']:
                input_id = '"'+input_id+'"'
                stacked_input_ids.append(input_id)
                joined_input_ids.append('['+', '.join(stacked_input_ids)+']')
            labels = data['labels']
            iterate_num = len(labels['x'])  

        if dataset_type == 'train':
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
                    labels_str = json.dumps([{"relation": x} for x in labels['r'][i]], ensure_ascii=False)
                    _create_input_and_label(tokenizer, dataset_type, system_prompt, user_input, labels_str, preprocessed_input)
               
                elif task == "conversation_ie":
                    for joined_input_id in joined_input_ids:
                        user_input = USER_TEMPLATE.format(joined_input_id, subject, object)
                        temp_labels = list()
                        for j, trigger_word in enumerate(labels['t'][i]):
                            if trigger_word in joined_input_id and subject in joined_input_id and object in joined_input_id:
                                label = labels['r'][i][j]
                            else:
                                label = 'unanswerable'
                            temp_labels.append(label)
                        labels_str = json.dumps([{"relation": x} for x in list(temp_labels)], ensure_ascii=False)
                        _create_input_and_label(tokenizer, dataset_type, system_prompt, user_input, labels_str, preprocessed_input)

        else:
            for i in range(iterate_num):
                subject = labels['x'][i].strip()
                object = labels['y'][i].strip()
                if generate_intermediate_output:
                    input_dialog = DIALOG_TEMPLATE
                    question = QUESTION_DIALOG_TEMPLATE.format(subject, object)
                    input_messages = [{"role": "system", "content": generate_intermediate_output_prompt}, {"role": "user", "content": joined_input_id}]
                    input_ids = tokenizer.apply_chat_template(input_messages, add_generation_prompt=True)
                    posterior_input_messages = [{"role": "system", "content": system_prompt}]
                    preprocessed_input['posterior_input_messages'].append(posterior_input_messages)
                    preprocessed_input['question'].append(question)
                    preprocessed_input['input_dialog'].append(input_dialog)
                else:
                    if task == "denoising":
                        joined_input_id, label = noise_data(input_text, tokenizer, **denoise_config)
                        label_str = json.dumps(label, ensure_ascii=False)
                        user_input = DIALOG_TEMPLATE.format(joined_input_id)
                        _create_input_and_label(tokenizer, dataset_type, system_prompt, user_input, label_str, preprocessed_input)
                                        
                    elif task in ['open_ie', 'closed_ie']:
                        user_input = USER_TEMPLATE.format(joined_input_id, subject, object)
                        labels_str = data['labels']['r'][i]
                        if model_for_predefined_intermediate_output:
                            if MERGE_DIALOG_AND_TABLE:
                                full_dialog = DIALOG_TEMPLATE.format(joined_input_id)
                                question_input = QUESTION_DIALOG_TEMPLATE.format(subject, object)
                                user_input = full_dialog + table_input_id + question_input
                            else:
                                user_input = table_input_id
                        _create_input_and_label(tokenizer, dataset_type, system_prompt, user_input, labels_str, preprocessed_input)

                    elif task == 'conversation_ie':
                        for j, joined_input_id in enumerate(joined_input_ids):
                            user_input = USER_TEMPLATE.format(joined_input_id, subject, object)
                            relations = list()
                            trigger_words = list()
                            for k, trigger_word in enumerate(labels['t'][i]):
                                if trigger_word in joined_input_id and subject in joined_input_id and object in joined_input_id:
                                    label = labels['r'][i][k]
                                    trigger_words.append(trigger_word)
                                else: 
                                    label = 'unanswerable'
                                    trigger_words.append('')
                                relations.append(label)
                            if j == len(joined_input_ids)-1:
                                temp_labels = copy.deepcopy(labels)
                                temp_labels['dialog_index'] = index
                                temp_labels['input_text'] = data['input_ids']
                            else:
                                temp_labels = {'dialog_index':index}
                            temp_labels['relations'] = relations
                            temp_labels['trigger_words'] = trigger_words
                            _create_input_and_label(tokenizer, dataset_type, system_prompt, user_input, temp_labels, preprocessed_input)

    return preprocessed_input


def prepare_dataset(dataset_name, tokenizer, task, generate_intermediate_output, model_for_predefined_intermediate_output, cache_file_name, **kwargs):
    """Name of the column for huggingface training should be fixed to 'input_ids' and 'labels'"""

    if task == 'denoising' and generate_intermediate_output:
        raise ValueError("Intermediate output generation is not supported for denoising task.")
    
    if generate_intermediate_output and model_for_predefined_intermediate_output:
        raise ValueError("Cannot generate intermediate output if the model for predefined intermediate output is provided.")
    
    if dataset_name in ['allenai/c4', 'c4']:
        configs = 'realnewslike'
    else:
        configs = None
    dataset = load_dataset(dataset_name, configs)
    cache_file_name = os.path.join(cache_file_name, dataset_name)
    folder_name = "intermediate_result" if generate_intermediate_output else "direct_result"

    if model_for_predefined_intermediate_output:
        folder_name = os.path.join(model_for_predefined_intermediate_output, str(PROMPT_NUM), 'merged' if MERGE_DIALOG_AND_TABLE else 'only_table')
    
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
    
    if dataset_name == 'dialog_re':
        dataset = dataset.select_columns(['dialog', 'relation_data']).rename_columns(dict(dialog='input_ids', relation_data='labels'))
    
    if dataset_name == 'c4':
        dataset = dataset.select_columns(['url', 'text']).rename_columns(dict(text='input_ids'))

    for dataset_type in dataset:
        if model_for_predefined_intermediate_output is None or dataset_type == 'test':
            dataset[dataset_type] = dataset[dataset_type].map(
                lambda x, indexes: dataset_pre_func(
                    x, indexes, tokenizer, task, dataset_type, generate_intermediate_output, model_for_predefined_intermediate_output, **kwargs), 
                    with_indices=True,
                    batched=True, 
                    batch_size=100, 
                    num_proc=8, 
                    cache_file_name=cache_file_names[dataset_type]
                    )

    return dataset
