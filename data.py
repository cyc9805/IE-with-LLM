from datasets import Dataset, load_dataset, list_metrics, load_metric, concatenate_datasets
from system_prompts import dialog_re_system_prompt

def dataset_pre_func(data, tokenizer, system_prompt):
    messages = {
        'input_ids': tokenizer.apply_chat_template(
            [{"role": "system", "content": system_prompt},  
             {"role": "user", "content": ' '.join(data['input_ids'])}]),
        'label': data['label']}
    return messages


def prepare_dataset(dataset_name, tokenizer):
    # Load dataset
    # Name of the column for huggingface training should be fixed to 'input_ids' and 'label'
    dataset = load_dataset(dataset_name)

    if dataset_name == 'dialog_re':

        dataset = dataset.select_columns(['dialog', 'relation_data']).rename_columns(dict(dialog='input_ids', relation_data='label'))
        dataset = dataset.map(lambda x: dataset_pre_func(x, tokenizer, dialog_re_system_prompt), num_proc=4)
        
        # Combine all dataset to test dataset
        # for i, dataset_type in enumerate(initial_dataset):
        #     encoded_dataset = initial_dataset[dataset_type].map(lambda x: dataset_pre_func(x, tokenizer, dialog_re_system_prompt), num_proc=4)
        #     if i == 0:
        #         dataset = encoded_dataset
        #     else:
        #         dataset = concatenate_datasets([dataset, encoded_dataset])
    return dataset