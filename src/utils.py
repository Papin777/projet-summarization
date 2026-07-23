import torch
from datasets import load_dataset
from transformers import AutoTokenizer
import json
import os

def load_and_prepare_data(dataset_name, model_name, max_length=512, max_target=128):
    """Charge et prépare les données"""
    
    print(f"📚 Chargement du dataset: {dataset_name}")
    
    if dataset_name == 'samsum':
        dataset = load_dataset('samsum')
        text_field = 'dialogue'
        summary_field = 'summary'
    elif dataset_name == 'orange_sum':
        dataset = load_dataset('orange_sum', 'abstract')
        text_field = 'text'
        summary_field = 'summary'
    elif dataset_name == 'mlsum':
        dataset = load_dataset('mlsum', 'fr')
        text_field = 'text'
        summary_field = 'summary'
    else:
        raise ValueError(f"Dataset {dataset_name} non supporté")
    
    print(f"✅ Dataset chargé: {len(dataset['train'])} entraînement, {len(dataset['test'])} test")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def preprocess_function(examples):
        inputs = tokenizer(
            examples[text_field],
            max_length=max_length,
            truncation=True,
            padding='max_length'
        )
        
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                examples[summary_field],
                max_length=max_target,
                truncation=True,
                padding='max_length'
            )
        
        inputs['labels'] = labels['input_ids']
        return inputs
    
    tokenized_dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset['train'].column_names
    )
    
    return tokenized_dataset, tokenizer

def save_metrics(metrics, filename):
    with open(filename, 'w') as f:
        json.dump(metrics, f, indent=2)

def load_metrics(filename):
    with open(filename, 'r') as f:
        return json.load(f)
