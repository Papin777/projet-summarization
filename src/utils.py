import torch
from datasets import load_dataset
from transformers import AutoTokenizer
import json
import os

def load_and_prepare_data(dataset_name, model_name, max_length=512, max_target=128):
    """Charge et prépare les données - VERSION AVEC VRAIS DATASETS"""
    
    print(f"📚 Chargement du dataset: {dataset_name}")
    
    # ===== VRAI DATASET SAMSum (anglais) =====
    if dataset_name == 'samsum':
        print("📥 Chargement du VRAI dataset SAMSum (format Parquet)...")
        dataset = load_dataset("parquet", data_files={
            "train": "hf://datasets/Samsung/samsum/samsum/samsum-train.parquet",
            "test": "hf://datasets/Samsung/samsum/samsum/samsum-test.parquet",
            "validation": "hf://datasets/Samsung/samsum/samsum/samsum-validation.parquet"
        })
        text_field = 'dialogue'
        summary_field = 'summary'
        print(f"✅ SAMSum chargé: {len(dataset['train'])} entraînement, {len(dataset['test'])} test")
    
    # ===== VRAI DATASET OrangeSum (français) =====
    elif dataset_name == 'orange_sum':
        print("📥 Chargement du VRAI dataset OrangeSum (format Parquet)...")
        # Charger le fichier Parquet
        dataset = load_dataset("parquet", data_files="hf://datasets/krm/modified-orangeSum/data/train-00000-of-00001-f75513f3be8c97f0.parquet")
        # Diviser en train/test (80/20)
        dataset = dataset["train"].train_test_split(test_size=0.2, seed=42)
        text_field = 'text'
        summary_field = 'summary'
        print(f"✅ OrangeSum chargé: {len(dataset['train'])} entraînement, {len(dataset['test'])} test")
    
    else:
        raise ValueError(f"Dataset {dataset_name} non supporté")
    
    # Tokenizer
    print(f"📥 Chargement du tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def preprocess_function(examples):
        inputs = tokenizer(
            examples[text_field],
            max_length=max_length,
            truncation=True,
            padding='max_length'
        )
        labels = tokenizer(
            examples[summary_field],
            max_length=max_target,
            truncation=True,
            padding='max_length'
        )
        inputs['labels'] = labels['input_ids']
        return inputs
    
    # Prétraitement
    columns_to_remove = [col for col in dataset['train'].column_names if col not in ['input_ids', 'attention_mask', 'labels']]
    
    tokenized_dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=columns_to_remove
    )
    
    print("✅ Prétraitement terminé")
    return tokenized_dataset, tokenizer

def save_metrics(metrics, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

def load_metrics(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)
