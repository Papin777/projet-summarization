import torch
from datasets import load_dataset
from transformers import AutoTokenizer
import json
import os

def load_and_prepare_data(dataset_name, model_name, max_length=512, max_target=128):
    """Charge et prépare les données - VERSION AVEC DATASETS ACCESSIBLES"""
    
    print(f"📚 Chargement du dataset: {dataset_name}")
    
    # ===== DATASET ANGLAIS : SAMSum =====
    if dataset_name == 'samsum':
        print("📥 Chargement de SAMSum (version accessible)...")
        # Utiliser la version "knkarthick/samsum" qui est publique
        dataset = load_dataset("knkarthick/samsum")
        text_field = 'dialogue'
        summary_field = 'summary'
        print(f"✅ SAMSum chargé: {len(dataset['train'])} entraînement, {len(dataset['test'])} test")
    
    # ===== DATASET FRANÇAIS : OrangeSum =====
    elif dataset_name == 'orange_sum':
        print("📥 Chargement de OrangeSum (version accessible)...")
        try:
            # Essayer d'abord via GEM
            dataset = load_dataset("GEM/OrangeSum")
            text_field = 'text'
            summary_field = 'summary'
            print(f"✅ OrangeSum chargé via GEM")
        except:
            try:
                # Alternative : orange_sum
                dataset = load_dataset("orange_sum", "abstract")
                text_field = 'text'
                summary_field = 'summary'
                print(f"✅ OrangeSum chargé via orange_sum")
            except:
                # Dernier recours : version Parquet d'un autre utilisateur
                print("📥 Tentative avec version Parquet alternative...")
                dataset = load_dataset("parquet", data_files="https://huggingface.co/datasets/legacy-datasets/orange_sum/resolve/main/abstract/train.parquet")
                text_field = 'text'
                summary_field = 'summary'
                print(f"✅ OrangeSum chargé via Parquet")
        
        # Si le dataset n'a pas de split test, en créer un
        if 'test' not in dataset:
            dataset = dataset["train"].train_test_split(test_size=0.1, seed=42)
            print("✅ Split train/test créé")
    
    else:
        raise ValueError(f"Dataset {dataset_name} non supporté")
    
    print(f"✅ Dataset: {len(dataset['train'])} entraînement, {len(dataset['test'])} test")
    
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
