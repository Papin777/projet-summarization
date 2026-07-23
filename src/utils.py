import torch
from datasets import load_dataset
from transformers import AutoTokenizer
import json
import os
from config import DATA_DIR

def load_and_prepare_data(dataset_name, model_name, max_length=512, max_target=128):
    """Charge et prépare les données avec OrangeSum pour le français"""
    
    print(f"📚 Chargement du dataset: {dataset_name}")
    
    if dataset_name == 'orange_sum':
        try:
            print("📥 Chargement de OrangeSum via GEM...")
            dataset = load_dataset('GEM/OrangeSum')
            text_field = 'text'
            summary_field = 'summary'
            print("✅ OrangeSum chargé avec succès")
        except Exception as e:
            print(f"❌ Erreur: {e}")
            print("📥 Création d'un dataset de démonstration...")
            from datasets import Dataset
            import pandas as pd
            data = {
                'text': [
                    "Le président français a annoncé un nouveau plan de relance économique de 100 milliards d'euros.",
                    "La France investit massivement dans la transition écologique et le numérique."
                ],
                'summary': [
                    "Plan de relance économique de 100 milliards d'euros.",
                    "Investissement dans la transition écologique et le numérique."
                ]
            }
            df = pd.DataFrame(data)
            dataset = Dataset.from_pandas(df)
            text_field = 'text'
            summary_field = 'summary'
            print("✅ Dataset de démonstration créé")
    
    elif dataset_name == 'samsum':
        try:
            print("📥 Chargement de SAMSum via knkarthick/samsum...")
            dataset = load_dataset("knkarthick/samsum")
            text_field = 'dialogue'
            summary_field = 'summary'
            print("✅ SAMSum chargé avec succès")
        except Exception as e:
            print(f"❌ Erreur: {e}")
            print("📥 Création d'un dataset de démonstration...")
            from datasets import Dataset
            import pandas as pd
            data = {
                'dialogue': [
                    "Amanda: I can't come to the meeting tomorrow.\nJohn: That's okay. We can reschedule.\nAmanda: Thursday works for me.\nJohn: How about 2 PM?\nAmanda: Perfect.",
                    "Sarah: Did you finish the report?\nMike: Almost, I need one more day.\nSarah: Okay, by Friday.\nMike: Will do!"
                ],
                'summary': [
                    "Amanda and John reschedule their meeting to Thursday at 2 PM.",
                    "Mike needs one more day to finish the report."
                ]
            }
            df = pd.DataFrame(data)
            dataset = Dataset.from_pandas(df)
            text_field = 'dialogue'
            summary_field = 'summary'
            print("✅ Dataset de démonstration créé")
    
    else:
        raise ValueError(f"Dataset {dataset_name} non supporté")
    
    if 'train' not in dataset:
        dataset = dataset.train_test_split(test_size=0.2)
    
    print(f"✅ Dataset: {len(dataset['train'])} entraînement, {len(dataset['test'])} test")
    
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
    
    train_cols = dataset['train'].column_names
    columns_to_remove = [col for col in train_cols if col not in ['input_ids', 'attention_mask', 'labels']]
    
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
