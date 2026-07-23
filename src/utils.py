import torch
from datasets import load_dataset
from transformers import AutoTokenizer
import json
import os
from config import DATA_DIR

def load_and_prepare_data(dataset_name, model_name, max_length=512, max_target=128):
    print(f"📚 Chargement du dataset: {dataset_name}")
    
    try:
        if dataset_name == 'samsum':
            dataset = load_dataset("samsum", trust_remote_code=True)
            text_field = 'dialogue'
            summary_field = 'summary'
        elif dataset_name == 'orange_sum':
            print("⚠️ OrangeSum n'est plus disponible. Utilisation de MLSUM.")
            dataset = load_dataset("mlsum", "fr", trust_remote_code=True)
            text_field = 'text'
            summary_field = 'summary'
        elif dataset_name == 'mlsum':
            dataset = load_dataset("mlsum", "fr", trust_remote_code=True)
            text_field = 'text'
            summary_field = 'summary'
        else:
            raise ValueError(f"Dataset {dataset_name} non supporté")
    except Exception as e:
        print(f"❌ Erreur: {e}")
        print("📥 Création d'un dataset de démonstration...")
        from datasets import Dataset
        import pandas as pd
        if dataset_name == 'samsum':
            data = {
                'dialogue': ["Amanda: I can't come to the meeting tomorrow.\nJohn: That's okay. We can reschedule.\nAmanda: Thursday works for me.\nJohn: How about 2 PM?\nAmanda: Perfect.", "Sarah: Did you finish the report?\nMike: Almost, I need one more day.\nSarah: Okay, by Friday.\nMike: Will do!"],
                'summary': ["Amanda and John reschedule their meeting to Thursday at 2 PM.", "Mike needs one more day to finish the report."]
            }
        else:
            data = {
                'text': ["Le président français a annoncé un nouveau plan de relance économique.", "La France investit 100 milliards d'euros dans la transition écologique."],
                'summary': ["Nouveau plan de relance économique.", "Investissement massif dans la transition écologique."]
            }
        df = pd.DataFrame(data)
        dataset = Dataset.from_pandas(df)
        text_field = 'dialogue' if dataset_name == 'samsum' else 'text'
        summary_field = 'summary'
        print("✅ Dataset de démonstration créé")
    
    if 'train' not in dataset:
        dataset = dataset.train_test_split(test_size=0.2)
    
    print(f"✅ Dataset: {len(dataset['train'])} entraînement, {len(dataset['test'])} test")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def preprocess_function(examples):
        inputs = tokenizer(examples[text_field], max_length=max_length, truncation=True, padding='max_length')
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(examples[summary_field], max_length=max_target, truncation=True, padding='max_length')
        inputs['labels'] = labels['input_ids']
        return inputs
    
    columns_to_remove = [col for col in dataset['train'].column_names if col not in ['input_ids', 'attention_mask', 'labels']]
    tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=columns_to_remove)
    return tokenized_dataset, tokenizer

def save_metrics(metrics, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

def load_metrics(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)
