import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)
from datasets import Dataset
import pandas as pd
import os
import json
import argparse

def train_simple(language='english'):
    """Entraînement ultra simple"""
    
    print(f"\n{'='*50}")
    print(f"🚀 Entraînement du modèle: {language.upper()}")
    print(f"{'='*50}\n")
    
    # Utiliser les vrais datasets
    if language == 'english':
        model_name = 'facebook/bart-base'
        # Charger le vrai SAMSum (fonctionne !)
        from datasets import load_dataset
        print("📥 Chargement du VRAI SAMSum...")
        dataset = load_dataset("knkarthick/samsum")
        text_field = 'dialogue'
        summary_field = 'summary'
        print(f"✅ SAMSum: {len(dataset['train'])} train, {len(dataset['test'])} test")
        
    else:  # french
        model_name = 'facebook/bart-base'  # On utilise BART pour éviter les bugs
        # Utiliser MLSUM en français (qui fonctionne)
        from datasets import load_dataset
        print("📥 Chargement du VRAI MLSUM (français)...")
        dataset = load_dataset("mlsum", "fr")
        text_field = 'text'
        summary_field = 'summary'
        print(f"✅ MLSUM: {len(dataset['train'])} train, {len(dataset['test'])} test")
    
    # Tokenizer et modèle
    print(f"📥 Chargement du tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    # Prétraitement
    def preprocess(examples):
        inputs = tokenizer(
            examples[text_field],
            max_length=256,
            truncation=True,
            padding='max_length'
        )
        labels = tokenizer(
            examples[summary_field],
            max_length=64,
            truncation=True,
            padding='max_length'
        )
        inputs['labels'] = labels['input_ids']
        return inputs
    
    # Pour MLSUM, il y a déjà un split test
    if 'test' not in dataset:
        dataset = dataset["train"].train_test_split(test_size=0.1, seed=42)
    
    # Tokenization
    columns_to_remove = [col for col in dataset['train'].column_names if col not in ['input_ids', 'attention_mask', 'labels']]
    tokenized = dataset.map(preprocess, batched=True, remove_columns=columns_to_remove)
    
    # Arguments d'entraînement - CORRECTION ICI !
    output_dir = f"./models/{language}_finetuned"
    
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",  # ← Changé de evaluation_strategy à eval_strategy
        save_strategy="epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=3,
        predict_with_generate=True,
        fp16=torch.cuda.is_available(),
        report_to="none",
        logging_steps=50,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )
    
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
    )
    
    print("🚀 Début de l'entraînement...")
    trainer.train()
    
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"✅ Modèle sauvegardé dans: {output_dir}")
    
    # Évaluation rapide
    test_results = trainer.evaluate(tokenized["test"])
    print(f"Test Loss: {test_results['eval_loss']:.4f}")
    
    return trainer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--language', choices=['english', 'french'], default='english')
    args = parser.parse_args()
    train_simple(args.language)
