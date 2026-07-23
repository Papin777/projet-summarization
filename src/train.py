import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    EarlyStoppingCallback
)
import os
import argparse
from config import MODELS, DEVICE, MODELS_DIR
from utils import load_and_prepare_data

def train_model(language='english', use_lora=False):
    """Entraîne un modèle sans LoRA"""
    
    config = MODELS[language]
    model_name = config['name']
    dataset_name = config['dataset']
    batch_size = config['batch_size']
    
    print(f"\n{'='*50}")
    print(f"🚀 Entraînement du modèle: {language.upper()}")
    print(f"📊 Dataset: {dataset_name}")
    print(f"🤖 Modèle: {model_name}")
    print(f"{'='*50}\n")
    
    tokenized_dataset, tokenizer = load_and_prepare_data(
        dataset_name, 
        model_name,
        max_length=512,
        max_target=128
    )
    
    print(f"🤖 Chargement du modèle: {model_name}")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    output_dir = os.path.join(MODELS_DIR, f"{language}_finetuned")
    
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.01,
        save_total_limit=2,
        num_train_epochs=2,
        predict_with_generate=True,
        fp16=torch.cuda.is_available(),
        push_to_hub=False,
        report_to="none",
        logging_dir=os.path.join(output_dir, 'logs'),
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        remove_unused_columns=False,
    )
    
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )
    
    print("🚀 Début de l'entraînement...")
    trainer.train()
    
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"✅ Modèle sauvegardé dans: {output_dir}")
    
    print("📊 Évaluation sur le test set...")
    test_results = trainer.evaluate(tokenized_dataset["test"])
    print(f"Test Loss: {test_results['eval_loss']:.4f}")
    
    return trainer, tokenizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--language', choices=['english', 'french'], default='english')
    parser.add_argument('--no-lora', action='store_true')
    args = parser.parse_args()
    
    train_model(args.language, use_lora=False)
