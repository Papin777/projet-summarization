import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    EarlyStoppingCallback
)
from peft import LoraConfig, get_peft_model, TaskType
import os
import argparse
from config import MODELS, DEVICE, MODELS_DIR
from utils import load_and_prepare_data

def train_model(language='english', use_lora=True):
    """Entraîne un modèle"""
    
    config = MODELS[language]
    model_name = config['name']
    dataset_name = config['dataset']
    batch_size = config['batch_size']
    
    tokenized_dataset, tokenizer = load_and_prepare_data(
        dataset_name, 
        model_name,
        max_length=512,
        max_target=128
    )
    
    print(f"🤖 Chargement du modèle: {model_name}")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    if use_lora and DEVICE == 'cuda':
        print("🔧 Application de LoRA")
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=8,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
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
        num_train_epochs=3,
        predict_with_generate=True,
        fp16=torch.cuda.is_available(),
        push_to_hub=False,
        report_to="none",
        logging_dir=os.path.join(output_dir, 'logs'),
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
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
    
    return trainer, tokenizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--language', choices=['english', 'french'], default='english')
    parser.add_argument('--no-lora', action='store_true')
    args = parser.parse_args()
    
    train_model(args.language, use_lora=not args.no_lora)
