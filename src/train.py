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

def create_demo_data(language='english'):
    """Crée des données de démonstration"""
    
    if language == 'english':
        data = {
            'dialogue': [
                "Amanda: I can't come to the meeting tomorrow. I have a dentist appointment.\nJohn: That's okay. We can reschedule for Thursday.\nAmanda: Thursday works for me. What time?\nJohn: How about 2 PM?\nAmanda: Perfect. See you then.",
                "Sarah: Did you finish the report?\nMike: Almost, I need one more day.\nSarah: Okay, but please send it by Friday.\nMike: Will do!",
                "David: Hi, I'm calling about the job opening.\nEmma: Yes, we're still accepting applications.\nDavid: Great, I'll send my CV today.\nEmma: Perfect, I'll review it.",
                "Tom: Are you coming to the party?\nLisa: I'm not sure, I have a lot of work.\nTom: It's Saturday night, you need a break!\nLisa: Okay, I'll come for a couple of hours.",
                "Anna: Have you seen the new project guidelines?\nMark: Yes, I read them yesterday.\nAnna: Do you think we can finish by Friday?\nMark: If we work together, yes."
            ],
            'summary': [
                "Amanda and John reschedule their meeting to Thursday at 2 PM.",
                "Mike needs one more day to finish the report and will send it by Friday.",
                "David applies for a job and Emma agrees to review his CV.",
                "Lisa agrees to come to the party for a couple of hours.",
                "Anna and Mark discuss project guidelines and plan to finish by Friday."
            ]
        }
        text_field = 'dialogue'
        summary_field = 'summary'
    else:  # french
        data = {
            'text': [
                "Le président français Emmanuel Macron a annoncé aujourd'hui un nouveau plan de relance économique. Ce plan prévoit 100 milliards d'euros d'investissements dans les secteurs de la transition écologique, du numérique et de la formation professionnelle.",
                "La France va investir massivement dans la transition énergétique avec un budget de 50 milliards d'euros dédié à la rénovation des bâtiments.",
                "Le gouvernement a dévoilé un plan de soutien aux entreprises touchées par la crise économique, avec des prêts garantis et des aides directes.",
                "La ministre de l'Éducation nationale a présenté une réforme du baccalauréat visant à réduire les inégalités entre les élèves.",
                "Le secteur automobile français annonce une transition vers l'électrique avec des investissements de 10 milliards d'euros."
            ],
            'summary': [
                "Plan de relance économique de 100 milliards d'euros pour la transition écologique.",
                "Investissement de 50 milliards d'euros pour la transition énergétique.",
                "Plan de soutien aux entreprises avec des prêts garantis.",
                "Réforme du baccalauréat pour réduire les inégalités.",
                "Investissement de 10 milliards d'euros pour la transition automobile."
            ]
        }
        text_field = 'text'
        summary_field = 'summary'
    
    df = pd.DataFrame(data)
    dataset = Dataset.from_pandas(df)
    return dataset.train_test_split(test_size=0.2, seed=42), text_field, summary_field

def train_model(language='english'):
    print(f"\n{'='*50}")
    print(f"🚀 Entraînement du modèle: {language.upper()}")
    print(f"{'='*50}\n")
    
    # Utiliser BART pour les deux langues (stable)
    model_name = 'facebook/bart-base'
    print(f"📥 Utilisation du modèle: {model_name}")
    
    # Créer les données
    dataset, text_field, summary_field = create_demo_data(language)
    print(f"✅ Dataset: {len(dataset['train'])} entraînement, {len(dataset['test'])} test")
    
    # Tokenizer et modèle
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
    
    columns_to_remove = [col for col in dataset['train'].column_names]
    tokenized = dataset.map(preprocess, batched=True, remove_columns=columns_to_remove)
    
    # Entraînement
    output_dir = f"./models/{language}_finetuned"
    
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        num_train_epochs=20,  # Plus d'époques car petit dataset
        predict_with_generate=True,
        fp16=torch.cuda.is_available(),
        report_to="none",
        logging_steps=5,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )
    
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
    )
    
    print("🚀 Début de l'entraînement...")
    trainer.train()
    
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"✅ Modèle sauvegardé dans: {output_dir}")
    
    # Évaluation
    test_results = trainer.evaluate(tokenized["test"])
    print(f"📊 Test Loss: {test_results['eval_loss']:.4f}")
    
    return trainer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--language', choices=['english', 'french'], default='english')
    args = parser.parse_args()
    train_model(args.language)
