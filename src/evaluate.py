import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_dataset
from rouge_score import rouge_scorer
import numpy as np
import json
import os
import argparse
from config import MODELS_DIR

def compute_rouge(predictions, references):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = []
    
    for pred, ref in zip(predictions, references):
        score = scorer.score(ref, pred)
        scores.append(score)
    
    avg_scores = {
        'rouge1': np.mean([s['rouge1'].fmeasure for s in scores]),
        'rouge2': np.mean([s['rouge2'].fmeasure for s in scores]),
        'rougeL': np.mean([s['rougeL'].fmeasure for s in scores])
    }
    return avg_scores

def evaluate_model(model_path, dataset_name):
    print(f"📊 Évaluation du modèle: {model_path}")
    
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.eval()
    
    if dataset_name == 'samsum':
        dataset = load_dataset('samsum', split='test')
        text_field = 'dialogue'
        summary_field = 'summary'
    elif dataset_name == 'orange_sum':
        dataset = load_dataset('orange_sum', 'abstract', split='test')
        text_field = 'text'
        summary_field = 'summary'
    else:
        raise ValueError(f"Dataset {dataset_name} non supporté")
    
    predictions = []
    references = []
    
    print(f"📝 Génération de {len(dataset)} résumés...")
    
    for i, example in enumerate(dataset):
        if i % 100 == 0:
            print(f"Progression: {i}/{len(dataset)}")
        
        inputs = tokenizer(
            example[text_field],
            return_tensors='pt',
            max_length=512,
            truncation=True
        ).to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=128,
                min_length=30,
                num_beams=4,
                early_stopping=True
            )
        
        pred = tokenizer.decode(outputs[0], skip_special_tokens=True)
        ref = example[summary_field]
        
        predictions.append(pred)
        references.append(ref)
    
    rouge_scores = compute_rouge(predictions, references)
    
    print(f"\n📈 Scores ROUGE pour {dataset_name}:")
    print(f"ROUGE-1: {rouge_scores['rouge1']:.4f}")
    print(f"ROUGE-2: {rouge_scores['rouge2']:.4f}")
    print(f"ROUGE-L: {rouge_scores['rougeL']:.4f}")
    
    results = {
        'dataset': dataset_name,
        'model_path': model_path,
        'rouge_scores': rouge_scores,
        'num_examples': len(dataset)
    }
    
    with open(f'{model_path}/evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return rouge_scores

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='Chemin du modèle')
    parser.add_argument('--dataset', required=True, choices=['samsum', 'orange_sum'])
    args = parser.parse_args()
    
    evaluate_model(args.model, args.dataset)
