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
        try:
            score = scorer.score(ref, pred)
            scores.append(score)
        except:
            continue
    if not scores:
        return {'rouge1': 0, 'rouge2': 0, 'rougeL': 0}
    return {
        'rouge1': np.mean([s['rouge1'].fmeasure for s in scores]),
        'rouge2': np.mean([s['rouge2'].fmeasure for s in scores]),
        'rougeL': np.mean([s['rougeL'].fmeasure for s in scores])
    }

def evaluate_model(model_path, dataset_name, max_samples=None):
    print(f"\n{'='*50}")
    print(f"📊 Évaluation du modèle: {os.path.basename(model_path)}")
    print(f"📊 Dataset: {dataset_name}")
    print(f"{'='*50}\n")
    
    try:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return None
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.eval()
    
    try:
        if dataset_name == 'orange_sum':
            print("📥 Chargement de OrangeSum via GEM...")
            dataset = load_dataset('GEM/OrangeSum', split='test')
            text_field = 'text'
            summary_field = 'summary'
        elif dataset_name == 'samsum':
            print("📥 Chargement de SAMSum via knkarthick/samsum...")
            dataset = load_dataset("knkarthick/samsum", split='test')
            text_field = 'dialogue'
            summary_field = 'summary'
        else:
            raise ValueError(f"Dataset {dataset_name} non supporté")
    except Exception as e:
        print(f"❌ Erreur: {e}")
        from datasets import Dataset
        import pandas as pd
        if dataset_name == 'samsum':
            data = {'dialogue': ["Test"], 'summary': ["Test"]}
        else:
            data = {'text': ["Test"], 'summary': ["Test"]}
        df = pd.DataFrame(data)
        dataset = Dataset.from_pandas(df)
    
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    predictions = []
    references = []
    print(f"📝 Génération de {len(dataset)} résumés...")
    
    for i, example in enumerate(dataset):
        if (i + 1) % 50 == 0:
            print(f"Progression: {i+1}/{len(dataset)}")
        inputs = tokenizer(example[text_field], return_tensors='pt', max_length=512, truncation=True).to(device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=128, min_length=30, num_beams=4, early_stopping=True)
        pred = tokenizer.decode(outputs[0], skip_special_tokens=True)
        predictions.append(pred)
        references.append(example[summary_field])
    
    rouge_scores = compute_rouge(predictions, references)
    print(f"\n📈 ROUGE-1: {rouge_scores['rouge1']:.4f}")
    print(f"📈 ROUGE-2: {rouge_scores['rouge2']:.4f}")
    print(f"📈 ROUGE-L: {rouge_scores['rougeL']:.4f}")
    
    results = {'dataset': dataset_name, 'model_path': model_path, 'rouge_scores': rouge_scores, 'num_examples': len(dataset)}
    with open(os.path.join(model_path, 'evaluation_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    return rouge_scores

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--dataset', required=True, choices=['samsum', 'orange_sum'])
    parser.add_argument('--max-samples', type=int, default=None)
    args = parser.parse_args()
    evaluate_model(args.model, args.dataset, args.max_samples)
