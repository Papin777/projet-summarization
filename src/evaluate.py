import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from rouge_score import rouge_scorer
import numpy as np
import json
import os
import argparse

def evaluate_model(model_path):
    """Évalue un modèle entraîné"""
    
    print(f"\n{'='*50}")
    print(f"📊 Évaluation du modèle: {os.path.basename(model_path)}")
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
    
    # Exemples de test
    if 'english' in model_path:
        test_texts = [
            "Amanda: I can't come to the meeting tomorrow. I have a dentist appointment.\nJohn: That's okay. We can reschedule for Thursday.\nAmanda: Thursday works for me.\nJohn: How about 2 PM?\nAmanda: Perfect.",
            "Sarah: Did you finish the report?\nMike: Almost, I need one more day.\nSarah: Okay, but please send it by Friday.\nMike: Will do!",
            "David: Hi, I'm calling about the job opening.\nEmma: Yes, we're still accepting applications.\nDavid: Great, I'll send my CV today.\nEmma: Perfect, I'll review it."
        ]
        references = [
            "Amanda and John reschedule their meeting to Thursday at 2 PM.",
            "Mike needs one more day to finish the report and will send it by Friday.",
            "David applies for a job and Emma agrees to review his CV."
        ]
    else:
        test_texts = [
            "Le président français Emmanuel Macron a annoncé aujourd'hui un nouveau plan de relance économique de 100 milliards d'euros.",
            "La France investit massivement dans la transition énergétique avec un budget de 50 milliards d'euros.",
            "Le gouvernement a dévoilé un plan de soutien aux entreprises touchées par la crise économique."
        ]
        references = [
            "Plan de relance économique de 100 milliards d'euros.",
            "Investissement de 50 milliards d'euros pour la transition énergétique.",
            "Plan de soutien aux entreprises avec des prêts garantis."
        ]
    
    predictions = []
    print(f"📝 Génération de {len(test_texts)} résumés...")
    
    for i, text in enumerate(test_texts):
        inputs = tokenizer(text, return_tensors='pt', max_length=256, truncation=True).to(device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=64, num_beams=4, early_stopping=True)
        pred = tokenizer.decode(outputs[0], skip_special_tokens=True)
        predictions.append(pred)
        print(f"\n  Texte {i+1}: {text[:60]}...")
        print(f"  → Résumé: {pred}")
    
    # Calcul ROUGE
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = []
    for pred, ref in zip(predictions, references):
        scores.append(scorer.score(ref, pred))
    
    avg_scores = {
        'rouge1': np.mean([s['rouge1'].fmeasure for s in scores]),
        'rouge2': np.mean([s['rouge2'].fmeasure for s in scores]),
        'rougeL': np.mean([s['rougeL'].fmeasure for s in scores])
    }
    
    print(f"\n📈 Scores ROUGE:")
    print(f"  ROUGE-1: {avg_scores['rouge1']:.4f}")
    print(f"  ROUGE-2: {avg_scores['rouge2']:.4f}")
    print(f"  ROUGE-L: {avg_scores['rougeL']:.4f}")
    
    # Sauvegarde
    results = {
        'model_path': model_path,
        'rouge_scores': avg_scores,
        'predictions': predictions,
        'references': references
    }
    
    with open(os.path.join(model_path, 'evaluation_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    return avg_scores

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--dataset', required=True)
    args = parser.parse_args()
    evaluate_model(args.model)
