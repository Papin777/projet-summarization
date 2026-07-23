import json
import os

def save_metrics(metrics, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

def load_metrics(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)
