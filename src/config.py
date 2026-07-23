import os
import torch

IS_COLAB = 'COLAB_GPU' in os.environ

MODELS = {
    'english': {
        'name': 'facebook/bart-base',
        'max_length': 512,
        'batch_size': 4 if torch.cuda.is_available() else 2,
        'dataset': 'samsum',
        'language': 'en'
    },
    'french': {
        'name': 'moussaKam/barthez',
        'max_length': 512,
        'batch_size': 2 if torch.cuda.is_available() else 1,
        'dataset': 'orange_sum',
        'language': 'fr'
    }
}

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"🔧 Utilisation de: {DEVICE}")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
DATA_DIR = os.path.join(BASE_DIR, 'data')

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
