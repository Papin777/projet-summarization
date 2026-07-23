import streamlit as st
import torch
from transformers import pipeline
import os
import sys
import json
import matplotlib.pyplot as plt
import pandas as pd

# Ajouter le chemin parent pour les imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import MODELS_DIR

# Configuration de la page
st.set_page_config(
    page_title="Résumé Automatique - Comparaison de Modèles",
    page_icon="📝",
    layout="wide"
)

# Titre principal
st.title("📝 Système de Résumé Automatique")
st.markdown("### Comparaison: BART (anglais) vs BARThez (français)")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    model_choice = st.radio(
        "🌐 Choisissez le modèle:",
        ["English (BART - SAMSum)", "Français (BARThez - OrangeSum)"]
    )
    
    st.subheader("🎛️ Paramètres de génération")
    max_length = st.slider("Longueur max du résumé", 50, 200, 150)
    min_length = st.slider("Longueur min du résumé", 10, 50, 30)
    num_beams = st.slider("Beam Search", 1, 8, 4)
    
    st.divider()
    st.caption("💡 Les modèles sont entraînés sur SAMSum (anglais) et OrangeSum (français)")

# Fonction de chargement des modèles (avec cache)
@st.cache_resource
def load_models():
    """Charge les modèles avec cache pour meilleure performance"""
    models = {}
    
    # Modèle anglais
    en_path = os.path.join(MODELS_DIR, "english_finetuned")
    if os.path.exists(en_path):
        try:
            models['english'] = pipeline(
                "summarization",
                model=en_path,
                device=0 if torch.cuda.is_available() else -1
            )
            st.sidebar.success("✅ Modèle anglais chargé")
        except Exception as e:
            st.sidebar.error(f"❌ Erreur chargement anglais: {e}")
    else:
        st.sidebar.warning("⚠️ Modèle anglais non trouvé. Entraînez-le d'abord.")
    
    # Modèle français
    fr_path = os.path.join(MODELS_DIR, "french_finetuned")
    if os.path.exists(fr_path):
        try:
            models['french'] = pipeline(
                "summarization",
                model=fr_path,
                device=0 if torch.cuda.is_available() else -1
            )
            st.sidebar.success("✅ Modèle français chargé")
        except Exception as e:
            st.sidebar.error(f"❌ Erreur chargement français: {e}")
    else:
        st.sidebar.warning("⚠️ Modèle français non trouvé. Entraînez-le d'abord.")
    
    return models

# Charger les modèles
models = load_models()

# Mapping des choix
model_map = {
    "English (BART - SAMSum)": "english",
    "Français (BARThez - OrangeSum)": "french"
}

# Layout principal
col1, col2 = st.columns([2, 2])

with col1:
    st.subheader("📄 Texte à résumer")
    
    text_input = st.text_area(
        "Entrez votre texte:",
        height=300,
        placeholder="Collez un article, un dialogue ou un document ici...",
        help="Le texte sera résumé par le modèle sélectionné"
    )
    
    # Exemples
    st.divider()
    st.caption("📚 Cliquez sur un exemple ci-dessous:")
    
    col_ex1, col_ex2 = st.columns(2)
    
    with col_ex1:
        if st.button("💬 Dialogue anglais", use_container_width=True):
            st.session_state['text_input'] = """Amanda: I can't come to the meeting tomorrow. I have a dentist appointment.
John: That's okay. We can reschedule for Thursday.
Amanda: Thursday works for me. What time?
John: How about 2 PM?
Amanda: Perfect. See you then."""
            st.rerun()
    
    with col_ex2:
        if st.button("📰 Article français", use_container_width=True):
            st.session_state['text_input'] = """Le président français Emmanuel Macron a annoncé aujourd'hui un nouveau plan de relance économique. Ce plan prévoit 100 milliards d'euros d'investissements dans les secteurs de la transition écologique, du numérique et de la formation professionnelle. Le chef de l'État a souligné l'importance de soutenir les entreprises françaises dans cette période de crise sans précédent."""
            st.rerun()
    
    # Bouton de génération
    summarize_button = st.button(
        "✨ Générer le résumé",
        type="primary",
        use_container_width=True
    )

with col2:
    st.subheader("📌 Résumé généré")
    
    # Zone de résultat
    summary_output = st.text_area(
        "Résultat:",
        height=300,
        placeholder="Le résumé apparaîtra ici...",
        disabled=True
    )
    
    # Métriques ROUGE
    st.divider()
    st.caption("📊 Métriques ROUGE (sur dataset test)")
    
    model_key = model_map[model_choice]
    results_file = os.path.join(MODELS_DIR, f"{model_key}_finetuned", "evaluation_results.json")
    
    if os.path.exists(results_file):
        with open(results_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
            scores = results['rouge_scores']
            
            col_m1, col_m2, col_m3 = st.columns(3)
            col_m1.metric("ROUGE-1", f"{scores['rouge1']:.3f}")
            col_m2.metric("ROUGE-2", f"{scores['rouge2']:.3f}")
            col_m3.metric("ROUGE-L", f"{scores['rougeL']:.3f}")
    else:
        st.info("ℹ️ Entraînez le modèle et lancez l'évaluation pour voir les métriques.")

# Fonction de résumé
def summarize(text, model_key):
    """Génère un résumé du texte"""
    if not text.strip():
        return "⚠️ Veuillez entrer un texte à résumer."
    
    if model_key not in models:
        return f"❌ Modèle {model_key} non disponible. Veuillez d'abord entraîner le modèle."
    
    try:
        with st.spinner("🧠 Génération du résumé en cours..."):
            result = models[model_key](
                text,
                max_length=max_length,
                min_length=min_length,
                do_sample=False,
                num_beams=num_beams,
                early_stopping=True,
                no_repeat_ngram_size=2
            )
            return result[0]['summary_text']
    except Exception as e:
        return f"❌ Erreur: {str(e)}"

# Action du bouton
if summarize_button and text_input:
    model_key = model_map[model_choice]
    summary = summarize(text_input, model_key)
    st.session_state['summary'] = summary
    st.rerun()

# Afficher le résumé s'il existe
if 'summary' in st.session_state:
    with col2:
        st.text_area("Résultat:", value=st.session_state['summary'], height=300, disabled=True)

# Récupérer le texte des exemples
if 'text_input' in st.session_state:
    with col1:
        st.text_area("Entrez votre texte:", value=st.session_state['text_input'], height=300)

# SECTION COMPARAISON DES MODÈLES
st.divider()
st.subheader("📊 Comparaison des Modèles")

# Vérifier si les deux modèles sont évalués
en_results_file = os.path.join(MODELS_DIR, "english_finetuned", "evaluation_results.json")
fr_results_file = os.path.join(MODELS_DIR, "french_finetuned", "evaluation_results.json")

if os.path.exists(en_results_file) and os.path.exists(fr_results_file):
    
    # Charger les résultats
    with open(en_results_file, 'r', encoding='utf-8') as f:
        en_results = json.load(f)
    with open(fr_results_file, 'r', encoding='utf-8') as f:
        fr_results = json.load(f)
    
    # Créer un dataframe
    df = pd.DataFrame({
        'Modèle': ['BART (English)', 'BARThez (French)'],
        'ROUGE-1': [en_results['rouge_scores']['rouge1'], fr_results['rouge_scores']['rouge1']],
        'ROUGE-2': [en_results['rouge_scores']['rouge2'], fr_results['rouge_scores']['rouge2']],
        'ROUGE-L': [en_results['rouge_scores']['rougeL'], fr_results['rouge_scores']['rougeL']]
    })
    
    # Afficher le tableau
    st.dataframe(
        df.style.format({'ROUGE-1': '{:.4f}', 'ROUGE-2': '{:.4f}', 'ROUGE-L': '{:.4f}'}),
        use_container_width=True
    )
    
    # Graphique de comparaison
    fig, ax = plt.subplots(figsize=(10, 5))
    
    df_plot = df.set_index('Modèle').T
    bars = df_plot.plot(kind='bar', ax=ax, color=['#2E86AB', '#A23B72'], width=0.7)
    
    ax.set_ylabel('Score', fontsize=12)
    ax.set_xlabel('Métriques', fontsize=12)
    ax.set_title('Comparaison des Scores ROUGE', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1)
    
    # Ajouter les valeurs sur les barres
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', fontsize=10, padding=3)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Interprétation
    st.info("""
    **📌 Interprétation des résultats :**
    - **ROUGE-1** : Mesure la correspondance des mots individuels (précision/rappel)
    - **ROUGE-2** : Mesure la correspondance des paires de mots (plus strict)
    - **ROUGE-L** : Mesure la correspondance des séquences les plus longues
    
    Des scores plus élevés indiquent de meilleures performances.
    """)
    
else:
    st.info("""
    ℹ️ **Entraînez et évaluez les deux modèles pour voir la comparaison.**
    
    1. `python src/train.py --language english`
    2. `python src/train.py --language french`
    3. `python src/evaluate.py --model models/english_finetuned --dataset samsum`
    4. `python src/evaluate.py --model models/french_finetuned --dataset orange_sum`
    """)

# Footer
st.divider()
st.caption("🚀 Projet de résumé automatique - Comparaison BART vs BARThez")
st.caption("👤 Développé avec Streamlit")

