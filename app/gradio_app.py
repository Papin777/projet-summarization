import gradio as gr
import torch
from transformers import pipeline
import os

def load_models():
    models = {}
    
    en_path = "../models/english_finetuned"
    if os.path.exists(en_path):
        try:
            models['english'] = pipeline(
                "summarization", 
                model=en_path,
                device=0 if torch.cuda.is_available() else -1
            )
            print("✅ Modèle anglais chargé")
        except:
            print("❌ Erreur chargement modèle anglais")
    
    fr_path = "../models/french_finetuned"
    if os.path.exists(fr_path):
        try:
            models['french'] = pipeline(
                "summarization",
                model=fr_path,
                device=0 if torch.cuda.is_available() else -1
            )
            print("✅ Modèle français chargé")
        except:
            print("❌ Erreur chargement modèle français")
    
    return models

models = load_models()

def summarize(text, language):
    if not text.strip():
        return "⚠️ Veuillez entrer un texte à résumer."
    
    if language not in models:
        return f"❌ Modèle {language} non disponible."
    
    try:
        model = models[language]
        result = model(
            text,
            max_length=150,
            min_length=30,
            do_sample=False,
            num_beams=4,
            early_stopping=True
        )
        return result[0]['summary_text']
    except Exception as e:
        return f"❌ Erreur: {str(e)}"

with gr.Blocks(title="Système de Résumé Automatique", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 📝 Système de Résumé Automatique
    **Comparaison de modèles:** BART (anglais) vs BARThez (français)
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            text_input = gr.Textbox(
                label="📄 Texte à résumer",
                placeholder="Collez votre texte ici...",
                lines=10
            )
            language_choice = gr.Radio(
                choices=["English (SAMSum)", "Français (OrangeSum)"],
                label="🌐 Modèle",
                value="English (SAMSum)"
            )
            with gr.Row():
                submit_btn = gr.Button("✨ Générer", variant="primary")
                clear_btn = gr.Button("🔄 Effacer", variant="secondary")
        
        with gr.Column(scale=1):
            summary_output = gr.Textbox(
                label="📌 Résumé généré",
                lines=10,
                interactive=False
            )
    
    gr.Examples(
        examples=[
            ["Amanda: I can't come to the meeting tomorrow.\nJohn: That's okay. We can reschedule.\nAmanda: Thursday works for me.\nJohn: How about 2 PM?\nAmanda: Perfect.", "English (SAMSum)"],
            ["Le président français Emmanuel Macron a annoncé aujourd'hui un nouveau plan de relance économique. Ce plan prévoit 100 milliards d'euros d'investissements.", "Français (OrangeSum)"],
        ],
        inputs=[text_input, language_choice],
        label="📚 Exemples"
    )
    
    submit_btn.click(fn=summarize, inputs=[text_input, language_choice], outputs=[summary_output])
    clear_btn.click(fn=lambda: ("", ""), inputs=[], outputs=[text_input, summary_output])

if __name__ == "__main__":
    demo.launch(share=True, debug=False)
