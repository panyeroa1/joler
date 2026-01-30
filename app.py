# coding=utf-8
# Maximo Primo - Powered by Qwen3-TTS
import os
import gradio as gr
import numpy as np
import torch
from huggingface_hub import snapshot_download
from qwen_tts import Qwen3TTSModel

# Redirect caches to local project directory
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")
os.makedirs(CACHE_DIR, exist_ok=True)
os.environ["HF_HOME"] = os.path.join(CACHE_DIR, "huggingface")
os.environ["TORCH_HOME"] = os.path.join(CACHE_DIR, "torch")
os.environ["XDG_CACHE_HOME"] = CACHE_DIR

MODEL_SIZES = ["0.6B", "1.7B"]
SPEAKERS = ["Aiden", "Dylan", "Eric", "Ono_anna", "Ryan", "Serena", "Sohee", "Uncle_fu", "Vivian"]

loaded_models = {}

def get_model(model_size):
    key = model_size
    if key not in loaded_models:
        print(f"Initializing Maximo Engine ({model_size})...")
        repo_id = f"Qwen/Qwen3-TTS-12Hz-{model_size}-CustomVoice"
        model_path = snapshot_download(repo_id)
        loaded_models[key] = Qwen3TTSModel.from_pretrained(
            model_path,
            device_map="auto",
            dtype=torch.float32,
            attn_implementation="eager"
        )
    return loaded_models[key]

def generate_voice(text, speaker, model_size):
    if not text or not text.strip(): return None, "Please enter some text."
    try:
        model = get_model(model_size)
        wavs, sr = model.generate_custom_voice(
            text=text.strip(), 
            language="Auto", 
            speaker=speaker.lower().replace(" ", "_"), 
            instruct=""
        )
        return (sr, wavs[0]), f"Success: Rendered {len(text)} characters."
    except Exception as e: 
        return None, f"Engine Error: {str(e)}"

# Premium CSS for Maximo Primo
PREMIUM_CSS = """
:root {
    --primary: #00ff88;
    --primary-glow: rgba(0, 255, 136, 0.4);
    --secondary: #00d4ff;
    --bg: #0a0a0a;
    --surface: #121212;
    --text: #ffffff;
}
body { background-color: var(--bg) !important; color: var(--text) !important; }
.gradio-container { background-color: var(--bg) !important; border: none !important; }
.tabs { background: transparent !important; border: none !important; }
.tab-nav { border-bottom: 1px solid rgba(255, 255, 255, 0.1) !important; }
button.primary { 
    background: linear-gradient(135deg, var(--primary), var(--secondary)) !important;
    color: black !important;
    font-weight: 600 !important;
    border: none !important;
    box-shadow: 0 0 15px var(--primary-glow) !important;
}
.message { border-radius: 15px !important; }
header {
    background: rgba(18, 18, 18, 0.8);
    backdrop-filter: blur(20px);
    padding: 20px;
    text-align: center;
    border-bottom: 1px solid rgba(255, 255, 255, 0.05);
    margin-bottom: 20px;
}
h1 {
    font-size: 1.8rem;
    font-weight: 600;
    letter-spacing: 3px;
    background: linear-gradient(135deg, var(--primary), var(--secondary));
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
    text-transform: uppercase;
}
"""

with gr.Blocks(css=PREMIUM_CSS, title="Maximo Primo") as demo:
    gr.HTML("<header><h1>MAXIMO PRIMO</h1></header>")
    
    with gr.Column(variant="panel"):
        t_text = gr.Textbox(
            label="Input Text", 
            placeholder="What should Maximo say?", 
            lines=3,
            value="Welcome to the future of voice synthesis. I am Maximo Primo."
        )
        
        with gr.Row():
            t_spk = gr.Dropdown(label="Voice Identity", choices=SPEAKERS, value="Ryan")
            t_size = gr.Dropdown(label="Engine Velocity", choices=MODEL_SIZES, value="0.6B")
            
        t_btn = gr.Button("SYNTHESIZE", variant="primary")
        
    with gr.Column(variant="panel"):
        t_audio = gr.Audio(label="Maximo Voice Output")
        t_status = gr.Markdown("System Status: **Standby**")
        
    t_btn.click(generate_voice, [t_text, t_spk, t_size], [t_audio, t_status])

if __name__ == "__main__":
    demo.launch()
