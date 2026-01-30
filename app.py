import os
import shutil
import uuid
import torch
import numpy as np
import uvicorn
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from qwen_tts.models import Qwen2AudioForConditionalGeneration
from qwen_tts.processors import Qwen2AudioProcessor
import soundfile as sf
import librosa
import requests
import json

app = FastAPI(title="Maximo Primo API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup directories
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
TEMP_DIR = os.path.join(os.path.dirname(__file__), "temp")
os.makedirs(TEMP_DIR, exist_ok=True)

# Mount premium UI
app.mount("/ui", StaticFiles(directory=STATIC_DIR, html=True), name="static")

# Global variables for the model
processor = None
model = None
device = "cuda" if torch.cuda.is_available() else "cpu"

def load_model():
    global processor, model
    print(f"Loading Qwen3-TTS engine on {device}...")
    model_path = "Qwen/Qwen2-Audio-7B-Instruct" # Placeholder for Qwen3-TTS path
    processor = Qwen2AudioProcessor.from_pretrained(model_path)
    model = Qwen2AudioForConditionalGeneration.from_pretrained(model_path, torch_dtype="auto").to(device)
    print("Engine Ready.")

@app.on_event("startup")
async def startup_event():
    # Model loading would be here, but we'll mock it for the demo if model files are missing
    # load_model()
    pass

@app.get("/")
async def redirect_to_ui():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))

@app.post("/process")
async def process_audio(file: UploadFile = File(...), engine: str = Form("tts")):
    session_id = str(uuid.uuid4())
    input_path = os.path.join(TEMP_DIR, f"{session_id}_in.webm")
    output_path = os.path.join(TEMP_DIR, f"{session_id}_out.wav")
    
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Engine routing logic
    if engine == "eburon":
        bot_text = "Eburon dynamic response processing..."
    elif engine == "orbit":
        bot_text = "Orbit satellite relay active."
    elif engine == "tts":
        bot_text = "Coqui TTS engine engaged."
    elif engine == "cartesia":
        bot_text = "Cartesia low-latency bridge initialized."
    elif engine == "gemini":
        bot_text = "Gemini Live Audio stream standby."
    else:
        bot_text = "System default voice active."

    # TTS Synthesis Routing
    audio_generated = False

    if engine == "cartesia":
        try:
            cartesia_url = "https://api.cartesia.ai/tts/bytes"
            headers = {
                "Cartesia-Version": "2025-04-16",
                "X-API-Key": "sk_car_SfvQvL1pKathEnBbiTQPUm",
                "Content-Type": "application/json"
            }
            payload = {
                "model_id": "sonic-3-latest",
                "transcript": bot_text,
                "voice": {
                    "mode": "id",
                    "id": "005af375-5aad-4c02-9551-7fc411430542"
                },
                "output_format": {
                    "container": "wav",
                    "encoding": "pcm_f32le",
                    "sample_rate": 44100
                },
                "language": "nl",
                "speed": "normal",
                "pronunciation_dict_id": "pdict_nyWBBphhMbxQmpmccYdMUy",
                "generation_config": {
                    "speed": 1,
                    "volume": 1,
                    "emotion": "excited"
                }
            }
            response = requests.post(cartesia_url, headers=headers, json=payload)
            if response.status_code == 200:
                with open(output_path, "wb") as f:
                    f.write(response.content)
                audio_generated = True
        except Exception as e:
            print(f"Cartesia error: {e}")

    if engine == "tts" and not audio_generated:
        # Attempt to use local Coqui TTS if available
        try:
            # Mocking command-line call for demonstration
            # subprocess.run(["tts", "--text", bot_text, "--out_path", output_path])
            pass 
        except Exception:
            pass

    # For demonstration/missing engine, generate fallback audio if nothing was generated
    if not audio_generated:
        sr = 16000
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration))
        y = 0.5 * np.sin(2 * np.pi * 440 * t) # Beep
        sf.write(output_path, y, sr)

    return {
        "text": bot_text,
        "audio_url": f"/audio/{session_id}",
        "engine": engine
    }

@app.get("/audio/{session_id}")
async def get_audio(session_id: str):
    file_path = os.path.join(TEMP_DIR, f"{session_id}_out.wav")
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="audio/wav")
    return JSONResponse(status_code=404, content={"message": "Audio not found"})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
