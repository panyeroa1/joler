import os
import sys
import shutil
import uuid
import torch
import numpy as np
import uvicorn
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
import soundfile as sf
import librosa
import requests
import json
import subprocess

# Local cache configuration to avoid permission issues
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")
os.makedirs(CACHE_DIR, exist_ok=True)
os.environ["HF_HOME"] = os.path.join(CACHE_DIR, "huggingface")
os.environ["TORCH_HOME"] = os.path.join(CACHE_DIR, "torch")

# Ensure we can find the qwen_tts package
sys.path.append(PROJECT_ROOT)

# Import Qwen-Audio and Whisper components
try:
    from transformers import pipeline
    ASR_PIPE = None # Lazy load
except ImportError:
    ASR_PIPE = None

app = FastAPI(title="Maximo Primo API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup directories
STATIC_DIR = os.path.join(PROJECT_ROOT, "static")
TEMP_DIR = os.path.join(PROJECT_ROOT, "temp")
os.makedirs(TEMP_DIR, exist_ok=True)

# Mount premium UI
app.mount("/ui", StaticFiles(directory=STATIC_DIR, html=True), name="static")

def get_asr_pipe():
    global ASR_PIPE
    if ASR_PIPE is None:
        try:
            print("Loading Whisper-Tiny ASR pipeline...")
            ASR_PIPE = pipeline(
                "automatic-speech-recognition", 
                model="openai/whisper-tiny",
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
            print("ASR Pipeline Ready.")
        except Exception as e:
            print(f"ASR loading failed: {e}")
            return None
    return ASR_PIPE

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
    
    # TRANSCRIPTION (Real-time STT)
    # Convert webm to wav for processing
    wav_input_path = os.path.join(TEMP_DIR, f"{session_id}_in.wav")
    transcription = "..."
    try:
        subprocess.run(["ffmpeg", "-i", input_path, wav_input_path, "-ar", "16000", "-ac", "1", "-y"], capture_output=True)
        pipe = get_asr_pipe()
        if pipe:
            result = pipe(wav_input_path)
            transcription = result.get("text", "I couldn't catch that.").strip()
        else:
            transcription = "ASR Engine initializing..."
    except Exception as e:
        print(f"Transcription error: {e}")
        transcription = "Audio processing error."

    # BOT LOGIC
    bot_text = "Processing..."
    
    if engine == "eburon":
        bot_text = f"Eburon 3.0 confirmed. I heard: '{transcription}'. All systems are at your command."
    elif engine == "orbit":
        bot_text = f"Orbit Pro relay active. Intelligence match: '{transcription}'. Ready for directive."
    elif engine == "orbit_beta":
        bot_text = f"Orbit Beta feed established. Captured: '{transcription}'. Processing via experimental parameters."
    elif engine == "tts":
        bot_text = f"Voice Core Local. You said: '{transcription}'."
    elif engine == "cartesia":
        bot_text = f"Orbit Sonic channel prioritized. Transcription: '{transcription}'. High-speed link active."
    elif engine == "gemini":
        bot_text = f"Gemini Live Stream sync. Understanding: '{transcription}'."
    else:
        bot_text = f"System Default. Transcription: '{transcription}'."

    # TTS Synthesis Routing (Cartesia Example)
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
                "voice": {"mode": "id", "id": "005af375-5aad-4c02-9551-7fc411430542"},
                "output_format": {"container": "wav", "encoding": "pcm_f32le", "sample_rate": 44100},
                "language": "nl",
                "speed": "normal",
                "generation_config": {"speed": 1, "volume": 1, "emotion": "excited"}
            }
            response = requests.post(cartesia_url, headers=headers, json=payload)
            if response.status_code == 200:
                with open(output_path, "wb") as f:
                    f.write(response.content)
                audio_generated = True
        except Exception as e:
            print(f"Cartesia error: {e}")

    # Fallback to beep if nothing was generated
    if not audio_generated:
        sr = 16000
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration))
        y = 0.5 * np.sin(2 * np.pi * 440 * t) # Beep
        sf.write(output_path, y, sr)

    return {
        "text": bot_text,
        "transcription": transcription,
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
