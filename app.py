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

# Import Whisper components
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

def generate_ollama_response(prompt):
    """Call local Ollama for text response."""
    try:
        url = "http://localhost:11434/api/generate"
        payload = {
            "model": "eburon-orbit-2.3",
            "prompt": f"User said: {prompt}\n\nResponse (be concise, professional, and helpful):",
            "stream": False
        }
        response = requests.post(url, json=payload, timeout=10)
        if response.status_code == 200:
            return response.json().get("response", "I am standing by.").strip()
    except Exception as e:
        print(f"Ollama error: {e}")
    return "I am processing your request. Please stand by."

@app.get("/")
async def redirect_to_ui():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))

@app.post("/process")
async def process_audio(file: UploadFile = File(...), engine: str = Form("cartesia")):
    session_id = str(uuid.uuid4())
    input_path = os.path.join(TEMP_DIR, f"{session_id}_in.webm")
    wav_input_path = os.path.join(TEMP_DIR, f"{session_id}_in.wav")
    output_path = os.path.join(TEMP_DIR, f"{session_id}_out.wav")
    
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    transcription = ""
    try:
        # 1. Audio Conversion
        converted = False
        
        # Try ffmpeg
        if shutil.which("ffmpeg"):
            subprocess.run(["ffmpeg", "-i", input_path, wav_input_path, "-ar", "16000", "-ac", "1", "-y"], capture_output=True)
            if os.path.exists(wav_input_path) and os.path.getsize(wav_input_path) > 0:
                converted = True
        
        # Try afconvert (macOS native)
        if not converted and shutil.which("afconvert"):
            subprocess.run(["afconvert", "-f", "WAVE", "-d", "LEI16@16000", input_path, wav_input_path], capture_output=True)
            if os.path.exists(wav_input_path) and os.path.getsize(wav_input_path) > 0:
                converted = True

        # 2. Transcription
        pipe = get_asr_pipe()
        if converted and pipe:
            result = pipe(wav_input_path)
            transcription = result.get("text", "").strip()
        elif pipe:
            # Last ditch direct load
            try:
                audio, sr = librosa.load(input_path, sr=16000)
                result = pipe(audio)
                transcription = result.get("text", "").strip()
            except Exception:
                pass

        if not transcription:
            transcription = "I couldn't hear you clearly."

    except Exception as e:
        print(f"Audio processing error: {e}")
        transcription = "System is processing your voice..."

    # 3. LLM Logic (Ollama)
    bot_text = generate_ollama_response(transcription)

    # 4. TTS Synthesis Routing (Cartesia as primary)
    audio_generated = False
    
    # Clean up bot text for TTS (optional, just in case)
    tts_text = bot_text.replace("*", "").replace("#", "")

    if engine == "cartesia" or engine == "orbit_sonic": # Support both names
        try:
            cartesia_url = "https://api.cartesia.ai/tts/bytes"
            headers = {
                "Cartesia-Version": "2025-04-16",
                "X-API-Key": "sk_car_SfvQvL1pKathEnBbiTQPUm",
                "Content-Type": "application/json"
            }
            payload = {
                "model_id": "sonic-3-latest",
                "transcript": tts_text,
                "voice": {"mode": "id", "id": "005af375-5aad-4c02-9551-7fc411430542"},
                "output_format": {"container": "wav", "encoding": "pcm_f32le", "sample_rate": 44100},
                "language": "en", # Defaulted to en for general usability
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

    # Fallback/Other engines...
    if not audio_generated and engine == "eburon":
        # Placeholder for local Qwen3-TTS generation if fully rigged
        pass

    # Final fallback to beep if nothing generated
    if not audio_generated:
        sr = 16000
        duration = 0.5
        t = np.linspace(0, duration, int(sr * duration))
        y = 0.5 * np.sin(2 * np.pi * 440 * t) 
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
