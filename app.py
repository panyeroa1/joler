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
    """Call local Ollama for text response in Dutch."""
    try:
        url = "http://localhost:11434/api/generate"
        # Prompt explicitly asks for Dutch to match the new Cartesia config
        payload = {
            "model": "eburon-orbit-2.3",
            "prompt": f"User said: {prompt}\n\nAntwoord in het Nederlands. Wees beknopt, professioneel en behulpzaam:",
            "stream": False
        }
        response = requests.post(url, json=payload, timeout=10)
        if response.status_code == 200:
            return response.json().get("response", "Ik sta stand-by.").strip()
    except Exception as e:
        print(f"Ollama error: {e}")
    return "Ik verwerk je verzoek. Even geduld alstublieft."

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
            try:
                audio, sr = librosa.load(input_path, sr=16000)
                result = pipe(audio)
                transcription = result.get("text", "").strip()
            except Exception:
                pass

        if not transcription:
            transcription = "Ik kon je niet goed horen."

    except Exception as e:
        print(f"Audio processing error: {e}")
        transcription = "Systeem verwerkt je stem..."

    # 3. LLM Logic (Ollama)
    bot_text = generate_ollama_response(transcription)

    # 4. TTS Synthesis Routing (New Cartesia Config)
    audio_generated = False
    
    # Format bot text with emotion tags for Cartesia
    # We add a happy emotion tag by default as requested in the example
    tts_text = f"<emotion value=\"happy\" />{bot_text}"

    if engine == "cartesia" or engine == "orbit_sonic":
        try:
            cartesia_url = "https://api.cartesia.ai/tts/bytes"
            headers = {
                "Cartesia-Version": "2025-04-16",
                "X-API-Key": "sk_car_hDqGrK59dHF3WYAZX5LXWx",
                "Content-Type": "application/json"
            }
            payload = {
                "model_id": "sonic-3-latest",
                "transcript": tts_text,
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
                    "emotion": "content"
                }
            }
            response = requests.post(cartesia_url, headers=headers, json=payload)
            if response.status_code == 200:
                with open(output_path, "wb") as f:
                    f.write(response.content)
                audio_generated = True
            else:
                print(f"Cartesia API error: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"Cartesia error: {e}")

    # Fallback to beep if nothing generated
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
