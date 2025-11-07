import os
import logging
import subprocess
import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from faster_whisper import WhisperModel
import google.generativeai as genai
import pyttsx3
from dotenv import load_dotenv
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from faster_whisper import WhisperModel

from fastapi import WebSocket



app = FastAPI()

# Mount static files to a specific path
app.mount("/static", StaticFiles(directory="static"), name="static")

# --- Configuration ---
logging.basicConfig(level=logging.INFO)
load_dotenv()

INPUT_WEBM = "live_input.webm"
INPUT_WAV = "live_input.wav"
OUTPUT_MP3 = "live_answer.mp3"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY not set")

genai.configure(api_key=GEMINI_API_KEY)
GEMINI_MODEL_NAME = "gemini-2.5-flash"
whisper_model = WhisperModel("tiny", device="cpu", compute_type="int8")

app = FastAPI(title="Gemini Voice Assistant")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.websocket("/ws/audio")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    audio_data = bytearray()

    try:
        clean_files()
        while True:
            chunk = await asyncio.wait_for(websocket.receive_bytes(), timeout=10.0)
            audio_data.extend(chunk)
            if len(audio_data) > 2_000_000:
                break
    except Exception:
        pass

    if not audio_data:
        await websocket.send_json({"error": "No audio received."})
        await websocket.close()
        return

    # Save received audio to WebM file
    with open(INPUT_WEBM, "wb") as f:
        f.write(audio_data)

    # 🔧 FFmpeg conversion with robust flags
    subprocess.run([
        "ffmpeg",
        "-probesize", "5000000",
        "-fflags", "+genpts",
        "-i", INPUT_WEBM,
        "-c:a", "pcm_s16le",
        "-ar", "48000",
        "-ac", "1",
        INPUT_WAV
    ], check=True)

    # Transcribe with FasterWhisper
    segments, _ = whisper_model.transcribe(INPUT_WAV)
    transcription = " ".join([seg.text.strip() for seg in segments if seg.text])
    if not transcription:
        transcription = "No clear speech detected."

    # Generate AI response
    answer = await asyncio.to_thread(get_ai_response, transcription)

    # Generate TTS audio
    await asyncio.to_thread(generate_tts_audio, answer, OUTPUT_MP3)

    # Send response
    await websocket.send_json({
        "transcription": transcription,
        "answer": answer
    })
    await websocket.close()



@app.get("/", response_class=HTMLResponse)
def root():
    return FileResponse("static/index.html")

@app.get("/favicon.ico")
def favicon():
    return FileResponse("static/favicon.ico")

# --- Utility Functions ---
def clean_files():
    for f in [INPUT_WEBM, INPUT_WAV, OUTPUT_MP3]:
        if os.path.exists(f):
            os.remove(f)

def get_ai_response(prompt: str) -> str:
    try:
        model = genai.GenerativeModel(GEMINI_MODEL_NAME)
        response = model.generate_content(prompt)
        return response.text if hasattr(response, "text") else "No response text found."
    except Exception as e:
        logging.error(f"Gemini error: {e}")
        return "Sorry, I couldn't generate a response."


def generate_tts_audio(text: str, output_path: str):
    engine = pyttsx3.init()
    engine.save_to_file(text, output_path)
    engine.runAndWait()

# --- WebSocket Endpoint ---
@app.websocket("/ws/audio")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    audio_data = bytearray()

    try:
        clean_files()
        while True:
            chunk = await asyncio.wait_for(websocket.receive_bytes(), timeout=10.0)
            audio_data.extend(chunk)
            if len(audio_data) > 2_000_000:
                break
    except Exception:
        pass

    if not audio_data:
        await websocket.send_json({"error": "No audio received."})
        await websocket.close()
        return

    with open(INPUT_WEBM, "wb") as f:
        f.write(audio_data)

    subprocess.run(["ffmpeg", "-y", "-i", INPUT_WEBM, INPUT_WAV], check=True)

    segments, _ = whisper_model.transcribe(INPUT_WAV)
    transcription = " ".join([seg.text.strip() for seg in segments if seg.text])
    if not transcription:
        transcription = "No clear speech detected."

    answer = await asyncio.to_thread(get_ai_response, transcription)
    await asyncio.to_thread(generate_tts_audio, answer, OUTPUT_MP3)

    await websocket.send_json({
        "transcription": transcription,
        "answer": answer
    })
    await websocket.close()

# --- Serve MP3 ---
@app.get("/live_answer.mp3")
async def get_audio():
    if not os.path.exists(OUTPUT_MP3):
        raise HTTPException(status_code=404, detail="Audio not found.")
    return FileResponse(OUTPUT_MP3, media_type="audio/mpeg")
