from fastapi import FastAPI, Request, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from contextlib import asynccontextmanager
import os
import shutil
import uuid
import asyncio
import logging
from pydub import AudioSegment
from logger_config import setup_logger
import logging
from Voice_assistent import VoiceChatbot


# Directories
PUBLIC_PATH = os.getcwd()
LOG_DIR = os.path.join(PUBLIC_PATH, "logs")
STATIC_DIR = os.path.join(PUBLIC_PATH, "static")
TEMPLATES_DIR = os.path.join(PUBLIC_PATH, "templates")
MODEL_DIR = os.path.join(PUBLIC_PATH, "models")
UPLOADS_DIR = os.path.join(PUBLIC_PATH, "uploads")
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(UPLOADS_DIR, exist_ok=True)

# 1. Setup logger FIRST
logger = setup_logger(log_name="Voicebot", debug_mode=True)

# 2. SUPPRESS Uvicorn logs
logging.getLogger("uvicorn").setLevel(logging.WARNING)
logging.getLogger("uvicorn.access").disabled = True
logging.getLogger("uvicorn.error").setLevel(logging.WARNING)


templates = Jinja2Templates(directory=TEMPLATES_DIR)

# Global state
chatbot = None
model_loaded = False

async def load_model_background():
    """Load model in background without blocking startup"""
    global chatbot, model_loaded
    try:
        m_path = os.path.join(MODEL_DIR, 'Qwen2.5-3B-Instruct-Q4_K_M.gguf')
        if os.path.exists(m_path):
            logger.info("🚀 Starting model load (1-5 minutes)...")
            chatbot = VoiceChatbot(model_path=m_path)
            model_loaded = True
            logger.info("✅ VoiceChatbot fully loaded!")
        else:
            logger.error(f"❌ Model not found: {m_path}")
    except Exception as e:
        logger.error(f"❌ Model load failed: {e}")
        chatbot = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start model loading in background
    model_task = asyncio.create_task(load_model_background())
    yield
    model_task.cancel()
    logger.info("Server shutdown")

app = FastAPI(title="Gunjan Voice Assistant API", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/")
async def index(request: Request):
    logger.info("Serving index page")
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/health")
async def health():
    return {
        "status": "ok", 
        "model_loaded": model_loaded,
        "chatbot_ready": chatbot is not None
    }

class TextRequest(BaseModel):
    text: str

@app.post("/process_audio")
async def process_audio(
    file: UploadFile = File(..., description="Audio file from frontend")
):
    """Process uploaded audio file - matches frontend 'file' key"""
    global model_loaded, chatbot
    
    # Mock responses for testing (no model needed)
    if not model_loaded or chatbot is None:
        logger.info("Using mock responses (no model)")
        return JSONResponse({
            "text": "Hello! This is a mock transcription. Say 'Gunjan tell me a joke'"
        })
    
    if not file.content_type or not file.content_type.startswith("audio/"):
        raise HTTPException(400, "Invalid file: must be audio")

    # Generate unique filename
    file_extension = file.filename.split(".")[-1] if "." in file.filename else "webm"
    unique_filename = f"user_audio_{uuid.uuid4().hex}.{file_extension}"
    audio_path = os.path.join(UPLOADS_DIR, unique_filename)

    try:
        # Save uploaded file
        with open(audio_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        logger.info(f"✅ Audio saved: {os.path.basename(audio_path)} ({file.content_type})")

        # Convert to WAV if needed (Whisper compatibility)
        if file_extension.lower() not in ["wav", "mp3"]:
            logger.info("🔄 Converting to WAV...")
            if hasattr(chatbot, 'convert_to_wav'):
                wav_path = chatbot.convert_to_wav(audio_path)
            else:
                # Fallback conversion
                audio = AudioSegment.from_file(audio_path)
                wav_path = audio_path.rsplit('.', 1)[0] + ".wav"
                audio.export(wav_path, format="wav")
            audio_path = wav_path

        # Transcribe
        text = chatbot.transcribe_audio_file(audio_path) if chatbot else "Mock transcription"
        if not text or not text.strip():
            text = "No clear speech detected"
        
        logger.info(f"✅ Transcription: '{text}'")
        return JSONResponse({"text": text.strip()})

    except Exception as e:
        logger.error(f"❌ Audio error: {str(e)}")
        raise HTTPException(500, f"Processing failed: {str(e)}")
    finally:
        # Cleanup
        if os.path.exists(audio_path):
            os.remove(audio_path)

@app.post("/test_audio")
async def test_audio(audio_data: UploadFile = File(...)):
    """Debug endpoint - test audio processing only"""
    audio_path = os.path.join(UPLOADS_DIR, f"test_{uuid.uuid4().hex}.webm")
    try:
        with open(audio_path, "wb") as f:
            shutil.copyfileobj(audio_data.file, f)
        
        # Mock conversion
        wav_path = audio_path.replace(".webm", "_converted.wav")
        if not os.path.exists("./models/gemma-2-2b-it-Q4_K_M.gguf"):
            return {"debug": "No model, skipping", "file": os.path.basename(audio_path)}
        
        if chatbot:
            wav_path = chatbot.convert_to_wav(audio_path)
            text = chatbot.transcribe_audio_file(wav_path)
        else:
            text = "No chatbot"
        
        return {
            "original": os.path.basename(audio_path),
            "wav": os.path.basename(wav_path) if 'wav_path' in locals() else None,
            "text": text or "Failed",
            "file_size": os.path.getsize(audio_path)
        }
    finally:
        if os.path.exists(audio_path): os.remove(audio_path)


@app.post("/get_response")
async def get_response(req: TextRequest):
    global model_loaded, chatbot
    if not model_loaded or chatbot is None:
        return JSONResponse({"response_text": "Gunjan here! Model loading..."})
    
    try:
        # Pass fresh context
        response_text = chatbot.get_response(req.text)
        return JSONResponse({"response_text": response_text})
    except Exception as e:
        logger.error(f"Response error: {e}")
        return JSONResponse({"response_text": "Okay, let me help you with that."})


@app.post("/synthesize_speech")
async def synthesize_speech(req: TextRequest):
    global model_loaded
    if not model_loaded or chatbot is None:
        raise HTTPException(503, "Model still loading...")
    
    text = req.text.strip()
    if not text:
        raise HTTPException(400, "No text provided")

    try:
        logger.info(f"Synthesizing speech for: {text[:50]}...")
        mp3_path = chatbot.speak_text(text, speed=1.1, play_audio=False)
        
        if not os.path.exists(mp3_path):
            raise HTTPException(500, "Failed to generate audio")
        
        filename = os.path.basename(mp3_path)
        logger.info(f"Speech generated: {filename}")
        return JSONResponse({"audio_filename": filename})
    
    except Exception as e:
        logger.error(f"Speech synthesis error: {e}")
        raise HTTPException(500, "Speech synthesis failed")

@app.get("/audio/{filename}")
async def get_audio(filename: str):
    file_path = os.path.join(UPLOADS_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(404, "Audio file not found")
    
    logger.info(f"Serving audio: {filename}")
    return FileResponse(file_path, media_type="audio/mpeg", filename=filename)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False, log_level="info")
