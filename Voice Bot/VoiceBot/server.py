from fastapi import FastAPI, Request, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import os
import shutil
import uuid
from logger_config import setup_logger  # Assuming you have this for custom logging setup
from Voice_assistent import VoiceChatbot

# ----------------------------- Logging Setup -----------------------------
logger = setup_logger(log_name="Voicebot", log_dir="logs", debug_mode=True)

# Create necessary directories
os.makedirs("uploads", exist_ok=True)
os.makedirs("static", exist_ok=True)   # For CSS/JS if needed
os.makedirs("templates", exist_ok=True)
os.makedirs("logs", exist_ok=True)

app = FastAPI(title="Gunjan Voice Assistant API")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates
templates = Jinja2Templates(directory="templates")

# Initialize the chatbot once at startup
chatbot = VoiceChatbot(model_path=r'./models/gemma-2-2b-it-Q4_K_M.gguf')
logger.info("VoiceChatbot initialized and ready.")


@app.get("/")
async def index(request: Request):
    """Serve the main HTML page."""
    logger.info("Serving index page")
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/process_audio")
async def process_audio(audio_data: UploadFile = File(...)):
    """
    Receive uploaded audio from frontend, save it, and transcribe using Whisper.
    """
    if not audio_data.content_type.startswith("audio/"):
        logger.warning(f"Invalid file type uploaded: {audio_data.content_type}")
        raise HTTPException(status_code=400, detail="Invalid file: must be audio")

    # Generate unique filename to avoid conflicts
    file_extension = audio_data.filename.split(".")[-1] if "." in audio_data.filename else "wav"
    unique_filename = f"user_audio_{uuid.uuid4().hex}.{file_extension}"
    audio_path = os.path.join("uploads", unique_filename)

    try:
        # Save uploaded audio
        with open(audio_path, "wb") as f:
            shutil.copyfileobj(audio_data.file, f)
        logger.info(f"Audio uploaded and saved: {audio_path}")

        # Transcribe using the updated method (Whisper-based)
        text = chatbot.transcribe_audio_file(audio_path)
        
        if text is None:
            logger.warning("Transcription failed or returned empty")
            text = ""

        logger.info(f"Transcribed text: {text}")
        return JSONResponse({"text": text or ""})

    except Exception as e:
        logger.error(f"Error processing audio: {str(e)}")
        raise HTTPException(status_code=500, detail="Error processing audio")
    finally:
        # Optional: clean up uploaded file after transcription
        if os.path.exists(audio_path):
            os.remove(audio_path)
            logger.debug(f"Cleaned up uploaded file: {audio_path}")


class TextRequest(BaseModel):
    text: str


@app.post("/get_response")
async def get_response(req: TextRequest):
    """
    Send transcribed text to LLM and get response text.
    """
    if not req.text.strip():
        logger.warning("Empty text received in /get_response")
        return JSONResponse({"response_text": "I didn't hear anything. Please try again."})

    try:
        logger.info(f"Generating LLM response for: {req.text}")
        response_text = chatbot.get_response(req.text)
        logger.info(f"LLM response generated: {response_text}")

        return JSONResponse({"response_text": response_text})

    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to generate response")


@app.post("/synthesize_speech")
async def synthesize_speech(req: TextRequest):
    """
    Convert response text to speech and return the audio file path (for frontend to fetch).
    """
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="No text provided for speech synthesis")

    try:
        logger.info(f"Synthesizing speech for: {text}")

        # Use speak_text but don't play (Play=False), returns mp3 path
        mp3_path = chatbot.speak_text(text, speed=1.1, play_audio=False)

        if not os.path.exists(mp3_path):
            logger.error("Speech synthesis failed: file not created")
            raise HTTPException(status_code=500, detail="Failed to generate audio")

        # Return just the filename so frontend can request /audio/filename.mp3
        filename = os.path.basename(mp3_path)
        logger.info(f"Speech generated: {filename}")

        return JSONResponse({"audio_filename": filename})

    except Exception as e:
        logger.error(f"Error in speech synthesis: {str(e)}")
        raise HTTPException(status_code=500, detail="Speech synthesis failed")


@app.get("/audio/{filename}")
async def get_audio(filename: str):
    """
    Serve the generated speech audio file (MP3).
    """
    file_path = os.path.join("uploads", filename)

    if not os.path.exists(file_path):
        logger.warning(f"Requested audio not found: {filename}")
        raise HTTPException(status_code=404, detail="Audio file not found")

    logger.info(f"Serving audio file: {filename}")
    return FileResponse(file_path, media_type="audio/mpeg", filename=filename)


# Optional: Cleanup old audio files on startup (if needed)
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "server:app",   # change "server" if the file has a different name
        host="0.0.0.0",
        port=8000,
        reload=True,
    )


