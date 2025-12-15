# Gunjan Voice Assistant

A **fully offline** voice assistant powered by **Qwen2.5-3B-Instruct-Q4_K_M** (GGUF), **Whisper** transcription, **gTTS** speech synthesis, and **FastAPI** web interface. **Single-turn responses** like Alexa with clean logging and background model loading.

## ✨ Features

- **Fully Offline LLM**: Gemma-2-2B-IT Q4_K_M (~1.4GB GGUF)
- **Whisper STT**: `openai/whisper-base.en` for accurate transcription
- **gTTS TTS**: Google Text-to-Speech → MP3 playback
- **FastAPI Web UI**: `http://localhost:8000` with real-time health checks
- **Background Loading**: Instant server startup, model loads async
- **Structured Logging**: File rotation + terminal output
- **Single-Turn Mode**: Clean Alexa-style responses (no conversation memory)

## 🛠️ Quick Start

### 1. Setup Environment
```bash
cd VoiceBot
python -m venv venv
source venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

### 2. Download Model (~1.4GB)
```bash
mkdir -p models
wget -O models/gemma-2-2b-it-Q4_K_M.gguf \
  "https://huggingface.co/bartowski/gemma-2-2b-it-GGUF/resolve/main/gemma-2-2b-it-Q4_K_M.gguf"
```

### 3. Audio Dependencies (Ubuntu)
```bash
sudo apt update && sudo apt install -y portaudio19-dev ffmpeg
pip install pyaudio
```

### 4. Run Server
```bash
python server.py
```
**Open:** `http://localhost:8000`  
**Wait:** Model loads in background (~1-2 min)  
**Status:** Check `/health` endpoint

## 📁 Project Structure
```
├── server.py              # FastAPI server + background model loading
├── Voice_assistent.py     # VoiceChatbot class (Whisper + Gemma + gTTS)
├── logger_config.py       # Structured logging (file + terminal)
├── templates/index.html   # Web UI
├── static/css/styles.css  # Responsive UI
├── static/js/script.js    # MediaRecorder + API calls
├── models/                # GGUF model (~1.4GB)
├── uploads/               # Temp audio files (auto-cleaned)
└── logs/                  # Rotated logs (Voicebot.log.*)
```

## 🚀 API Endpoints

| Method | Endpoint              | Description                  |
|--------|----------------------|------------------------------|
| `GET`  | `/`                  | Web UI                      |
| `GET`  | `/health`            | Model status (`model_loaded`) |
| `POST` | `/process_audio`     | Audio file → transcription  |
| `POST` | `/get_response`      | Text → single-turn LLM      |
| `POST` | `/synthesize_speech` | Text → MP3 filename         |
| `GET`  | `/audio/{filename}`  | Serve TTS MP3               |

## 📦 Requirements
```txt
fastapi
uvicorn[standard]
python-multipart
jinja2
pydantic
torch
transformers
langchain-community
llama-cpp-python
speechrecognition
pydub
gtts
python-dotenv
```

**Install:** `pip install -r requirements.txt`

## 🎵 Workflow

```
🎙️ Record (WebM) → WAV conversion → Whisper STT
         ↓
     "Hello Gunjan" → Gemma-2-2B → "Okay I'm doing great!"
         ↓
     gTTS → MP3 → Browser playback
```

## ⚙️ Logging

**Terminal + File** (`logs/Voicebot.log`):
```
21:34:56,123 server.py [MainThread] - INFO  :  89 - ✅ Audio saved: user_audio_xyz.webm
21:34:57,456 server.py [MainThread] - INFO  : 102 - ✅ Transcribed: 'hello gunjan'
21:34:58,789 server.py [MainThread] - INFO  : 145 - Gunjan: Okay I'm doing great thanks!
```

**Monitor:** `tail -f logs/Voicebot.log`

## 🔧 Troubleshooting

| Issue | Solution |
|-------|----------|
| `422 Unprocessable Entity` | `pip install python-multipart` |
| `No speech detected` | Check WAV conversion in `/process_audio` |
| `Model still loading` | Wait 1-2 min, check `/health` |
| Uvicorn spam | Add `logging.getLogger("uvicorn").setLevel(logging.WARNING)` |
| PyAudio fails | `sudo apt install portaudio19-dev` |

## 📱 Web UI Features

- **Responsive design** (mobile + desktop)
- **Real-time health** (`/health` polling)
- **MediaRecorder API** (WebM → Whisper)
- **Visual feedback** (recording pulse, status colors)
- **Error handling** (network, autoplay, transcription)

## 🚀 Production

```bash
# No reload for model stability
uvicorn server:app --host 0.0.0.0 --port 8000 --reload=False --log-level warning
```

## 💻 Development Commands

```bash
# Clean logs
python -c "from logger_config import delete_old_logs; delete_old_logs('logs')"

# Test endpoints
curl -X POST -F "file=@test.wav" http://localhost:8000/process_audio

# Health check
curl http://localhost:8000/health
```

## 📈 Performance (CPU)

- **Server startup**: <1s (model loads async)
- **Model load**: 30-120s (Gemma-2-2B Q4_K_M)
- **STT**: 2-5s (Whisper base)
- **LLM**: 3-8s (single-turn, 100 tokens)
- **TTS**: 1-2s
- **RAM**: 4-6GB peak

## 🤝 License
MIT License

***

**Gunjan - Your offline voice companion** 🎙️🚀