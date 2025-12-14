# Gunjan Voice Assistant

A local, offline voice assistant powered by **Gemma-2-2B** (LlamaCpp), **Whisper** for speech-to-text, and **gTTS** for text-to-speech. Features conversational memory, wake-word detection ("stop"), and a FastAPI web interface.

## ✨ Features

- **Fully Offline LLM**: Gemma-2-2B-IT Q4_K_M (2.6GB GGUF model)
- **Whisper Transcription**: Accurate speech-to-text with `openai/whisper-base.en`
- **Audio Classification**: Detect "stop" command using MIT AST model
- **Conversational Memory**: LangChain `ConversationBufferMemory`
- **Web Interface**: FastAPI server on `http://localhost:8000`
- **Jupyter/IPython Support**: Inline audio playback
- **Clean Logging**: Structured logs with rotation

## 🛠️ Quick Start

### 1. Clone & Setup
```bash
git clone <your-repo>
cd voice-assistant
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate    # Windows
pip install -r requirements.txt
```

### 2. Download Model
```bash
mkdir -p models
# Download Gemma-2-2B-IT Q4_K_M.gguf (~2.6GB)
wget -O models/gemma-2-2b-it-Q4_K_M.gguf \
  "https://huggingface.co/bartowski/gemma-2-2b-it-GGUF/resolve/main/gemma-2-2b-it-Q4_K_M.gguf"
```

### 3. Fix Audio Dependencies (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install python3-dev portaudio19-dev build-essential alsa-utils
pip install pyaudio
```

### 4. Run Standalone
```bash
python Voice_assistent.py
```
*Say something → Hear response → Say "STOP" to exit*

### 5. Run Web Server
```bash
python server.py
```
*Visit `http://localhost:8000`*

## 📁 Project Structure

```
├── Voice_assistent.py      # Core voice chatbot class
├── server.py              # FastAPI web server
├── logger_config.py       # Structured logging
├── models/                # GGUF model files
├── uploads/              # Temporary audio files
├── logs/                 # Rotated log files
├── templates/            # HTML templates
├── static/               # CSS/JS assets
└── requirements.txt      # Dependencies
```

## 🚀 API Endpoints

| Method | Endpoint              | Description |
|--------|-----------------------|-------------|
| `GET`  | `/`                  | Web UI |
| `POST` | `/process_audio`     | Upload audio → Get transcription |
| `POST` | `/get_response`      | Text → LLM response |
| `POST` | `/synthesize_speech` | Text → MP3 filename |
| `GET`  | `/audio/{filename}`  | Serve generated speech |

## 🔧 Dependencies

```txt
torch
SpeechRecognition
gTTS
playsound
langchain-core
langchain-groq
langchain-community
ipython
transformers
datasets
pydub
tf-keras
soundfile
sentencepiece
llama-cpp-python
librosa
pygame
```

## ⚙️ Configuration

| Env Var | Default | Purpose |
|---------|---------|---------|
| `CUDA_VISIBLE_DEVICES` | `""` | Force CPU (MX250 workaround) |
| `LANGCHAIN_TRACING_V2` | `"false"` | Disable LangSmith |
| `TF_CPP_MIN_LOG_LEVEL` | `"2"` | Quiet TensorFlow |

## 🎵 Audio Pipeline

```
Microphone → speech_recognition → WAV file
    ↓
Whisper (transcriber) → Text
    ↓
Gemma-2-2B (LLM) → Response text
    ↓
gTTS → MP3 → pydub playback
```

## 🔇 Suppress Noisy Warnings

✅ **ALSA/JACK**: Suppressed via `stderr` redirect  
✅ **TensorFlow CPU**: `TF_CPP_MIN_LOG_LEVEL=2`  
✅ **PyTorch CUDA**: Forced CPU-only  
✅ **LangChain**: Tracing disabled  

## 🖥️ Web Interface Workflow

1. **Record** → POST `/process_audio` → Get text
2. **Query LLM** → POST `/get_response` → Get response
3. **Generate Speech** → POST `/synthesize_speech` → Get MP3 filename
4. **Play** → GET `/audio/filename.mp3`

## 💻 Development

```bash
# Auto-reload server
uvicorn server:app --host 0.0.0.0 --port 8000 --reload

# Logs in logs/voicebot.log.*
tail -f logs/voicebot.log
```

## 📱 Frontend Example (JavaScript)

```javascript
// Record → Transcribe → LLM → TTS → Play
async function processVoice() {
    const audioBlob = await recordAudio();
    const formData = new FormData();
    formData.append('file', audioBlob, 'audio.wav');
    
    // 1. Transcribe
    const { text } = await fetch('/process_audio', { method: 'POST', body: formData }).then(r => r.json());
    
    // 2. LLM Response
    const { response_text } = await fetch('/get_response', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text })
    }).then(r => r.json());
    
    // 3. Speech
    const { audio_filename } = await fetch('/synthesize_speech', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: response_text })
    }).then(r => r.json());
    
    // 4. Play
    const audio = new Audio(`/audio/${audio_filename}`);
    audio.play();
}
```

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| **PyAudio build fails** | `sudo apt install portaudio19-dev python3-dev` |
| **ALSA warnings** | Already suppressed |
| **Model not found** | Check `models/gemma-2-2b-it-Q4_K_M.gguf` |
| **CUDA warnings** | MX250 unsupported → CPU fallback |
| **Port 8000 busy** | `sudo ss -tulpn \| grep :8000` + `kill <PID>` |

## 📈 Performance

- **Cold Start**: ~30s (model loading)
- **Transcription**: ~2-5s (Whisper base)
- **LLM Response**: ~3-8s (Gemma-2 2B, 4096 ctx)
- **TTS**: ~1-2s
- **RAM**: ~4-6GB peak

## 🤝 Contributing

1. Fork & PR
2. Add tests in `tests/`
3. Update `requirements.txt` for new deps


## 📄 License

MIT License - See [LICENSE](LICENSE) file.

***

**Built with ❤️ for offline voice AI**  
*Gunjan - Your local voice companion* 🚀