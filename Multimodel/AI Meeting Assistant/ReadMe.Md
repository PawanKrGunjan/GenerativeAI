# 🤖 AI Meeting Assistant

A **local, CPU-only AI Meeting Assistant** that transcribes meeting audio, normalizes financial terminology, and generates **professional meeting minutes with task assignments** — no cloud APIs, no keys required.

Powered by **Ollama (Qwen 2.5 3B)** and **Whisper Speech-to-Text**.

---

## ✨ Features

- 🎙 **Speech-to-Text** using OpenAI Whisper (CPU-optimized)
- 💰 **Financial Term Normalization**
  - Expands acronyms (ROA → Return on Assets (ROA))
  - Converts spoken numbers (four zero one k → 401(k) Retirement Savings Plan)
  - Context-aware disambiguation (LTV, EBITDA, etc.)
- 📝 **Structured Meeting Minutes**
  - Key discussion points
  - Decisions made
  - Task assignments table (assignee, deadline, priority)
- 📄 **Markdown Export**
- 🖥 **Fully Local**
  - No IBM Watsonx
  - No OpenAI API
  - No internet required after model download
- ⚙️ **CPU-only friendly** (8 GB RAM supported)

---

## 🧱 Architecture

```text
Audio File
   │
   ▼
Whisper STT (local, CPU)
   │
   ▼
Financial Terminology Normalizer (LLM)
   │
   ▼
Meeting Intelligence Generator (LLM)
   │
   ▼
Markdown Report + Download
````

---

## 📦 Requirements

### System

* Linux / macOS / Windows
* Python **3.9+**
* **8 GB RAM** recommended
* CPU-only (GPU optional but not required)

### Software

* Ollama (running locally)

---

## 🔧 Installation

### 1️⃣ Install Ollama

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### 2️⃣ Pull Required Models

```bash
ollama pull qwen2.5:3b
```

(Whisper models are downloaded automatically by Transformers.)

---

### 3️⃣ Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 4️⃣ Install Dependencies

```bash
pip install -U \
  torch \
  gradio>=4.0 \
  transformers \
  langchain-core \
  langchain-ollama
```

---

## ⚙️ Environment Configuration (IMPORTANT)

For **CPU-only and low-memory systems**, set the following environment variables **before running the app**:

```bash
export OLLAMA_NUM_PARALLEL=1
export OLLAMA_MAX_LOADED_MODELS=1
export TOKENIZERS_PARALLELISM=false
```

### Why this matters

* Prevents Ollama from loading multiple models simultaneously
* Reduces RAM pressure and swap usage
* Avoids tokenizer thread over-subscription
* Improves stability on 8 GB systems

Optional (Linux only):

```bash
sudo sysctl vm.swappiness=10
```

---

## 🚀 Running the Application

```bash
python speech_analyzer.py
```

Then open your browser at:

```
http://localhost:5000
```

---

## 🎧 Supported Audio Formats

* MP3
* WAV
* M4A
* FLAC

**Recommended:**

* Mono audio
* ≤ 30 minutes for best CPU performance

---

## 📄 Output

* On-screen **Markdown-formatted meeting minutes**
* Downloadable `.md` file stored in:

  ```text
  outputs/meeting_minutes_<audio_name>.md
  ```

---

## 🧠 Model Choices

### LLM (Ollama)

| Model         | Use Case                            |
| ------------- | ----------------------------------- |
| `qwen2.5:3b`  | ✅ Best balance of reasoning & speed |
| `llama3.2:1b` | Low-RAM fallback                    |

### Speech-to-Text

| Model            | RAM   | Accuracy        |
| ---------------- | ----- | --------------- |
| `whisper-tiny`   | ~1 GB | Fast            |
| `whisper-small`  | ~2 GB | ⭐ Recommended   |
| `whisper-medium` | ~5 GB | Best (slow CPU) |

---

## 🛠 Troubleshooting

### ❌ Whisper Long-Audio Error

> You have passed more than 3000 mel input features…

✅ **Solution:**
The app enables `return_timestamps=True`, which is required for long-form Whisper transcription.

---

### ❌ Ollama Not Responding

```bash
ollama ps
```

Ensure `qwen2.5:3b` is running.

---

## 🔒 Privacy

* All processing is **local**
* No data leaves your machine
* Suitable for confidential meetings

---

## 🧩 Project Structure

```text
.
├── speech_analyzer.py
├── outputs/
│   └── meeting_minutes_*.md
├── README.md
└── venv/
```

---

## 🛣 Roadmap (Optional Enhancements)

* Speaker diarization
* RAG over past meetings
* JSON / Pydantic outputs
* OpenVINO-accelerated Whisper
* FastAPI backend
* Streaming transcription

---

## 📜 License

MIT License — free to use, modify, and distribute.

---

## 🙌 Acknowledgements

* Ollama
* LangChain
* Hugging Face Transformers
* OpenAI Whisper

```

