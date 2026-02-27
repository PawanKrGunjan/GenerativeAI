# 🧠 Offline Personal Identity Agent

An **offline, privacy-first AI identity agent** that answers exactly like the user by learning from:

* 📄 User’s private documents
* 💬 Past chat history
* 🧠 Long-term memory (PostgreSQL + pgvector)

No cloud.
No data sharing.
Fully local.

---

## 🚀 Overview

The **Offline Personal Identity Agent** is designed to:

* Speak exactly like the user
* Maintain consistent tone and personality
* Retrieve relevant information from private documents
* Retrieve similar past conversations
* Operate 100% offline
* Store everything locally in PostgreSQL

This agent represents the user — not an AI assistant.

---

## 🏗 Architecture

Built using:

* **LangGraph** – workflow orchestration
* **PostgreSQL + pgvector** – vector database
* **Local LLM** – fully offline inference
* **Local embeddings model** – 768-dimension vectors
* **Semantic search (RAG)** – document + chat retrieval

---

### 🔁 Graph Flow

```
START
  ↓
Retrieve Memory
  - Document Chunks
  - Past Chat Messages
  ↓
Generate Identity Response
  - Impersonates user
  - Uses reference memory
  ↓
Save Chat History
  ↓
END
```

---

## 📂 Project Structure

```
PersonalAgent/
│
├── Agent/
│   ├── config.py              # Configuration & DB settings
│   ├── db_postgresql.py       # PostgreSQL + pgvector logic
│   ├── db_sqlite.py           # Optional SQLite support
│   ├── document_reader.py     # Document loading & chunking
│   ├── graph.py               # LangGraph workflow
│   ├── llm.py                 # Local LLM + embeddings
│   ├── logger.py              # Logging setup
│   ├── schemas.py             # DB schemas
│   ├── ui.py                  # Optional UI layer
│   └── __init__.py
│
├── data/                      # User private documents
├── logs/                      # Log files
├── run.py                     # Entry point
└── README.md
```

---

## 🔐 Privacy Design

* No external API calls
* No OpenAI usage
* No cloud storage
* No telemetry
* All embeddings computed locally
* All chats stored locally

The system is designed for **complete data sovereignty**.

---

## 🛠 Requirements

### 1️⃣ Python

Python 3.10+

### 2️⃣ PostgreSQL with pgvector

Install PostgreSQL and enable vector extension:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

### 3️⃣ Install Dependencies

Inside virtual environment:

```bash
pip install -r requirements.txt
```

Typical dependencies:

```
langchain
langgraph
psycopg[binary]
pgvector
tqdm
```

---

## ⚙️ Setup

### 1️⃣ Configure Database

Edit:

```
Agent/config.py
```

Set:

```python
DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "personal_agent",
    "user": "postgres",
    "password": "your_password"
}
```

---

### 2️⃣ Add User Documents

Place private files inside:

```
data/
```

Supported formats depend on `document_reader.py`.

---

### 3️⃣ Index Documents

On first run, documents are:

* Loaded
* Split into chunks
* Embedded
* Stored in PostgreSQL

---

## ▶️ Running the Agent

From project root:

```bash
python3 run.py
```

Example:

```
User: Hi, how are you?
Agent: I'm good. What's up?
```

The agent will:

1. Retrieve relevant memory
2. Respond like the user
3. Save the chat for future learning

---

## 🧠 How Identity Imitation Works

The system prompt enforces:

* You are the user
* Never reveal you are AI
* Use past writing style
* Ground answers in memory
* Stay concise and natural

It retrieves:

* Document chunks (semantic search)
* Past user messages (style learning)

This avoids assistant-style contamination.

---

## 🗄 Database Tables

### 📄 documents

Stores raw file content + metadata

### 📑 document_chunks

Stores chunk embeddings (VECTOR(768))

### 💬 chat_history

Stores:

* user_id
* human_text
* ai_text
* embeddings

Optimized with `ivfflat` index for fast semantic search.

---

## 📊 Graph Visualization

Graph diagram is automatically saved:

```
/graph/personalAgent.md
/graph/personalAgent.png
```

Requires `pyppeteer` for PNG generation.

---

## 🔎 Key Features

* ✅ Offline LLM
* ✅ Offline embeddings
* ✅ PostgreSQL + pgvector
* ✅ Semantic RAG
* ✅ Identity impersonation
* ✅ Self-learning from conversations
* ✅ Human-like responses
* ✅ Privacy-first design

---

## 🧪 Example Use Cases

* Personal digital twin
* Executive email responder
* Knowledge assistant
* Offline private AI
* Persona-consistent chatbot

---

## ⚠️ Important Notes

This agent:

* Will mimic writing tone
* Will reuse phrasing patterns
* Will reference private documents
* Should not be exposed publicly without safeguards

It is designed for **private local usage only**.

---

## 🔮 Future Enhancements

Possible improvements:

* Style fingerprint modeling
* Emoji distribution learning
* Sentence-length matching
* Response timing simulation
* Multi-persona support
* Encrypted local database

---

## 📜 License

Private / Internal Use
