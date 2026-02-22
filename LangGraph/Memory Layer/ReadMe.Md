# Persistent Memory Chatbot (LangGraph + SQLite)

A professional, minimal repository demonstrating how to build a **terminal chatbot with persistent memory** using **LangGraph** and **SQLite**.

This repo focuses on three memory patterns—**RAM-only**, **hybrid (RAM + JSON)**, and **fully persistent (SQLite)**—and shows exactly what changes in behavior when you switch the checkpointer and long‑term storage.

---

## Highlights

- **Persistent short-term memory (per session)** using LangGraph **SQLite checkpointer** (`SqliteSaver`).
- **Persistent long-term memory (per user)** stored in SQLite (`user_facts` table).
- **Improved terminal UX** using `prompt_toolkit` (multiline input, editing, command history, built-in keybindings).
- Graph visualization export (Mermaid → PNG/MD) to understand the workflow.

> In LangGraph, checkpoint persistence is keyed by `thread_id`; you must pass `{"configurable": {"thread_id": ...}}` during execution for checkpoints to be stored and later resumed. [web:1]

---

## Repository Structure

### Production-style chatbot (recommended)
- `Terminal_Chatbot_with_Persistent_Memory.py`  
  Terminal chatbot with:
  - session persistence via `SqliteSaver` (SQLite checkpoints)
  - user-wise long-term facts via SQLite table `user_facts`
  - CLI experience via `prompt_toolkit`

### Utilities
- `inspect_checkpoints.py`  
  Inspect checkpoints inside `agent_state.db` for debugging and validation.

### Learning / comparison scripts
- `short_term_momory_chatbot.py`  
  Demonstrates short-term memory with `MemorySaver` (RAM-only). Memory exists only within the same `thread_id` and is lost on restart.
- `hybrid_memory_chatbot_json_prefs.py`  
  Demonstrates a hybrid approach: short-term memory in RAM + long-term user preferences in JSON files (`user_preferences/`).

---

## Quick Start

### 1) Install dependencies
```bash
pip install langgraph langchain-core langchain-ollama pydantic colorama prompt_toolkit
```

### 2) Start Ollama and pull a model
Example:
```bash
ollama pull llama3.2:3b
```

### 3) Run the persistent chatbot
```bash
python3 Terminal_Chatbot_with_Persistent_Memory.py
```

---

## How Memory Works (in this repo)

### A) Short-term memory (conversation history)
Short-term memory is handled by the **LangGraph checkpointer**, and is stored **per session**.

- A session is identified by `thread_id`
- Same `thread_id` → same conversation history continues
- New `thread_id` → new conversation history

LangGraph uses `thread_id` as the primary key for storing and retrieving checkpoints. [web:1]

### B) Long-term memory (user profile / facts)
Long-term memory is stored **per user** in SQLite so it persists across sessions and restarts.

Database:
- `agent_state.db` (SQLite)

Table:
- `user_facts(user_id, key, value, confidence, source, updated_at)`

This allows:
- New session (`/new`) to start a fresh chat history
- But still retain user identity and facts across sessions

---

## Script Comparison

| Script | Checkpointer (short-term) | Long-term storage | Persists after restart? | Typical use case |
|---|---|---|---|---|
| `short_term_momory_chatbot.py` | `MemorySaver` (RAM) | None | No | Understand `thread_id` + reducers |
| `hybrid_memory_chatbot_json_prefs.py` | `MemorySaver` (RAM) | JSON files | Profile: Yes, chat: No | Simple “remember user” demos |
| `Terminal_Chatbot_with_Persistent_Memory.py` | `SqliteSaver` (SQLite) | SQLite table (`user_facts`) | Yes | Real terminal chatbot memory |

---

## Terminal Commands (persistent chatbot)

Inside `Terminal_Chatbot_with_Persistent_Memory.py`:

- `/help`  Show commands and keybindings
- `/clear` Clear the screen
- `/new`   Start a new session (new `thread_id`)
- `/exit`  Quit

The CLI uses `prompt_toolkit` to support multiline input and persistent command history through `PromptSession` + `FileHistory`. [web:24][web:46]

---

## Outputs and Artifacts

When you run the persistent chatbot, these are created:

- `agent_state.db`  
  SQLite DB containing LangGraph checkpoints and the `user_facts` table.

- `logs/`  
  One log file per run.

- `graphs/`  
  Mermaid graph exports (PNG/MD), if enabled.

- `history/`  
  Per-user terminal input history files (for Up/Down navigation).

---

## Debugging & Verification

### Inspect checkpoints
```bash
python3 inspect_checkpoints.py
```

This helps verify that:
- checkpoints are written under the correct `thread_id`
- messages are accumulating as expected
- your state keys match what the graph expects

---

## Recommended Next Improvements

- Add `/profile` command to print current `user_facts` for the logged-in user.
- Add `/sessions` and `/resume` to list and resume previous `thread_id`s.
- Replace heuristic “fact extraction” with structured extraction (Pydantic) for better accuracy and confidence scoring.

---

## License
Add a license file if you plan to publish (MIT is common for demo repos).
