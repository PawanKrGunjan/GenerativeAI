"""
personal_agent
==============

A local personal AI agent with long-term memory (facts + user profile),
RAG over your private documents (PDFs, Excel, TXT, CSV), terminal UI,
and persistent conversation threads.

Features:
- Ollama-based LLM + embeddings
- SQLite for facts, document metadata & chunked vector storage
- LangGraph for agent workflow (retrieve → generate → save facts)
- Terminal chat interface with history and keybindings

Usage:
    python run.py

Structure:
    Agent/     ← package
    ├── __init__.py
    ├── config.py
    ├── logger.py
    ├── schemas.py
    ├── llm.py
    ├── database.py
    ├── ui.py
    ├── graph.py
    └── run.py          ← main entry point


Quick imports:

    from Agent import llm, embeddings_model, MODEL_NAME
    from Agent import sync_documents_to_db, semantic_search
    from Agent import TerminalChatUI, build_graph, run_turn

Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "Pawan Kumar Gunjan"
__description__ = (
    "Personal long-term memory + document-aware agent " "(Ollama + SQLite + LangGraph)"
)
