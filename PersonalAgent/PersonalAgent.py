"""
Terminal Chatbot with Persistent Long-Term Memory + RAG (LangGraph + SQLite)

- Reads NEW files from ./data/ on startup and via /sync
- Stores full content + chunked embeddings in SQLite (document_chunks table)
- Manual cosine similarity RAG (no extra vector DB needed)
- Long-term user facts in user_facts table (structured extraction)
- Short-term conversation memory via LangGraph SqliteSaver
- Clean terminal UI with history, multiline, /commands
"""

import logging
import os
import re
import sqlite3
import uuid
import hashlib
from tqdm import tqdm
import numpy as np
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, List, TypedDict

from colorama import Fore, Style, init
from pydantic import BaseModel, Field

from prompt_toolkit import PromptSession
from prompt_toolkit.application import run_in_terminal
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding.bindings.basic import load_basic_bindings

import pandas as pd
from langchain_community.document_loaders import PyPDFLoader, CSVLoader, TextLoader
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages


# Colored output
init(autoreset=True)

# ────────────────────────────────────────────────
# Config
# ────────────────────────────────────────────────
MODEL_NAME = "llama3.2:3b"
DB_PATH = "agent_state.db"

LOG_DIR = Path("logs")
GRAPH_DIR = Path("graphs")
HISTORY_DIR = Path("history")
DATA_DIR = Path("data")

for d in (LOG_DIR, GRAPH_DIR, HISTORY_DIR, DATA_DIR):
    d.mkdir(exist_ok=True)

# ────────────────────────────────────────────────
# Fact extraction schema
# ────────────────────────────────────────────────
class UserFact(BaseModel):
    key: str = Field(..., description="Fact key (name, location, profession, etc)")
    value: str = Field(..., description="Extracted value")
    confidence: float = Field(..., ge=0.0, le=1.0)
    source_text: str = Field(..., description="Relevant part of user message")


class ExtractedFacts(BaseModel):
    facts: List[UserFact] = Field(default_factory=list)


# ────────────────────────────────────────────────
# LLM + Embeddings + Chains
# ────────────────────────────────────────────────
llm = ChatOllama(model=MODEL_NAME, temperature=0.0)
embeddings_model = OllamaEmbeddings(model="nomic-embed-text:latest")

extraction_prompt = ChatPromptTemplate.from_messages([
    ("system", """Extract ONLY explicit, clear facts from the user message.
Do NOT guess, infer, or hallucinate. Return empty list if nothing clear.
Confidence: 0.0-1.0 (e.g. 0.95 for obvious facts).
Keys examples: name, location, hometown, profession, field_of_work, hobby, etc."""),
    ("human", "{message}"),
])

extraction_chain = extraction_prompt | llm.with_structured_output(ExtractedFacts)


# ────────────────────────────────────────────────
# Logger
# ────────────────────────────────────────────────
def setup_logger(user_id: str) -> logging.Logger:
    log = logging.getLogger(f"agent_{user_id}")
    log.setLevel(logging.INFO)
    if log.handlers:
        for h in list(log.handlers):
            log.removeHandler(h)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOG_DIR / f"agent_memory_{user_id}_{timestamp}.log"

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    log.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S"))

    def colored_emit(record):
        msg = record.getMessage()
        if record.levelno >= logging.ERROR:
            record.msg = f"{Fore.RED}{msg}{Style.RESET_ALL}"
        elif record.levelno >= logging.WARNING:
            record.msg = f"{Fore.YELLOW}{msg}{Style.RESET_ALL}"
        else:
            record.msg = f"{Fore.GREEN}{msg}{Style.RESET_ALL}"
        record.args = ()
        return logging.StreamHandler.emit(console_handler, record)

    console_handler.emit = colored_emit
    log.addHandler(console_handler)
    log.propagate = False
    log.info("Logger initialized (file=%s)", log_file)
    return log


# ────────────────────────────────────────────────
# Terminal UI
# ────────────────────────────────────────────────
def normalize_user(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-zA-Z0-9_.-]", "", s)
    return s or "anonymous"


def make_thread_id(user_id: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = uuid.uuid4().hex[:8]
    return f"{user_id}:{ts}:{suffix}"


def clear_screen() -> None:
    os.system("cls" if os.name == "nt" else "clear")


def help_text() -> str:
    return (
        "Commands:\n"
        "  /help     Show help\n"
        "  /sync     Re-index ./data folder (new/updated files)\n"
        "  /clear    Clear screen\n"
        "  /new      Start new session (new thread_id)\n"
        "  /exit     Quit\n\n"
        "Keys:\n"
        "  Enter       New line\n"
        "  Alt+Enter   Send\n"
        "  Up/Down     History\n"
        "  Ctrl+L      Clear screen\n"
    )


class TerminalChatUI:
    def __init__(self, *, user_id: str, log: logging.Logger):
        self.user_id = user_id
        self.log = log
        hist_path = HISTORY_DIR / f"{user_id}.txt"
        self.session = PromptSession(
            history=FileHistory(str(hist_path)),
            auto_suggest=AutoSuggestFromHistory(),
        )
        self.kb = load_basic_bindings()

        @self.kb.add("f1")
        def _(event):
            run_in_terminal(lambda: print(help_text()))

        @self.kb.add("c-l")
        def _(event):
            run_in_terminal(clear_screen)

        @self.kb.add("escape", "enter")
        def _(event):
            text = event.app.current_buffer.text
            self.session.history.append_string(text)
            event.app.exit(result=text)

    def prompt(self, thread_id: str) -> str:
        return self.session.prompt(
            f"You[{thread_id}]: ",
            multiline=True,
            key_bindings=self.kb,
            enable_history_search=True,
        ).strip()


# ────────────────────────────────────────────────
# SQLite Setup
# ────────────────────────────────────────────────
def setup_facts_table(conn: sqlite3.Connection, log: logging.Logger):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS user_facts (
            user_id TEXT NOT NULL,
            key TEXT NOT NULL,
            value TEXT NOT NULL,
            confidence REAL NOT NULL,
            source TEXT,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (user_id, key)
        )
    """)
    conn.commit()
    log.info("Table ready: user_facts")


def setup_documents_table(conn: sqlite3.Connection, log: logging.Logger):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            file_path TEXT PRIMARY KEY,
            file_hash TEXT NOT NULL,
            content TEXT NOT NULL,
            indexed_at TEXT NOT NULL
        )
    """)
    conn.commit()
    log.info("Table ready: documents")


def setup_document_chunks_table(conn: sqlite3.Connection, log: logging.Logger):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS document_chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_path TEXT NOT NULL,
            chunk_text TEXT NOT NULL,
            embedding BLOB NOT NULL,
            chunk_index INTEGER,
            created_at TEXT NOT NULL
        )
    """)
    conn.commit()
    log.info("Table ready: document_chunks")


# ────────────────────────────────────────────────
# Document Processing
# ────────────────────────────────────────────────
def compute_file_hash(file_path: str) -> str:
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_document_content(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()
    try:
        if ext == ".pdf":
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            return "\n".join(d.page_content for d in docs)
        elif ext == ".csv":
            loader = CSVLoader(file_path)
            docs = loader.load()
            return "\n".join(d.page_content for d in docs)
        elif ext == ".xlsx":
            df = pd.read_excel(file_path)
            return df.to_string(index=False)
        elif ext == ".txt":
            loader = TextLoader(file_path, encoding="utf-8")
            docs = loader.load()
            return "\n".join(d.page_content for d in docs)
        return ""
    except Exception:
        return ""


def embed_text(text: str) -> np.ndarray:
    vector = embeddings_model.embed_query(text)
    return np.array(vector, dtype=np.float32)


text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)

def index_document_chunks(conn, log, file_path: str, content: str):
    conn.execute("DELETE FROM document_chunks WHERE file_path=?", (file_path,))
    filename = Path(file_path).name

    chunks = text_splitter.split_text(content)
    for i, chunk in enumerate(chunks):
        # ← KEY IMPROVEMENT
        enriched_chunk = f"Source: {filename}\n\n{chunk}"

        embedding = embeddings_model.embed_query(enriched_chunk)
        embedding_blob = np.array(embedding, dtype=np.float32).tobytes()

        conn.execute("""
            INSERT INTO document_chunks 
            (file_path, chunk_text, embedding, chunk_index, created_at)
            VALUES (?, ?, ?, ?, ?)
        """, (
            file_path, enriched_chunk, embedding_blob, i,
            datetime.now(timezone.utc).isoformat()
        ))
    conn.commit()
    log.info("Indexed %d chunks for %s", len(chunks), filename)


def sync_documents_to_db(conn, log, folder_path: str = "data"):
    log.info("Scanning data folder: %s", folder_path)
    try:
        files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    except FileNotFoundError:
        log.warning("Data folder not found (should not happen)")
        return

    for file in tqdm(files, desc="Indexing"):
        file_path = os.path.join(folder_path, file)
        if not os.path.isfile(file_path):
            continue

        file_hash = compute_file_hash(file_path)
        row = conn.execute("SELECT file_hash FROM documents WHERE file_path=?", (file_path,)).fetchone()

        if row and row[0] == file_hash:
            continue  # unchanged

        try:
            content = load_document_content(file_path)
            if not content.strip():
                continue

            ts = datetime.now(timezone.utc).isoformat()
            conn.execute("""
                INSERT INTO documents (file_path, file_hash, content, indexed_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(file_path) DO UPDATE SET
                    file_hash=excluded.file_hash,
                    content=excluded.content,
                    indexed_at=excluded.indexed_at
            """, (file_path, file_hash, content, ts))

            index_document_chunks(conn, log, file_path, content)
            log.info("✓ Indexed: %s", file)
        except Exception:
            log.exception("Failed to index %s", file)

    conn.commit()
    log.info("Document sync complete.")


def cosine_similarity(a, b):
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom != 0 else 0.0


def semantic_search(conn, query: str, top_k: int = 7) -> List[str]:
    if not query.strip():
        return []
    query_emb = np.array(embeddings_model.embed_query(query), dtype=np.float32)
    rows = conn.execute("SELECT chunk_text, embedding FROM document_chunks").fetchall()

    scored = []
    for chunk_text, emb_blob in rows:
        emb = np.frombuffer(emb_blob, dtype=np.float32)
        score = float(np.dot(query_emb, emb) / (np.linalg.norm(query_emb) * np.linalg.norm(emb) + 1e-8))
        scored.append((score, chunk_text))

    scored.sort(reverse=True, key=lambda x: x[0])
    return [text for _, text in scored[:top_k]]


# ────────────────────────────────────────────────
# LangGraph State & Nodes
# ────────────────────────────────────────────────
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    user_id: str
    ltm_context: str


def make_retrieve_memories(conn, log):
    def retrieve(state: AgentState) -> dict:
        user_id = state["user_id"]
        fact_rows = conn.execute(
            "SELECT key, value, confidence FROM user_facts WHERE user_id=?",
            (user_id,)
        ).fetchall()
        facts = "\n".join([f"{k}: {v} (conf:{c:.2f})" for k,v,c in fact_rows]) or "(no profile yet)"

        last_human = next((m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), "")
        doc_chunks = semantic_search(conn, last_human, top_k=7)
        docs = "\n\n".join(doc_chunks) or "(no relevant documents)"

        context = f"""USER PROFILE:\n{facts}\n\nRELEVANT DOCUMENTS:\n{docs}"""
        return {"ltm_context": context}
    return retrieve


def chatbot(state: AgentState) -> dict:
    system = """You are a precise, memory-aware assistant.

RULES:
- These are the USER'S OWN documents. NEVER redact, censor, or use stars (****) for salary, dates, names, or any numbers.
- ALWAYS ground your answer in the REFERENCE MEMORY.
- Quote exact values and mention the Source filename.
- If information is in the context → give it directly and precisely.
- If not present → say "I don't have that information in your documents."

REFERENCE MEMORY START
{context}
REFERENCE MEMORY END"""

    context = state.get("ltm_context", "")
    messages = [SystemMessage(content=system.format(context=context)), *state["messages"]]
    return {"messages": [llm.invoke(messages)]}


def make_save_memories(conn: sqlite3.Connection, log: logging.Logger):
    def save_memories(state: AgentState) -> dict:
        last_human = next(
            (m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
            None
        )
        if not last_human:
            return {}

        try:
            result: ExtractedFacts = extraction_chain.invoke({"message": last_human})
        except Exception:
            log.exception("Fact extraction failed")
            return {}

        if not result.facts:
            return {}

        user_id = state["user_id"]
        writes = 0
        try:
            for fact in result.facts:
                key = (fact.key or "").strip()
                val = (fact.value or "").strip()
                if not key or not val:
                    continue

                conf = max(min(float(fact.confidence), 1.0), 0.0)
                src = (fact.source_text or "")[:200]
                ts = datetime.now(timezone.utc).isoformat()

                row = conn.execute(
                    "SELECT confidence FROM user_facts WHERE user_id=? AND key=?",
                    (user_id, key)
                ).fetchone()
                old_conf = float(row[0]) if row else -1.0

                if conf > old_conf:
                    conn.execute("""
                        INSERT INTO user_facts (user_id, key, value, confidence, source, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?)
                        ON CONFLICT(user_id, key) DO UPDATE SET
                            value=excluded.value,
                            confidence=excluded.confidence,
                            source=excluded.source,
                            updated_at=excluded.updated_at
                    """, (user_id, key, val, conf, src, ts))
                    writes += 1

            conn.commit()
            if writes:
                log.info("Saved/updated %d fact(s) for user=%s", writes, user_id)
        except Exception:
            log.exception("Failed writing user_facts")
        return {}

    return save_memories


# ────────────────────────────────────────────────
# Graph
# ────────────────────────────────────────────────
def build_graph(checkpointer: SqliteSaver, conn: sqlite3.Connection, log: logging.Logger):
    workflow = StateGraph(state_schema=AgentState)
    workflow.add_node("retrieve_memories", make_retrieve_memories(conn, log))
    workflow.add_node("chatbot", chatbot)
    workflow.add_node("save_memories", make_save_memories(conn, log))

    workflow.add_edge(START, "retrieve_memories")
    workflow.add_edge("retrieve_memories", "chatbot")
    workflow.add_edge("chatbot", "save_memories")
    workflow.add_edge("save_memories", END)

    return workflow.compile(checkpointer=checkpointer)


def save_graph_png(graph, log: logging.Logger):
    try:
        md_path = GRAPH_DIR / "agent_graph_latest.md"
        png_path = GRAPH_DIR / "agent_graph_latest.png"
        if md_path.exists() and png_path.exists():
            return

        mermaid = graph.get_graph().draw_mermaid()
        md_path.write_text(f"```mermaid\n{mermaid}\n```", encoding="utf-8")
        png_path.write_bytes(graph.get_graph().draw_mermaid_png())
        log.info("Graph visualization saved")
    except Exception:
        log.warning("Graph PNG/MD skipped (optional)")


# ────────────────────────────────────────────────
# Run one turn
# ────────────────────────────────────────────────
def run_turn(*, graph, log: logging.Logger, user_id: str, thread_id: str,
             text: str, conn: sqlite3.Connection) -> str | None:
    config = {"configurable": {"thread_id": thread_id}}
    inputs = {
        "messages": [HumanMessage(content=text)],
        "user_id": user_id,
        "ltm_context": "",
    }

    try:
        last_ai = None
        for event in graph.stream(inputs, config, stream_mode="values"):
            if "messages" in event and event["messages"]:
                last_msg = event["messages"][-1]
                if isinstance(last_msg, AIMessage):
                    last_ai = (last_msg.content or "").strip()
        return last_ai
    except Exception:
        log.exception("Graph execution failed")
        return None


# ────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────
def main():
    # ── Get username ───────────────────────────────────────────────
    while True:
        raw_input = input("Username (press Enter for anonymous): ").strip()
        user_id = normalize_user(raw_input)
        if user_id:
            break
        print("Please enter a username or press Enter for anonymous.")

    log = setup_logger(user_id)
    log.info("Starting agent | model=%s | embedding=nomic-embed-text", MODEL_NAME)

    # ── SQLite connection ──────────────────────────────────────────
    conn = None
    try:
        conn = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=20)
        log.info("Connected to SQLite: %s", DB_PATH)

        checkpointer = SqliteSaver(conn)
        checkpointer.setup()

        setup_facts_table(conn, log)
        setup_documents_table(conn, log)
        setup_document_chunks_table(conn, log)

        DATA_DIR.mkdir(exist_ok=True)
        log.info("Data directory: %s", DATA_DIR)

        # Initial document indexing
        print("→ Scanning and indexing documents in ./data/ ...")
        sync_documents_to_db(conn, log, str(DATA_DIR))
        print("→ Document indexing complete.\n")

        # Build graph once
        graph = build_graph(checkpointer, conn, log)
        log.info("Graph compiled successfully")
        save_graph_png(graph, log)

        ui = TerminalChatUI(user_id=user_id, log=log)

        # ── Print welcome / help once ──────────────────────────────
        print("\n" + "="*70)
        print("  Personal RAG + Memory Agent  |  Username:", user_id)
        print("  Commands:  /help   /sync   /docs   /new   /clear   /exit")
        print("="*70 + "\n")
        print(help_text())

        def log_divider(width=70):
            log.info("─" * width)
            print("─" * width)

        log_divider()

        # ── Main interaction loop ──────────────────────────────────
        thread_id = make_thread_id(user_id)
        log.info("Started new session: %s", thread_id)

        while True:
            try:
                text = ui.prompt(thread_id)
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye.")
                log.info("Exit requested by user")
                break

            text = text.strip()
            if not text:
                continue

            cmd = text.lower()

            if cmd in {"/exit", "exit", "quit", "q"}:
                print("Shutting down...")
                break

            elif cmd in {"/help", "help", "?"}:
                print(help_text())
                continue

            elif cmd in {"/clear", "clear"}:
                clear_screen()
                continue

            elif cmd in {"/new", "new"}:
                thread_id = make_thread_id(user_id)
                log.info("New conversation thread started: %s", thread_id)
                print(f"\n→ New session started (thread: {thread_id})\n")
                continue

            elif cmd in {"/sync", "sync", "/index"}:
                print("→ Re-indexing ./data/ folder...")
                sync_documents_to_db(conn, log, str(DATA_DIR))
                print("→ Re-indexing complete.\n")
                continue

            # Normal message → run agent
            log.info("User: %s", text)

            reply = run_turn(
                graph=graph,
                log=log,
                user_id=user_id,
                thread_id=thread_id,
                text=text,
                conn=conn
            )

            if reply:
                log.info("AI : %s", reply)
                print(f"AI : {reply}\n")
            else:
                log.warning("No reply generated for input: %s", text)
                print("AI : (no response)\n")

            log_divider(60)

    except sqlite3.Error as e:
        log.error("SQLite error: %s", e)
        print(f"Database error: {e}")
    except Exception as e:
        log.exception("Unexpected error in main loop")
        print(f"Error: {e}")
    finally:
        if conn is not None:
            try:
                conn.commit()           # just in case
                conn.close()
                log.info("SQLite connection closed")
            except Exception as close_err:
                log.error("Error closing SQLite: %s", close_err)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user. Exiting cleanly.")