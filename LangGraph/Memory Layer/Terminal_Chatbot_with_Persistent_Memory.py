"""
Terminal Chatbot with Persistent Memory (LangGraph + SQLite) 

- Terminal chatbot (no forced greetings)
- Persistent short-term memory per session (LangGraph SqliteSaver, keyed by thread_id)
- Persistent long-term memory per user (SQLite table: user_facts)
- Structured fact extraction using LLM + Pydantic
- Colored terminal logging + single log file per run in logs/
- Optional Mermaid graph saved to graphs/ (PNG + MD)
"""

import logging
import os
import re
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, List, Optional, TypedDict

from colorama import Fore, Style, init
from pydantic import BaseModel, Field

from prompt_toolkit import PromptSession
from prompt_toolkit.application import run_in_terminal
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding.bindings.basic import load_basic_bindings

from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages


# Colored output for console logs
init(autoreset=True)

# ────────────────────────────────────────────────
# Config
# ────────────────────────────────────────────────
MODEL_NAME = "llama3.2:3b"
DB_PATH = "agent_state.db"

LOG_DIR = Path("logs")
GRAPH_DIR = Path("graphs")
HISTORY_DIR = Path("history")
for d in (LOG_DIR, GRAPH_DIR, HISTORY_DIR):
    d.mkdir(exist_ok=True)


# ────────────────────────────────────────────────
# Logger
# ────────────────────────────────────────────────
def setup_logger(user_id: str) -> logging.Logger:
    log = logging.getLogger(f"agent_{user_id}")
    log.setLevel(logging.INFO)

    # Avoid duplicate handlers if re-run in same interpreter
    if log.handlers:
        for h in list(log.handlers):
            log.removeHandler(h)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOG_DIR / f"agent_memory_{user_id}_{timestamp}.log"

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    ))
    log.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(
        "%(asctime)s %(message)s",
        datefmt="%H:%M:%S"
    ))

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


def log_divider(log: logging.Logger, width: int = 60) -> None:
    log.info("─" * width)


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
        "  /clear    Clear screen\n"
        "  /new      Start a new session (new thread_id)\n"
        "  /exit     Quit\n\n"
        "Keys:\n"
        "  Enter       New line (multiline input)\n"
        "  Alt+Enter   Send message\n"
        "  Up/Down     History\n"
        "  Ctrl+L      Clear screen\n"
        "  F1          Help\n"
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
            # Alt+Enter => submit multiline
            text = event.app.current_buffer.text
            # ensure stored in history for custom submit
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
# SQLite (long-term facts)
# ────────────────────────────────────────────────
def setup_facts_table(conn: sqlite3.Connection, log: logging.Logger) -> None:
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
    log.info("Ensured table: user_facts")


# ────────────────────────────────────────────────
# Fact extraction schema
# ────────────────────────────────────────────────
class UserFact(BaseModel):
    key: str = Field(..., description="Fact key (name, location, profession, etc)")
    value: str = Field(..., description="Extracted value")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence 0.0–1.0")
    source_text: str = Field(..., description="Relevant part of user message")


class ExtractedFacts(BaseModel):
    facts: List[UserFact] = Field(default_factory=list)


# ────────────────────────────────────────────────
# LLM + extraction chain
# ────────────────────────────────────────────────
llm = ChatOllama(model=MODEL_NAME, temperature=0.0)

extraction_prompt = ChatPromptTemplate.from_messages([
    ("system", """Extract only explicit, clear facts from the user message.
Do NOT guess, infer, or hallucinate. Return empty list if nothing is clear.
Confidence must be between 0.0 and 1.0 (e.g. 0.95 for very clear facts).
Examples of keys: name, location, hometown, profession, field_of_work, etc."""),
    ("human", "{message}"),
])

extraction_chain = extraction_prompt | llm.with_structured_output(ExtractedFacts)


# ────────────────────────────────────────────────
# LangGraph state
# ────────────────────────────────────────────────
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    user_id: str
    ltm_context: str


# ────────────────────────────────────────────────
# Nodes
# ────────────────────────────────────────────────
def make_retrieve_memories(conn: sqlite3.Connection, log: logging.Logger):
    def retrieve_memories(state: AgentState) -> dict:
        user_id = state["user_id"]
        try:
            rows = conn.execute(
                "SELECT key, value, confidence FROM user_facts WHERE user_id=? ORDER BY key",
                (user_id,),
            ).fetchall()
        except Exception:
            log.exception("Failed reading user_facts (user_id=%s)", user_id)
            return {"ltm_context": "(no long-term profile yet)"}

        if not rows:
            return {"ltm_context": "(no long-term profile yet)"}

        lines = [f"{k}: {v} (conf: {float(c):.2f})" for (k, v, c) in rows]
        return {"ltm_context": "\n".join(lines)}

    return retrieve_memories


def chatbot(state: AgentState) -> dict:
    system_lines = [
        "You are a helpful, memory-aware assistant.",
        "When the user asks about themselves (name, where they live, origin, profession, background, etc.)",
        "→ ALWAYS check the LONG-TERM MEMORY / USER PROFILE section first.",
        "→ If the information is there → answer directly using it.",
        "→ Only if the profile has no information on the topic → say you don't know yet.",
    ]

    if state.get("ltm_context"):
        system_lines.extend([
            "",
            "──────────────────────────────────────",
            "LONG-TERM MEMORY / USER PROFILE:",
            state["ltm_context"],
            "──────────────────────────────────────",
            "",
        ])

    messages = [SystemMessage(content="\n".join(system_lines)), *state["messages"]]
    response = llm.invoke(messages)
    return {"messages": [response]}


def make_save_memories(conn: sqlite3.Connection, log: logging.Logger):
    def save_memories(state: AgentState) -> dict:
        last_human = next(
            (m.content for m in reversed(state["messages"]) if getattr(m, "type", None) == "human"),
            None,
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
                src = (fact.source_text or "")[:150]
                ts = datetime.now(timezone.utc).isoformat()

                row = conn.execute(
                    "SELECT confidence FROM user_facts WHERE user_id=? AND key=?",
                    (user_id, key),
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

        except Exception:
            log.exception("Failed writing user_facts (user_id=%s)", user_id)
            return {}

        if writes:
            log.info("Saved/updated %d fact(s) for user=%s", writes, user_id)

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


def save_graph_png(graph, log: logging.Logger) -> None:
    md_path = GRAPH_DIR / "agent_graph_latest.md"
    png_path = GRAPH_DIR / "agent_graph_latest.png"
    if md_path.exists() and png_path.exists():
        return

    try:
        mermaid_text = graph.get_graph().draw_mermaid()
        md_path.write_text(f"```mermaid\n{mermaid_text}\n```", encoding="utf-8")

        png_bytes = graph.get_graph().draw_mermaid_png()
        png_path.write_bytes(png_bytes)

        log.info("Saved graph visualization: %s, %s", md_path, png_path)
    except Exception:
        log.warning("Graph visualization skipped (missing deps or unsupported environment)")


# ────────────────────────────────────────────────
# Run one turn
# ────────────────────────────────────────────────
def run_turn(*, graph, log: logging.Logger, user_id: str, thread_id: str, text: str) -> Optional[str]:
    config = {"configurable": {"thread_id": thread_id}}
    inputs = {
        "messages": [("human", text)],
        "user_id": user_id,
        "ltm_context": "",
    }

    last_ai = None
    try:
        for event in graph.stream(inputs, config, stream_mode="values"):
            msgs = event.get("messages") or []
            if not msgs:
                continue
            last = msgs[-1]
            if getattr(last, "type", None) == "ai":
                last_ai = (last.content or "").strip()
    except Exception:
        log.exception("graph.stream failed (user=%s, thread=%s)", user_id, thread_id)
        return None

    return last_ai


# ────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────
def main():
    user_id = normalize_user(input("Username: "))
    log = setup_logger(user_id)
    log.info("Model=%s", MODEL_NAME)

    thread_id = make_thread_id(user_id)
    log.info("Session thread_id=%s", thread_id)

    conn = None
    try:
        conn = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=15)
        log.info("SQLite connected: %s", DB_PATH)

        checkpointer = SqliteSaver(conn)
        checkpointer.setup()
        log.info("SqliteSaver ready")

        setup_facts_table(conn, log)

        graph = build_graph(checkpointer, conn, log)
        log.info("Graph compiled")
        save_graph_png(graph, log)

        ui = TerminalChatUI(user_id=user_id, log=log)

        log_divider(log)
        print(help_text())
        log_divider(log)

        while True:
            try:
                text = ui.prompt(thread_id)
            except (EOFError, KeyboardInterrupt):
                print()
                log.info("Exit requested")
                break

            if not text:
                continue

            cmd = text.lower()
            if cmd in {"/exit", "exit", "quit"}:
                log.info("Exit command received")
                break

            if cmd in {"/help", "help"}:
                print(help_text())
                continue

            if cmd in {"/clear", "clear"}:
                clear_screen()
                continue

            if cmd in {"/new", "new"}:
                thread_id = make_thread_id(user_id)
                log.info("New session thread_id=%s", thread_id)
                continue

            log.info("User: %s", text)
            reply = run_turn(graph=graph, log=log, user_id=user_id, thread_id=thread_id, text=text)

            if not reply:
                log.warning("No reply produced")
                continue

            log.info("AI: %s", reply)
            print(f"AI: {reply}")

    except Exception:
        log.exception("Fatal error")
        raise

    finally:
        if conn is not None:
            try:
                conn.close()
                log.info("SQLite connection closed")
            except Exception:
                log.exception("Failed to close SQLite connection")


if __name__ == "__main__":
    main()
