"""
hybrid_memory_chatbot_json_prefs.py

- Short-term memory: LangGraph MemorySaver (RAM) per thread_id
- Long-term memory: per-user JSON preferences on disk
- Minimal LangGraph graph: load_prefs -> chatbot -> save_prefs
"""

import json
import logging
import re
from pathlib import Path
from typing import Annotated, TypedDict, Optional, Dict, Any

from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages


# ────────────────────────────────────────────────
# Logging & folders
# ────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("hybrid_memory_chatbot")

PREFS_DIR = Path("history")
PREFS_DIR.mkdir(exist_ok=True)


def prefs_path(user_id: str) -> Path:
    return PREFS_DIR / f"{user_id}.json"


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("Failed to read JSON: %s (%s)", path, e)
        return {}


def write_json_atomic(path: Path, data: Dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def last_human_text(messages) -> Optional[str]:
    # Robust: find the most recent human message (don’t rely on fixed indexes)
    for m in reversed(messages):
        if getattr(m, "type", None) == "human":
            return (m.content or "").strip()
    return None


# ────────────────────────────────────────────────
# State
# ────────────────────────────────────────────────
class ChatState(TypedDict):
    messages: Annotated[list, add_messages]
    user_id: str


# ────────────────────────────────────────────────
# LLM
# ────────────────────────────────────────────────
llm = ChatOllama(model="llama3.2:1b", temperature=0.7)


# ────────────────────────────────────────────────
# Nodes
# ────────────────────────────────────────────────
def load_prefs(state: ChatState) -> dict:
    """Load long-term preferences from JSON and add a greeting message."""
    user_id = state["user_id"]
    path = prefs_path(user_id)
    prefs = read_json(path)

    name = prefs.get("name")
    location = prefs.get("location")

    if name and location:
        greeting = f"Welcome back, {name}! How's everything in {location} today?"
    elif name:
        greeting = f"Welcome back, {name}! What are we working on today?"
    else:
        greeting = "Hello! Nice to meet you — tell me something about yourself."

    # Use role "ai" so downstream checks like last.type == "ai" work consistently.
    return {"messages": [("ai", greeting)]}


def chatbot(state: ChatState) -> dict:
    """Main chat node."""
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


def extract_and_save_prefs(state: ChatState) -> dict:
    """Very basic preference extraction from the most recent human message."""
    user_id = state["user_id"]
    text = last_human_text(state["messages"])
    if not text:
        return {}

    lower = text.lower()
    new_prefs: Dict[str, str] = {}

    # Extremely simple heuristics (replace with LLM structured extraction later)
    m = re.search(r"\bmy name is\s+([a-zA-Z][a-zA-Z0-9_-]{1,30})\b", lower)
    if m:
        new_prefs["name"] = m.group(1).title()
    elif "i'm " in lower or "i am " in lower or "call me " in lower:
        # Placeholder behavior (keep minimal, avoid wrong extraction)
        pass

    if "delhi" in lower:
        new_prefs["location"] = "Delhi"

    if not new_prefs:
        return {}

    path = prefs_path(user_id)
    current = read_json(path)
    current.update(new_prefs)

    try:
        write_json_atomic(path, current)
        logger.info("Saved prefs for user=%s -> %s", user_id, new_prefs)
    except Exception as e:
        logger.error("Failed saving prefs for user=%s (%s)", user_id, e)

    return {}


# ────────────────────────────────────────────────
# Build graph + checkpointer
# ────────────────────────────────────────────────
memory = MemorySaver()

workflow = StateGraph(state_schema=ChatState)
workflow.add_node("load_prefs", load_prefs)
workflow.add_node("chatbot", chatbot)
workflow.add_node("save_prefs", extract_and_save_prefs)

workflow.add_edge(START, "load_prefs")
workflow.add_edge("load_prefs", "chatbot")
workflow.add_edge("chatbot", "save_prefs")
workflow.add_edge("save_prefs", END)

graph = workflow.compile(checkpointer=memory)


# ────────────────────────────────────────────────
# Runner
# ────────────────────────────────────────────────
def run_turn(thread_id: str, user_id: str, user_message: str, label: str = "User") -> None:
    config = {"configurable": {"thread_id": thread_id}}
    inputs = {"messages": [("human", user_message)], "user_id": user_id}

    print(f"\n{label} ({thread_id}): {user_message}")
    print("─" * 60)

    for event in graph.stream(inputs, config, stream_mode="values"):
        msgs = event.get("messages") or []
        if not msgs:
            continue
        last = msgs[-1]
        if getattr(last, "type", None) == "ai":
            print("AI:", (last.content or "").strip())
            print("─" * 60)


if __name__ == "__main__":
    user = "PawanGunjan"

    print("=== First conversation ===")
    run_turn("t1", user, "Hi! My name is Pawan and I'm from Delhi.", "Msg 1")

    print("\n=== Same thread — second message ===")
    run_turn("t1", user, "What do you remember about me?", "Msg 2")

    print("\n=== New thread — same user → should load from disk ===")
    run_turn("t2", user, "Hey, remind me — where am I from?", "Msg 1 (new thread)")

    print(f"\nCheck folder '{PREFS_DIR}/' for {user}.json")
