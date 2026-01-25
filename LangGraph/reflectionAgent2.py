"""
Generic Reflection Agent (AI Researcher style)
- LLM generates drafts
- LLM critiques in strict JSON with a numeric score
- Loops until score threshold or max iterations
- State is checkpointer-safe (only serializable data)
- MLflow tracing via mlflow.langchain.autolog()

Logging:
- Uses logger_config.setup_logger -> logs/reflection-agent-generic.log (+ rotated)
- Also writes per-run JSONL timeline to runs_generic_reflect/<session_id>/timeline.jsonl
"""

from __future__ import annotations

import os
import json
import uuid
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, TypedDict, Literal, Annotated
import operator

import mlflow
from dotenv import load_dotenv

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from logger_config import setup_logger


# =========================
# Config
# =========================
load_dotenv()

class Config:
    TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT", "reflection-agent")

    MODEL_DRAFT = os.getenv("OLLAMA_MODEL_DRAFT", "granite4:350m")
    MODEL_CRITIC = os.getenv("OLLAMA_MODEL_CRITIC", "llama3.2:1b")

    TEMPERATURE_DRAFT = float(os.getenv("TEMP_DRAFT", "0.2"))
    TEMPERATURE_CRITIC = float(os.getenv("TEMP_CRITIC", "0.0"))

    MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", "5"))
    MIN_SCORE = float(os.getenv("MIN_SCORE", "8.0"))

    RUNS_DIR = Path(os.getenv("RUNS_DIR", "runs_generic_reflect"))

    LOG_DIR = os.getenv("LOG_DIR", "logs")
    DEBUG = os.getenv("DEBUG", "1") == "1"


# =========================
# Logger (logger_config)
# =========================
logger = setup_logger(
    debug_mode=Config.DEBUG,
    log_name="reflection-agent-generic",
    log_dir=Config.LOG_DIR,
)
logger.info("Logger configured", extra={"status": "INFO"})


# =========================
# MLflow (autolog traces)
# =========================
mlflow.set_tracking_uri(Config.TRACKING_URI)
mlflow.set_experiment(Config.EXPERIMENT)
mlflow.langchain.autolog()


# =========================
# Helpers
# =========================
def now_ms() -> int:
    return int(time.time() * 1000)

def new_id(n: int = 8) -> str:
    return uuid.uuid4().hex[:n]

def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def append_jsonl(path: Path, rec: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

def event(state: "ResearchState", name: str, data: Dict[str, Any] | None = None) -> None:
    """Write an event to JSONL timeline and also to the rotating log file."""
    run_dir = Path(state["meta"]["run_dir"])
    safe_mkdir(run_dir)
    timeline = run_dir / "timeline.jsonl"

    rec = {"ts": datetime.now().isoformat(), "event": name, "data": data or {}}
    append_jsonl(timeline, rec)

    # Also log to file/console (rotating handler)
    logger.info("%s | %s", name, json.dumps(rec["data"], ensure_ascii=False)[:800])

def parse_first_json_object(text: str) -> Dict[str, Any]:
    t = text.strip()
    if t.startswith("```"):
        t = t.removeprefix("```json").removeprefix("```").strip()
        if t.endswith("```"):
            t = t.removesuffix("```").strip()

    start = t.find("{")
    if start < 0:
        raise ValueError("No JSON object found in critic output")

    obj, _ = json.JSONDecoder().raw_decode(t[start:])
    if not isinstance(obj, dict):
        raise ValueError("Critic JSON is not an object")
    return obj


# =========================
# State
# =========================
class ResearchState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    iteration: int
    best_score: float
    drafts: Annotated[List[Dict[str, Any]], operator.add]
    meta: Dict[str, Any]


def get_last_draft_text(state: ResearchState) -> str:
    for m in reversed(state.get("messages", [])):
        if isinstance(m, AIMessage) and m.content.startswith("DRAFT"):
            return m.content
    return ""

def get_question(state: ResearchState) -> str:
    return state["meta"]["question"]


# =========================
# LLMs + Prompts
# =========================
draft_llm = ChatOllama(model=Config.MODEL_DRAFT, temperature=Config.TEMPERATURE_DRAFT)
critic_llm = ChatOllama(model=Config.MODEL_CRITIC, temperature=Config.TEMPERATURE_CRITIC, format="json")

draft_prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content=(
        "You are an AI researcher and technical writer.\n"
        "Write a crisp, correct explanation.\n"
        "Constraints:\n"
        "- No emojis.\n"
        "- Use clear structure: Overview -> Key components -> How it works -> Practical notes.\n"
        "- Prefer correct terminology and include 1 small concrete example if helpful.\n"
        "Return only the answer text."
    )),
    ("human", "Question: {question}\n\nImprove using critique (if any):\n{critique}\n"),
])

critic_prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content=(
        "You are a strict technical reviewer.\n"
        "Evaluate the draft for correctness, completeness, clarity, and structure.\n"
        "Return JSON ONLY with this schema:\n"
        "{\n"
        '  "score": <number 0..10>,\n'
        '  "critique": "<max 240 chars>",\n'
        '  "fixes": ["<short fix 1>", "<short fix 2>", "<short fix 3>"]\n'
        "}\n"
        "No markdown. No extra keys."
    )),
    ("human", "Question: {question}\n\nDraft:\n{draft}\n"),
])

draft_chain = draft_prompt | draft_llm
critic_chain = critic_prompt | critic_llm


# =========================
# Nodes
# =========================
def generate_node(state: ResearchState) -> Dict[str, Any]:
    t0 = now_ms()
    iteration = state.get("iteration", 0) + 1

    critique = ""
    for m in reversed(state.get("messages", [])):
        if isinstance(m, AIMessage) and m.content.startswith("CRITIQUE"):
            critique = m.content
            break

    event(state, "node_start.generate", {"iteration": iteration})
    res = draft_chain.invoke({"question": get_question(state), "critique": critique})

    draft_text = res.content.strip()
    draft_msg = AIMessage(content=f"DRAFT {iteration}:\n{draft_text}")

    event(state, "node_end.generate", {"iteration": iteration, "duration_ms": now_ms() - t0})
    return {"messages": [draft_msg], "iteration": iteration}


def critique_node(state: ResearchState) -> Dict[str, Any]:
    t0 = now_ms()
    iteration = state.get("iteration", 0)

    draft_msg = get_last_draft_text(state) or "DRAFT:\n<missing>"

    event(state, "node_start.critique", {"iteration": iteration})
    res = critic_chain.invoke({"question": get_question(state), "draft": draft_msg})

    score = 0.0
    critique = "Critic parse failed."
    fixes: List[str] = []

    try:
        obj = parse_first_json_object(res.content)
        score = float(obj.get("score", 0.0))
        critique = str(obj.get("critique", "")).strip()[:240]
        fixes_raw = obj.get("fixes", [])
        fixes = [str(x).strip() for x in fixes_raw][:3] if isinstance(fixes_raw, list) else []
    except Exception as e:
        critique = f"Critic parse failed: {e}"
        logger.exception("Critique parse failed")

    best_score = max(state.get("best_score", 0.0), score)

    critique_msg = AIMessage(
        content="CRITIQUE {it}: score={s:.1f}/10 | {c} | fixes: {f}".format(
            it=iteration, s=score, c=critique, f="; ".join(fixes) if fixes else "none"
        )
    )

    draft_record = {
        "iteration": iteration,
        "score": score,
        "best_score_after": best_score,
        "draft": draft_msg,  # string
        "critique": critique,
        "fixes": fixes,
        "ts": datetime.now().isoformat(),
    }

    event(state, "node_end.critique", {
        "iteration": iteration,
        "score": score,
        "best_score": best_score,
        "duration_ms": now_ms() - t0
    })

    return {"messages": [critique_msg], "best_score": best_score, "drafts": [draft_record]}


def finalize_node(state: ResearchState) -> Dict[str, Any]:
    drafts = state.get("drafts", [])
    best = max(drafts, key=lambda d: float(d.get("score", 0.0))) if drafts else None

    final_text = "FINAL:\n"
    if best and isinstance(best.get("draft"), str):
        final_text += best["draft"].split("\n", 1)[-1].strip()
    else:
        last_draft = get_last_draft_text(state)
        final_text += last_draft.split("\n", 1)[-1].strip() if last_draft else "No draft available."

    event(state, "finalize", {"best_score": state.get("best_score", 0.0)})
    return {"messages": [AIMessage(content=final_text)]}


# =========================
# Router
# =========================
def route(state: ResearchState) -> Literal["generate", "critique", "finalize", "__end__"]:
    it = state.get("iteration", 0)
    best = state.get("best_score", 0.0)

    if it >= Config.MAX_ITERATIONS or best >= Config.MIN_SCORE:
        return "finalize"

    last = state["messages"][-1].content if state.get("messages") else ""
    if last.startswith("DRAFT"):
        return "critique"
    if last.startswith("CRITIQUE"):
        return "generate"
    return "generate"


# =========================
# Build graph
# =========================
def create_app():
    g = StateGraph(ResearchState)
    g.add_node("generate", generate_node)
    g.add_node("critique", critique_node)
    g.add_node("finalize", finalize_node)

    g.set_entry_point("generate")
    g.add_conditional_edges("generate", route, {"critique": "critique", "finalize": "finalize", "generate": "generate"})
    g.add_conditional_edges("critique", route, {"generate": "generate", "finalize": "finalize", "critique": "critique"})
    g.add_edge("finalize", END)

    return g.compile(checkpointer=MemorySaver())


# =========================
# Run
# =========================
def run(question: str) -> Dict[str, Any]:
    session_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{new_id(6)}"
    run_dir = Config.RUNS_DIR / session_id
    safe_mkdir(run_dir)

    app = create_app()

    init: ResearchState = {
        "messages": [HumanMessage(content=question)],
        "iteration": 0,
        "best_score": 0.0,
        "drafts": [],
        "meta": {
            "question": question,
            "session_id": session_id,
            "run_dir": str(run_dir),
        },
    }

    config = {"configurable": {"thread_id": session_id}}

    logger.info("RUN_START | session_id=%s", session_id)
    with mlflow.start_run(run_name=session_id):
        mlflow.log_text(question, "input/question.txt")
        result = app.invoke(init, config)

        timeline = run_dir / "timeline.jsonl"
        if timeline.exists():
            mlflow.log_artifact(str(timeline), artifact_path="logs")

        drafts_path = run_dir / "drafts.json"
        drafts_path.write_text(json.dumps(result.get("drafts", []), indent=2, ensure_ascii=False), encoding="utf-8")
        mlflow.log_artifact(str(drafts_path), artifact_path="outputs")

        mlflow.log_metric("best_score", float(result.get("best_score", 0.0)))
        mlflow.log_metric("iterations", int(result.get("iteration", 0)))

    logger.info("RUN_END | session_id=%s | best_score=%.2f | iterations=%d",
                session_id, float(result.get("best_score", 0.0)), int(result.get("iteration", 0)))
    return {"session_id": session_id, "run_dir": str(run_dir), "result": result}


if __name__ == "__main__":
    out = run("Explain Transformers architecture.")
    final_msg = out["result"]["messages"][-1].content
    print("\n" + "=" * 80)
    print(final_msg)
    print("=" * 80)
    print(f"Session: {out['session_id']}")
    print(f"Run dir: {out['run_dir']}")
