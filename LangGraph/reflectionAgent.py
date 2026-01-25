"""
ReflectAgent: Self-Improving LinkedIn Post Generator (Production Clean)
- Uses logger_config.setup_logger for file + console logs
- Saves timeline JSONL under logs/<session_id>/timeline.jsonl
- Keeps state serializable for LangGraph checkpointers
"""

from __future__ import annotations

import os
import re
import json
import uuid
import time
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, TypedDict, Literal, Annotated, Optional

import mlflow
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

from logger_config import setup_logger


# =========================
# CONFIG
# =========================
load_dotenv()

class Config:
    DEBUG: bool = True

    MAX_ITERATIONS: int = 5
    MAX_CHARS: int = 160

    MODEL_NAME: str = os.getenv("OLLAMA_MODEL", "granite4:350m")
    TEMPERATURE: float = 0.0

    RUNS_DIR: str = "runs"
    GRAPH_DIR: str = "Graphs"
    LOG_DIR: str = "logs"  # <--- NEW: all logs go here

    LOG_LEVEL: int = logging.DEBUG if DEBUG else logging.INFO


# =========================
# LOGGER (logger_config)
# =========================
logger = setup_logger(
    debug_mode=Config.DEBUG,
    log_name="reflection-agent",   # logs/reflection-agent.log (rotated daily)
    log_dir=Config.LOG_DIR
)
logger.info("Logger configured", extra={"status": "INFO"})


# =========================
# MLFLOW
# =========================
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("reflection-agent")
mlflow.langchain.autolog()


# =========================
# UTIL
# =========================
def now_ms() -> int:
    return int(time.time() * 1000)

def new_id(n: int = 10) -> str:
    return uuid.uuid4().hex[:n]

def safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

def dump_message(m: BaseMessage) -> Dict[str, Any]:
    return {"type": getattr(m, "type", m.__class__.__name__), "content": getattr(m, "content", "")}

def extract_hashtags(text: str) -> List[str]:
    return re.findall(r"#\w+", text)

def enforce_max_chars(post: str, max_chars: int) -> Dict[str, Any]:
    post = post.strip()
    hashtags = extract_hashtags(post)

    if len(post) <= max_chars:
        return {"post": post, "truncated": False, "hashtags": hashtags, "post_chars": len(post)}

    uniq = []
    for h in hashtags:
        if h not in uniq:
            uniq.append(h)
    hashtags = uniq

    hashtag_blob = " ".join(hashtags).strip()
    joiner = " " if hashtag_blob else ""

    allowed_body = max_chars - len(joiner) - len(hashtag_blob)
    if allowed_body < 0:
        trimmed = ""
        kept = []
        for h in hashtags:
            candidate = (trimmed + (" " if trimmed else "") + h).strip()
            if len(candidate) <= max_chars:
                trimmed = candidate
                kept.append(h)
            else:
                break
        final_post = trimmed[:max_chars]
        return {"post": final_post, "truncated": True, "hashtags": kept, "post_chars": len(final_post)}

    first_hash = post.find("#")
    body = post if first_hash == -1 else post[:first_hash].rstrip()
    body = body[:allowed_body].rstrip()

    final_post = (body + joiner + hashtag_blob).strip()
    final_post = final_post[:max_chars].rstrip()

    return {
        "post": final_post,
        "truncated": True,
        "hashtags": extract_hashtags(final_post),
        "post_chars": len(final_post),
    }


# =========================
# RunLogger (JSONL timeline) -> saved under logs/<session_id>/
# =========================
class RunLogger:
    def __init__(self, log_run_dir: Path, app_logger: logging.Logger):
        self.log_run_dir = log_run_dir
        self.logger = app_logger
        self.jsonl_path = log_run_dir / "timeline.jsonl"

    def event(self, name: str, data: Optional[Dict[str, Any]] = None) -> None:
        safe_mkdir(self.log_run_dir)
        rec = {"ts": datetime.now().isoformat(), "event": name, "data": data or {}}
        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        # Also goes to logs/reflection-agent.log
        self.logger.info(f"{name} | {json.dumps(rec['data'], ensure_ascii=False)[:500]}")


def get_runlog_from_state(state: "AgentState") -> RunLogger:
    meta = state.get("metadata", {})
    log_run_dir = Path(meta["log_run_dir"])   # <--- logs/<session_id>
    return RunLogger(log_run_dir=log_run_dir, app_logger=logger)


# =========================
# Iteration storage
# =========================
@dataclass
class IterationData:
    iteration: int
    human_prompt: str
    ai_post: str
    critique: str
    post_chars: int
    hashtags: List[str]
    truncated: bool
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# =========================
# STATE
# =========================
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    metadata: Dict[str, Any]
    iteration: int
    iterations_data: List[Dict[str, Any]]


def state_summary(state: AgentState) -> Dict[str, Any]:
    meta = state.get("metadata", {})
    return {
        "iteration": state.get("iteration", 0),
        "messages_len": len(state.get("messages", [])),
        "trace_id": meta.get("trace_id"),
        "session_id": meta.get("session_id"),
    }

def last_ai_critique_text(state: AgentState) -> str:
    for m in reversed(state.get("messages", [])):
        if isinstance(m, AIMessage) and m.content.strip().startswith("CRITIQUE"):
            return m.content.strip()
    return ""

def original_prompt(state: AgentState) -> str:
    return state.get("metadata", {}).get("original_prompt", "")


# =========================
# LLM + PROMPTS
# =========================
llm = ChatOllama(model=Config.MODEL_NAME, temperature=Config.TEMPERATURE)

generation_prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
         "LinkedIn content specialist. No emojis. Max {max_chars} characters.\n"
         "Structure: Hook + Value + CTA + 2-4 hashtags.\n"
         "Use critique feedback to improve.\n"
         "Return only the post text."),
        ("human", "{user_request}"),
    ]
)

reflection_prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
         "Professional critique for LinkedIn engagement.\n"
         "Format exactly:\n"
         "STRENGTHS: - ... - ...\n"
         "WEAKNESSES: - ... - ...\n"
         "IMPROVEMENTS: 1) ... 2) ... 3) ...\n"
         "Max 300 chars. No emojis."),
        ("human", "{post_to_critique}"),
    ]
)

generate_chain = generation_prompt | llm
reflect_chain = reflection_prompt | llm


# =========================
# NODES
# =========================
def generation_node(state: AgentState) -> Dict[str, Any]:
    t0 = now_ms()
    runlog = get_runlog_from_state(state)
    runlog.event("node_start.generate", {"state": state_summary(state)})

    iteration = state.get("iteration", 0) + 1
    base_request = original_prompt(state)
    critique = last_ai_critique_text(state)
    user_request = f"{base_request}\n\nRefine using this critique:\n{critique}" if critique else base_request

    runlog.event("model_call.generate", {"iteration": iteration, "max_chars": Config.MAX_CHARS})
    result = generate_chain.invoke({"user_request": user_request, "max_chars": Config.MAX_CHARS})
    raw_post = result.content.strip()

    enforced = enforce_max_chars(raw_post, Config.MAX_CHARS)
    post = enforced["post"]

    iter_data = IterationData(
        iteration=iteration,
        human_prompt=base_request,
        ai_post=post,
        critique="",
        post_chars=enforced["post_chars"],
        hashtags=enforced["hashtags"],
        truncated=enforced["truncated"],
        timestamp=datetime.now().isoformat(),
    ).to_dict()

    new_iters = list(state.get("iterations_data", [])) + [iter_data]

    runlog.event("node_end.generate", {"duration_ms": now_ms() - t0, "iteration": iteration, "post_chars": enforced["post_chars"]})
    return {"messages": [AIMessage(content=post)], "iteration": iteration, "iterations_data": new_iters}


def reflection_node(state: AgentState) -> Dict[str, Any]:
    t0 = now_ms()
    runlog = get_runlog_from_state(state)
    runlog.event("node_start.reflect", {"state": state_summary(state)})

    iteration = state.get("iteration", 0)

    last_post = ""
    for m in reversed(state.get("messages", [])):
        if isinstance(m, AIMessage) and not m.content.strip().startswith("CRITIQUE"):
            last_post = m.content.strip()
            break

    runlog.event("model_call.reflect", {"iteration": iteration, "post_chars": len(last_post)})
    critique_result = reflect_chain.invoke({"post_to_critique": last_post})
    critique_text = critique_result.content.strip()

    iters = list(state.get("iterations_data", []))
    if iters:
        iters[-1] = dict(iters[-1])
        iters[-1]["critique"] = critique_text

    runlog.event("node_end.reflect", {"duration_ms": now_ms() - t0, "iteration": iteration})
    return {"messages": [AIMessage(content=f"CRITIQUE {iteration}: {critique_text}")], "iterations_data": iters}


def should_continue(state: AgentState) -> Literal["generate", END]:
    runlog = get_runlog_from_state(state)
    iteration = state.get("iteration", 0)
    decision: Literal["generate", END] = END if iteration >= Config.MAX_ITERATIONS else "generate"
    runlog.event("route_decision", {"iteration": iteration, "max": Config.MAX_ITERATIONS, "decision": str(decision)})
    return decision


# =========================
# WORKFLOW
# =========================
def create_workflow() -> Any:
    graph = StateGraph(AgentState)
    graph.add_node("generate", generation_node)
    graph.add_node("reflect", reflection_node)

    graph.set_entry_point("generate")
    graph.add_edge("generate", "reflect")
    graph.add_conditional_edges("reflect", should_continue, {"generate": "generate", END: END})

    return graph.compile(checkpointer=MemorySaver())


# =========================
# EXPORTS
# =========================
def export_run_artifacts(state: AgentState, run_dir: Path) -> Dict[str, str]:
    safe_mkdir(run_dir)
    meta = dict(state.get("metadata", {}))

    full_state = {
        "iteration": state.get("iteration", 0),
        "metadata": meta,
        "messages": [dump_message(m) for m in state.get("messages", [])],
        "iterations_data": state.get("iterations_data", []),
    }
    full_path = run_dir / "full_state.json"
    full_path.write_text(json.dumps(full_state, indent=2, ensure_ascii=False), encoding="utf-8")

    iters_path = run_dir / "iterations.json"
    iters_path.write_text(json.dumps(state.get("iterations_data", []), indent=2, ensure_ascii=False), encoding="utf-8")

    best_path = run_dir / "best_post.txt"
    iterations = state.get("iterations_data", [])
    valid = [it for it in iterations if int(it.get("post_chars", 10**9)) <= Config.MAX_CHARS]
    best = valid[-1].get("ai_post", "") if valid else (iterations[-1].get("ai_post", "") if iterations else "")
    best_path.write_text(best, encoding="utf-8")

    return {"full_state": str(full_path), "iterations": str(iters_path), "best_post": str(best_path), "run_dir": str(run_dir)}


# =========================
# MAIN RUNNER
# =========================
def run_reflect_agent(user_prompt: str, session_id: Optional[str] = None) -> Dict[str, Any]:
    session_id = session_id or f"reflect_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{new_id(6)}"
    trace_id = new_id(10)

    run_dir = Path(Config.RUNS_DIR) / session_id
    safe_mkdir(run_dir)

    # per-run log folder for timeline.jsonl
    log_run_dir = Path(Config.LOG_DIR) / session_id
    safe_mkdir(log_run_dir)

    workflow = create_workflow()

    inputs: AgentState = {
        "messages": [HumanMessage(content=user_prompt)],
        "metadata": {
            "original_prompt": user_prompt,
            "max_chars": Config.MAX_CHARS,
            "session_id": session_id,
            "trace_id": trace_id,
            "run_dir": str(run_dir),            # outputs/artifacts
            "log_run_dir": str(log_run_dir),    # logs/timeline.jsonl
        },
        "iteration": 0,
        "iterations_data": [],
    }

    config = {"configurable": {"thread_id": session_id}}

    runlog = RunLogger(log_run_dir=log_run_dir, app_logger=logger)
    runlog.event("run_start", {"session_id": session_id, "trace_id": trace_id, "max_chars": Config.MAX_CHARS})

    graphs_dir = Path(Config.GRAPH_DIR)
    safe_mkdir(graphs_dir)
    try:
        png_data = workflow.get_graph().draw_mermaid_png()
        (graphs_dir / f"reflection_agent_{session_id}.png").write_bytes(png_data)
    except Exception as e:
        logger.warning(f"Graph render failed: {e}", extra={"status": "WARN"})

    with mlflow.start_run(run_name=session_id):
        final_state = workflow.invoke(inputs, config)
        runlog.event("run_end", {"iteration": final_state.get("iteration", 0), "messages_len": len(final_state.get("messages", []))})

        files = export_run_artifacts(final_state, run_dir)
        runlog.event("files_saved", files)

        # MLflow artifacts
        mlflow.log_text(user_prompt, "input/prompt.txt")
        for k, v in files.items():
            if k != "run_dir":
                mlflow.log_artifact(v, artifact_path="outputs")

        # log timeline
        timeline_path = log_run_dir / "timeline.jsonl"
        if timeline_path.exists():
            mlflow.log_artifact(str(timeline_path), artifact_path="logs")

        # log the rotating app log file (current)
        app_log = Path(Config.LOG_DIR) / "reflection-agent.log"
        if app_log.exists():
            mlflow.log_artifact(str(app_log), artifact_path="logs")

    return {"session_id": session_id, "trace_id": trace_id, "files": files, "state": final_state}


if __name__ == "__main__":
    result = run_reflect_agent("Write LinkedIn post: I am casually looking for Agentic AI Role in Noida")
    print("\nFiles generated:")
    for k, v in result["files"].items():
        print(f"  {k}: {v}")
