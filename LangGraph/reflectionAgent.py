"""
ReflectAgent: Self-Improving LinkedIn Post Generator (Production Clean)
- Separate Human/AI/Critique storage
- No emojis
- Strict MAX_CHARS enforcement
- Detailed logs: console + JSONL timeline per run
"""

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

from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver


# =========================
# CONFIG
# =========================
load_dotenv()


class Config:
    DEBUG: bool = True

    MAX_ITERATIONS: int = 5
    MAX_CHARS: int = 160  # strict

    MODEL_NAME: str = os.getenv("OLLAMA_MODEL", "granite4:350m")
    TEMPERATURE: float = 0.0

    RUNS_DIR: str = "runs"          # each run => runs/<session_id>/
    GRAPH_DIR: str = "Graphs"       # optional graph PNGs

    # Logging
    LOG_LEVEL: int = logging.DEBUG if DEBUG else logging.INFO


# =========================
# UTIL: IDs, time, safe dumps
# =========================
def now_ms() -> int:
    return int(time.time() * 1000)


def new_id(n: int = 10) -> str:
    return uuid.uuid4().hex[:n]


def safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def dump_message(m: BaseMessage) -> Dict[str, Any]:
    return {
        "type": getattr(m, "type", m.__class__.__name__),
        "content": getattr(m, "content", ""),
    }


def extract_hashtags(text: str) -> List[str]:
    return re.findall(r"#\w+", text)


def enforce_max_chars(post: str, max_chars: int) -> Dict[str, Any]:
    """
    Ensures final post length <= max_chars, tries to keep hashtags at the end.
    Returns dict with: post, truncated(bool), hashtags(list), post_chars(int)
    """
    post = post.strip()
    hashtags = extract_hashtags(post)

    if len(post) <= max_chars:
        return {
            "post": post,
            "truncated": False,
            "hashtags": hashtags,
            "post_chars": len(post),
        }

    # Remove hashtags from body first (keep unique order)
    # (Simple approach: keep detected hashtags at the end.)
    uniq = []
    for h in hashtags:
        if h not in uniq:
            uniq.append(h)
    hashtags = uniq

    hashtag_blob = " ".join(hashtags).strip()
    # Space between body and hashtags if hashtags exist
    joiner = " " if hashtag_blob else ""

    allowed_body = max_chars - len(joiner) - len(hashtag_blob)
    if allowed_body < 0:
        # Hashtags alone exceed max_chars; trim hashtags to fit.
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
        return {
            "post": final_post,
            "truncated": True,
            "hashtags": kept,
            "post_chars": len(final_post),
        }

    # Body = everything before first hashtag occurrence, fallback to full post
    # This avoids duplicating hashtags mid-body after truncation.
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
# Structured JSONL run logger
# =========================
class RunLogger:
    def __init__(self, run_dir: Path, console_logger: logging.Logger):
        self.run_dir = run_dir
        self.console = console_logger
        self.jsonl_path = run_dir / "timeline.jsonl"

    def event(self, name: str, data: Optional[Dict[str, Any]] = None) -> None:
        rec = {
            "ts": datetime.now().isoformat(),
            "event": name,
            "data": data or {},
        }
        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        # concise console line
        self.console.info(f"{name} | " + json.dumps(rec["data"], ensure_ascii=False)[:500])


def setup_console_logger(name: str, level: int) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setLevel(level)
        fmt = logging.Formatter("[%(levelname)s] %(asctime)s - %(name)s - %(message)s")
        ch.setFormatter(fmt)
        logger.addHandler(ch)

    return logger


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
    iterations_data: List[IterationData]


def state_summary(state: AgentState) -> Dict[str, Any]:
    return {
        "iteration": state.get("iteration", 0),
        "messages_len": len(state.get("messages", [])),
        "trace_id": state.get("metadata", {}).get("trace_id"),
        "session_id": state.get("metadata", {}).get("session_id"),
    }


def last_ai_critique_text(state: AgentState) -> str:
    # critique messages are AIMessage starting with "CRITIQUE"
    for m in reversed(state.get("messages", [])):
        if isinstance(m, AIMessage) and m.content.strip().startswith("CRITIQUE"):
            return m.content.strip()
    return ""


def original_prompt(state: AgentState) -> str:
    return state.get("metadata", {}).get("original_prompt", "")


# =========================
# LLM + PROMPTS (no emojis + strict length)
# =========================
llm = ChatOllama(model=Config.MODEL_NAME, temperature=Config.TEMPERATURE)

generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "LinkedIn content specialist. No emojis. Max {max_chars} characters.\n"
            "Structure: Hook + Value + CTA + 2-4 hashtags.\n"
            "Use critique feedback to improve.\n"
            "Return only the post text.",
        ),
        ("human", "{user_request}"),
    ]
)

reflection_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Professional critique for LinkedIn engagement.\n"
            "Format exactly:\n"
            "STRENGTHS: - ... - ...\n"
            "WEAKNESSES: - ... - ...\n"
            "IMPROVEMENTS: 1) ... 2) ... 3) ...\n"
            "Max 300 chars. No emojis.",
        ),
        ("human", "{post_to_critique}"),
    ]
)

generate_chain = generation_prompt | llm
reflect_chain = reflection_prompt | llm


# =========================
# NODES
# =========================
def generation_node(state: AgentState) -> AgentState:
    t0 = now_ms()
    runlog: RunLogger = state["metadata"]["runlog"]  # injected object (not serialized)
    runlog.event("node_start.generate", {"state": state_summary(state)})

    iteration = state.get("iteration", 0) + 1

    base_request = original_prompt(state)
    critique = last_ai_critique_text(state)

    # Feed critique back into generation (this is the "reflect" loop)
    if critique:
        user_request = f"{base_request}\n\nRefine using this critique:\n{critique}"
    else:
        user_request = base_request

    runlog.event(
        "model_call.generate",
        {
            "iteration": iteration,
            "max_chars": Config.MAX_CHARS,
            "request_preview": user_request[:200],
        },
    )

    result = generate_chain.invoke({"user_request": user_request, "max_chars": Config.MAX_CHARS})
    raw_post = result.content.strip()

    enforced = enforce_max_chars(raw_post, Config.MAX_CHARS)
    post = enforced["post"]

    iter_data = IterationData(
        iteration=iteration,
        human_prompt=base_request,
        ai_post=post,
        critique="",  # filled in reflection_node
        post_chars=enforced["post_chars"],
        hashtags=enforced["hashtags"],
        truncated=enforced["truncated"],
        timestamp=datetime.now().isoformat(),
    )

    out: AgentState = {
        "messages": [AIMessage(content=post)],
        "metadata": state["metadata"],
        "iteration": iteration,
        "iterations_data": state.get("iterations_data", []) + [iter_data],
    }

    runlog.event(
        "node_end.generate",
        {
            "duration_ms": now_ms() - t0,
            "iteration": iteration,
            "post_chars": enforced["post_chars"],
            "hashtags": enforced["hashtags"],
            "truncated": enforced["truncated"],
        },
    )
    return out


def reflection_node(state: AgentState) -> AgentState:
    t0 = now_ms()
    runlog: RunLogger = state["metadata"]["runlog"]
    runlog.event("node_start.reflect", {"state": state_summary(state)})

    iteration = state.get("iteration", 0)

    last_post = ""
    for m in reversed(state.get("messages", [])):
        if isinstance(m, AIMessage) and not m.content.strip().startswith("CRITIQUE"):
            last_post = m.content.strip()
            break

    runlog.event(
        "model_call.reflect",
        {"iteration": iteration, "post_chars": len(last_post), "post_preview": last_post[:200]},
    )

    critique_result = reflect_chain.invoke({"post_to_critique": last_post})
    critique_text = critique_result.content.strip()

    critique_msg = AIMessage(content=f"CRITIQUE {iteration}: {critique_text}")

    # Backfill iteration data critique
    iterations_data = list(state.get("iterations_data", []))
    if iterations_data:
        iterations_data[-1].critique = critique_text

    out: AgentState = {
        "messages": [critique_msg],
        "metadata": state["metadata"],
        "iteration": iteration,
        "iterations_data": iterations_data,
    }

    runlog.event(
        "node_end.reflect",
        {"duration_ms": now_ms() - t0, "iteration": iteration, "critique_preview": critique_text[:200]},
    )
    return out


def should_continue(state: AgentState) -> Literal["generate", END]:
    runlog: RunLogger = state["metadata"]["runlog"]
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

    workflow = graph.compile(checkpointer=MemorySaver())
    return workflow


# =========================
# FILE SAVING (run folder)
# =========================
def export_run_artifacts(state: AgentState, run_dir: Path) -> Dict[str, str]:
    safe_mkdir(run_dir)

    # 1) Full state JSON (messages + metadata without runlog)
    meta = dict(state.get("metadata", {}))
    meta.pop("runlog", None)

    full_state = {
        "iteration": state.get("iteration", 0),
        "metadata": meta,
        "messages": [dump_message(m) for m in state.get("messages", [])],
        "iterations_data": [it.to_dict() for it in state.get("iterations_data", [])],
    }
    full_path = run_dir / "full_state.json"
    full_path.write_text(json.dumps(full_state, indent=2, ensure_ascii=False), encoding="utf-8")

    # 2) Iterations JSON
    iters_path = run_dir / "iterations.json"
    iters_path.write_text(
        json.dumps([it.to_dict() for it in state.get("iterations_data", [])], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # 3) Separate files
    human_path = run_dir / "human_prompt.txt"
    human_path.write_text(meta.get("original_prompt", ""), encoding="utf-8")

    posts_dir = run_dir / "posts"
    critiques_dir = run_dir / "critiques"
    safe_mkdir(posts_dir)
    safe_mkdir(critiques_dir)

    for it in state.get("iterations_data", []):
        (posts_dir / f"post_{it.iteration:02d}.txt").write_text(it.ai_post, encoding="utf-8")
        (critiques_dir / f"critique_{it.iteration:02d}.txt").write_text(it.critique, encoding="utf-8")

    # 4) Best post (last valid <= MAX_CHARS, fallback last)
    iterations = state.get("iterations_data", [])
    valid = [it for it in iterations if it.post_chars <= Config.MAX_CHARS]
    best = valid[-1].ai_post if valid else (iterations[-1].ai_post if iterations else "")
    best_path = run_dir / "best_post.txt"
    best_path.write_text(best, encoding="utf-8")

    return {
        "full_state": str(full_path),
        "iterations": str(iters_path),
        "best_post": str(best_path),
        "run_dir": str(run_dir),
    }


def try_save_graph_png(workflow: Any, out_path: Path, console: logging.Logger) -> None:
    try:
        safe_mkdir(out_path.parent)
        png_data = workflow.get_graph().draw_mermaid_png()
        out_path.write_bytes(png_data)
        console.info(f"Graph PNG saved: {out_path}")
    except Exception as e:
        console.warning(f"Graph render failed: {e}")


# =========================
# MAIN RUNNER
# =========================
def run_reflect_agent(user_prompt: str, session_id: Optional[str] = None) -> Dict[str, Any]:
    session_id = session_id or f"reflect_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{new_id(6)}"
    trace_id = new_id(10)

    run_dir = Path(Config.RUNS_DIR) / session_id
    safe_mkdir(run_dir)

    console = setup_console_logger("ReflectAgent", Config.LOG_LEVEL)
    runlog = RunLogger(run_dir=run_dir, console_logger=console)

    workflow = create_workflow()

    # optional graph png
    graph_dir = Path(Config.GRAPH_DIR)
    safe_mkdir(graph_dir)
    try_save_graph_png(workflow, graph_dir / f"reflection_agent_{session_id}.png", console)

    # IMPORTANT: runlog is injected into metadata for node access (not serialized in exports)
    inputs: AgentState = {
        "messages": [HumanMessage(content=user_prompt)],
        "metadata": {
            "original_prompt": user_prompt,
            "max_chars": Config.MAX_CHARS,
            "session_id": session_id,
            "trace_id": trace_id,
            "runlog": runlog,
        },
        "iteration": 0,
        "iterations_data": [],
    }

    # checkpointer uses thread_id to separate sessions
    config = {"configurable": {"thread_id": session_id}}

    runlog.event("run_start", {"session_id": session_id, "trace_id": trace_id, "max_chars": Config.MAX_CHARS})
    final_state = workflow.invoke(inputs, config)
    runlog.event("run_end", {"iteration": final_state.get("iteration", 0), "messages_len": len(final_state.get("messages", []))})

    # export (remove non-serializable runlog from metadata)
    final_state["metadata"] = dict(final_state.get("metadata", {}))
    final_state["metadata"].pop("runlog", None)

    files = export_run_artifacts(final_state, run_dir)
    runlog.event("files_saved", files)

    return {
        "session_id": session_id,
        "trace_id": trace_id,
        "files": files,
        "state": final_state,
    }


if __name__ == "__main__":
    result = run_reflect_agent("Write LinkedIn post: IBM software developer job under 160 characters")
    print("\nFiles generated:")
    for k, v in result["files"].items():
        print(f"  {k}: {v}")
