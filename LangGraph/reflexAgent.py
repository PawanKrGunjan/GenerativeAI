from __future__ import annotations

import os
import json
import time
import logging
import operator
from pathlib import Path
from typing import Annotated, TypedDict, List, Literal, Dict, Any

import mlflow
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

from logger_config import setup_logger


# =============================================================================
# LOGGER (logger_config -> logs/<name>.log with rotation)
# =============================================================================
logger = setup_logger(
    debug_mode=True,          # set False for less verbosity
    log_name="reflex-agent",
    log_dir="logs",
)
logger.info("Logger configured", extra={"status": "INFO"})


# =============================================================================
# MLFLOW
# =============================================================================
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT", "reflex-agent"))


# =============================================================================
# CONFIG
# =============================================================================
class Config:
    MAX_STEPS = int(os.getenv("MAX_STEPS", "10"))

    # This is for run artifacts/traces (JSONL, outputs)
    RUNS_DIR = Path(os.getenv("RUNS_DIR", "runs_reflex"))

    SAVE_TRACE = os.getenv("SAVE_TRACE", "1") == "1"


# =============================================================================
# STATE
# =============================================================================
class ReflexState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    step: int
    max_steps: int
    env: Dict[str, Any]
    percepts: Annotated[List[Dict[str, Any]], operator.add]
    actions: Annotated[List[Dict[str, Any]], operator.add]
    done: bool
    last_node: str
    run_dir: str  # per-run artifacts directory (runs_reflex/<run_id>/)


def _append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def log_event(state: ReflexState, event: str, data: Dict[str, Any] | None = None) -> None:
    if not Config.SAVE_TRACE:
        return
    run_dir = Path(state["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    _append_jsonl(run_dir / "timeline.jsonl", {"ts": time.time(), "event": event, "data": data or {}})


# =============================================================================
# ENV: Vacuum world
# =============================================================================
def env_is_clean(env: Dict[str, Any]) -> bool:
    dirt = env.get("dirt", {})
    return (not dirt.get("A", False)) and (not dirt.get("B", False))


def sense(env: Dict[str, Any]) -> Dict[str, Any]:
    loc = env["loc"]
    dirty_here = bool(env["dirt"].get(loc, False))
    return {"loc": loc, "dirty_here": dirty_here}


def reflex_policy(percept: Dict[str, Any]) -> str:
    if percept["dirty_here"]:
        return "SUCK"
    return "MOVE_RIGHT" if percept["loc"] == "A" else "MOVE_LEFT"


def step_env(env: Dict[str, Any], action: str) -> Dict[str, Any]:
    env = json.loads(json.dumps(env))  # cheap deep-copy

    if action == "SUCK":
        env["dirt"][env["loc"]] = False
        return env
    if action == "MOVE_RIGHT":
        env["loc"] = "B"
        return env
    if action == "MOVE_LEFT":
        env["loc"] = "A"
        return env

    return env


# =============================================================================
# NODES
# =============================================================================
def perceive_node(state: ReflexState) -> Dict[str, Any]:
    env = state["env"]
    percept = sense(env)

    logger.info("PERCEIVE | step=%s | percept=%s", state["step"], percept)
    log_event(state, "perceive", {"step": state["step"], "percept": percept, "env": env})

    return {
        "messages": [AIMessage(content=f"PERCEPT(step={state['step']}): {percept}")],
        "percepts": [percept],
        "last_node": "perceive",
    }


def act_node(state: ReflexState) -> Dict[str, Any]:
    step = state["step"]
    env = state["env"]

    percept = state["percepts"][-1] if state.get("percepts") else sense(env)
    action = reflex_policy(percept)
    new_env = step_env(env, action)

    done = env_is_clean(new_env) or (step + 1 >= state["max_steps"])

    logger.info("ACT | step=%s | action=%s | done=%s | env_before=%s | env_after=%s", step, action, done, env, new_env)
    log_event(state, "act", {"step": step, "action": action, "env_before": env, "env_after": new_env})

    return {
        "messages": [AIMessage(content=f"ACTION(step={step}): {action} | done={done}")],
        "actions": [{"step": step, "action": action}],
        "env": new_env,
        "step": step + 1,
        "done": done,
        "last_node": "act",
    }


def finalize_node(state: ReflexState) -> Dict[str, Any]:
    env = state["env"]
    clean = env_is_clean(env)

    summary = {
        "steps": state.get("step", 0),
        "clean": clean,
        "final_env": env,
        "best_case": "cleaned both rooms" if clean else "stopped by max_steps",
    }

    logger.info("FINAL | %s", summary)
    log_event(state, "finalize", summary)

    return {
        "messages": [AIMessage(content=f"FINAL: {json.dumps(summary)}")],
        "last_node": "finalize",
    }


# =============================================================================
# ROUTER
# =============================================================================
def route(state: ReflexState) -> Literal["perceive", "act", "finalize", "__end__"]:
    if state.get("done", False):
        return "finalize"

    last = state.get("last_node", "")
    if last == "perceive":
        return "act"
    if last == "act":
        return "perceive"
    return "perceive"


# =============================================================================
# GRAPH
# =============================================================================
def create_reflex_agent():
    builder = StateGraph(ReflexState)
    builder.add_node("perceive", perceive_node)
    builder.add_node("act", act_node)
    builder.add_node("finalize", finalize_node)

    builder.set_entry_point("perceive")

    builder.add_conditional_edges(
        "perceive",
        route,
        {"act": "act", "finalize": "finalize", "perceive": "perceive"},
    )
    builder.add_conditional_edges(
        "act",
        route,
        {"perceive": "perceive", "finalize": "finalize", "act": "act"},
    )

    builder.add_edge("finalize", END)
    return builder.compile()


# =============================================================================
# RUN
# =============================================================================
if __name__ == "__main__":
    initial_env = {"loc": "A", "dirt": {"A": True, "B": True}}

    run_id = f"reflex_{int(time.time())}"
    run_dir = Config.RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    logger.info("RUN_START | run_id=%s | run_dir=%s", run_id, run_dir)

    app = create_reflex_agent()

    init_state: ReflexState = {
        "messages": [HumanMessage(content="Run reflex vacuum agent")],
        "step": 0,
        "max_steps": Config.MAX_STEPS,
        "env": initial_env,
        "percepts": [],
        "actions": [],
        "done": False,
        "last_node": "",
        "run_dir": str(run_dir),
    }

    with mlflow.start_run(run_name=run_id):
        mlflow.log_params({"max_steps": Config.MAX_STEPS, "env_init": json.dumps(initial_env)})

        result = app.invoke(init_state, {"recursion_limit": 100})

        mlflow.log_metric("steps", int(result.get("step", 0)))
        mlflow.log_metric("clean", 1.0 if env_is_clean(result["env"]) else 0.0)

        timeline = Path(result["run_dir"]) / "timeline.jsonl"
        if timeline.exists():
            mlflow.log_artifact(str(timeline), artifact_path="logs")

        actions_path = Path(result["run_dir"]) / "actions.json"
        actions_path.write_text(json.dumps(result.get("actions", []), indent=2), encoding="utf-8")
        mlflow.log_artifact(str(actions_path), artifact_path="outputs")

    logger.info("RUN_END | run_id=%s | steps=%s | clean=%s", run_id, result.get("step", 0), env_is_clean(result["env"]))

    print("\n" + "=" * 80)
    print("MESSAGES:")
    for m in result["messages"]:
        if isinstance(m, (HumanMessage, AIMessage)):
            print("-", m.content)

    print("\nFINAL ENV:", result["env"])
    print("ACTIONS:", result.get("actions", []))
    print("RUN DIR:", result["run_dir"])
