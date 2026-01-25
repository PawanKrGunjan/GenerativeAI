"""
Tool Agent: Single Logger + Planner → Tool → Format → Reflect
Fixes:
- Planner uses JSON-mode LLM; Reflect uses normal LLM (so it can output YES/NO)
- Planner always reads last HumanMessage (not VERDICT: NO)
- Strict validation of planner operations (prevents dict a/b, missing keys like 'op')
- Reflect can END the graph (no unconditional reflect->planner edge)
- Fix tool_trace printing key: input (not args)
- Optional: wrap run in mlflow.start_run + log tool_trace artifact
"""

from __future__ import annotations

from typing import Annotated, Sequence, TypedDict, Literal, List, Dict, Any, Optional
import uuid
import operator
from pathlib import Path
from datetime import datetime
import json

import mlflow
from dotenv import load_dotenv

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, BaseMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool

from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field, ConfigDict

from logger_config import setup_logger


# ==============================
# Setup: logger + mlflow
# ==============================
load_dotenv()
logger = setup_logger(debug_mode=True, log_name="ToolAgent", log_dir="logs")

GRAPH_DIR = Path("Graphs")
GRAPH_DIR.mkdir(exist_ok=True)

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("ToolAgent")
mlflow.langchain.autolog()


# ==============================
# Calculator tool
# ==============================
class CalcInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    operation: Literal["add", "subtract", "multiply", "divide", "power"] = Field(...)
    a: float = Field(..., ge=-1e12, le=1e12)
    b: float = Field(0.0, ge=-1e12, le=1e12)


@tool(args_schema=CalcInput)
def calculator(operation: str, a: float, b: float = 0.0) -> float:
    """Financial calculator with per-call logging."""
    call_id = str(uuid.uuid4())[:8]
    logger.info(f"CALL {call_id}: {operation}(a={a}, b={b})")

    operations = {
        "add": operator.add,
        "subtract": operator.sub,
        "multiply": operator.mul,
        "divide": operator.truediv,
        "power": operator.pow,
    }

    try:
        if operation not in operations:
            raise ValueError(f"Invalid operation '{operation}'")

        if operation == "divide" and abs(b) < 1e-12:
            result = float("inf")
        else:
            result = operations[operation](a, b)
            result = round(float(result), 8)

        logger.info(f"RESULT {call_id}: {result}")
        return result

    except Exception as e:
        logger.error(f"ERROR {call_id}: {e}")
        return 0.0


# ==============================
# LLMs: JSON planner + text judge
# ==============================
PLANNER_MODEL_NAME = "granite4:350m"
JUDGE_MODEL_NAME = "llama3.2:1b"

planner_llm = ChatOllama(model=PLANNER_MODEL_NAME, temperature=0.0, format="json")
judge_llm = ChatOllama(model=JUDGE_MODEL_NAME, temperature=0.0)


# ==============================
# State
# ==============================
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    iteration: int
    operations: List[Dict[str, Any]]
    results: Dict[str, float]
    tool_trace: List[Dict[str, Any]]


# ==============================
# Helpers
# ==============================
def get_last_human_text(messages: Sequence[BaseMessage]) -> str:
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            return m.content
    return ""


def strip_json_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = t.removeprefix("```json").removeprefix("```").strip()
        if t.endswith("```"):
            t = t.removesuffix("```").strip()
    return t


def parse_first_json_object(text: str) -> dict:
    t = strip_json_fences(text)
    start = t.find("{")
    if start < 0:
        raise ValueError("No JSON object found")
    obj, _ = json.JSONDecoder().raw_decode(t[start:])
    return obj


def validate_operations(operations: Any) -> None:
    if not isinstance(operations, list) or len(operations) == 0:
        raise ValueError("operations must be a non-empty list")

    allowed_ops = {"add", "subtract", "multiply", "divide", "power"}

    def ok_val(x: Any) -> bool:
        return isinstance(x, (int, float)) or (isinstance(x, str) and x.startswith("{{") and x.endswith("}}"))

    for i, op in enumerate(operations):
        if not isinstance(op, dict):
            raise ValueError(f"op[{i}] must be an object")

        if "op" not in op or "a" not in op:
            raise ValueError(f"op[{i}] missing required keys ('op','a')")

        if op["op"] not in allowed_ops:
            raise ValueError(f"op[{i}].op invalid: {op['op']}")

        a = op["a"]
        b = op.get("b", 0.0)

        if isinstance(a, dict) or isinstance(b, dict):
            raise ValueError(f"op[{i}] produced dict for a/b (a={type(a)}, b={type(b)})")

        if not ok_val(a) or not ok_val(b):
            raise ValueError(f"op[{i}] invalid a/b types: a={type(a)}, b={type(b)}")


# ==============================
# 1) Planner
# ==============================
planner_system_text = """
Return JSON only. No markdown. No extra keys.

Schema:
{"operations":[{"step":"name","op":"multiply|divide|add|subtract|power","a":<number or "{{step}}">,"b":<number or "{{step}}">}]}

For simple interest (SI = P*R*T):
- Convert percentage R% to decimal: R/100 (e.g., 3% -> 0.03)
Example:
{"operations":[
  {"step":"rate","op":"multiply","a":1000,"b":0.03},
  {"step":"final","op":"multiply","a":"{{rate}}","b":2}
]}
""".strip()

planner_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage(content=planner_system_text),
        ("human", "{query}"),
    ]
)
planner_chain = planner_prompt | planner_llm


def planner_node(state: AgentState) -> dict:
    iteration = state.get("iteration", 0)
    query = get_last_human_text(state["messages"])
    logger.info(f"Planner iter{iteration}: '{query[:60]}...'")

    plan = planner_chain.invoke({"query": query})

    try:
        obj = parse_first_json_object(plan.content)
        operations = obj["operations"]
        validate_operations(operations)

        logger.info(f"Planner: {len(operations)} operations queued")
        return {
            "messages": [AIMessage(content=f"Planned {len(operations)} steps")],
            "iteration": iteration,
            "operations": operations,
            "results": {},
            "tool_trace": [],
        }

    except Exception as e:
        logger.error(f"Planner failed: {e}")
        return {
            "messages": [AIMessage(content="PLAN_ERROR")],
            "iteration": iteration + 1,  # bump so router can stop after N failures
            "operations": [],
            "results": {},
            "tool_trace": [],
        }


# ==============================
# 2) Tools executor
# ==============================
def tool_node(state: AgentState) -> dict:
    operations = state.get("operations", [])
    results = state.get("results", {})
    trace = state.get("tool_trace", [])

    logger.info(f"Executor: {len(operations)} operations")

    tool_msgs: List[ToolMessage] = []

    for i, op in enumerate(operations):
        step_id = op.get("step", f"step{i+1}")

        try:
            a = op["a"]
            b = op.get("b", 0.0)

            if isinstance(a, str) and a.startswith("{{") and a.endswith("}}"):
                a = results.get(a[2:-2], 0.0)
            if isinstance(b, str) and b.startswith("{{") and b.endswith("}}"):
                b = results.get(b[2:-2], 0.0)

            if isinstance(a, dict) or isinstance(b, dict):
                raise ValueError(f"Bad a/b types after resolve: a={type(a)}, b={type(b)}")

            args = CalcInput(operation=op["op"], a=float(a), b=float(b))
            result = calculator.invoke(args.model_dump())

            results[step_id] = result
            trace.append(
                {
                    "step": step_id,
                    "op": op["op"],
                    "input": {"a": a, "b": b},
                    "output": result,
                }
            )

            tool_msgs.append(
                ToolMessage(
                    content=str(result),
                    tool_call_id=str(uuid.uuid4()),
                    name="calculator",
                )
            )

            logger.info(f"Step {i+1}: {step_id} = {result}")

        except Exception as e:
            logger.error(f"Step {i+1} error ({step_id}): {e}")
            results[step_id] = 0.0
            trace.append(
                {
                    "step": step_id,
                    "op": op.get("op", "UNKNOWN"),
                    "input": {"a": op.get("a"), "b": op.get("b")},
                    "output": 0.0,
                    "error": str(e),
                }
            )

    logger.info(f"Executor done: {len(results)} results")
    return {"messages": tool_msgs, "results": results, "tool_trace": trace}


# ==============================
# 3) Formatter
# ==============================
def formatter_node(state: AgentState) -> dict:
    operations = state.get("operations", [])
    results = state.get("results", {})

    steps = []
    for op in operations:
        step_id = op.get("step", "unknown")
        r = results.get(step_id, 0.0)
        steps.append(f"{op.get('op','?')}({op.get('a','?')},{op.get('b','?')})={r}")

    final_value = results.get("final", (list(results.values())[-1] if results else 0.0))

    final = f"""1. FORMULA: SI = P * R * T / 100
2. VALUES: Parsed from query
3. STEPS: {' → '.join(steps)}
4. RESULT: {float(final_value):.2f}"""

    logger.info("Format complete")
    return {"messages": [AIMessage(content=final.strip())]}


# ==============================
# 4) Reflect (YES/NO)
# ==============================
reflect_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """YES if format perfect:
- 1.FORMULA, 2.VALUES P/R/T, 3.STEPS w/ calculator calls, 4.RESULT
NO otherwise. ONLY "YES"/"NO".""",
        ),
        MessagesPlaceholder("messages"),
    ]
)
reflect_chain = reflect_prompt | judge_llm


def reflect_node(state: AgentState) -> dict:
    messages = state["messages"]

    user_query: Optional[HumanMessage] = None
    for msg in messages:
        if isinstance(msg, HumanMessage):
            user_query = msg
            break
    if not user_query:
        user_query = HumanMessage(content="Unknown query")

    final_answer: Optional[AIMessage] = None
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and "1. FORMULA:" in msg.content:
            final_answer = msg
            break
    if not final_answer:
        for msg in reversed(messages):
            if isinstance(msg, AIMessage):
                final_answer = msg
                break
    if not final_answer:
        final_answer = AIMessage(content="No output generated.")

    reflection_messages = [user_query, final_answer]
    logger.info("Reflect input: HumanMessage → AIMessage")

    try:
        verdict = reflect_chain.invoke({"messages": reflection_messages})
        verdict_text = verdict.content.strip().upper()

        if verdict_text not in ("YES", "NO"):
            logger.warning(f"Invalid verdict: '{verdict.content}'. Forcing NO.")
            verdict_text = "NO"

        logger.info(f"Reflect verdict: {verdict_text}")
        return {
            "messages": [AIMessage(content=f"VERDICT: {verdict_text}")],
            "iteration": state.get("iteration", 0) + 1,
        }

    except Exception as e:
        logger.error(f"Reflection failed: {e}")
        return {
            "messages": [AIMessage(content="VERDICT: NO")],
            "iteration": state.get("iteration", 0) + 1,
        }


# ==============================
# Router
# ==============================
def route_node(state: AgentState) -> Literal["planner", "tools", "format", "reflect", END]:
    iteration = state.get("iteration", 0)
    ops = state.get("operations", [])
    results = state.get("results", {})

    if iteration >= 3:
        return END

    if state.get("messages"):
        last_msg = state["messages"][-1]
        if isinstance(last_msg, AIMessage):
            text = last_msg.content.upper()
            if "VERDICT: YES" in text:
                return END
            if "VERDICT: NO" in text:
                return "planner"

    if not ops:
        return "planner"

    if ops and not results:
        return "tools"

    if results:
        # If we already formatted, go to reflect; else format now
        last = state["messages"][-1] if state.get("messages") else None
        if isinstance(last, AIMessage) and "1. FORMULA:" in last.content:
            return "reflect"
        return "format"

    return "planner"


# ==============================
# Build graph
# ==============================
def create_workflow():
    graph = StateGraph(AgentState)

    graph.add_node("planner", planner_node)
    graph.add_node("tools", tool_node)
    graph.add_node("format", formatter_node)
    graph.add_node("reflect", reflect_node)

    graph.set_entry_point("planner")

    path_map = {"planner": "planner", "tools": "tools", "format": "format", "reflect": "reflect", END: END}

    graph.add_conditional_edges("planner", route_node, path_map)
    graph.add_conditional_edges("tools", route_node, path_map)
    graph.add_conditional_edges("format", route_node, path_map)
    graph.add_conditional_edges("reflect", route_node, path_map)

    app = graph.compile()

    png_path = GRAPH_DIR / f"pipeline_{datetime.now().strftime('%H%M%S')}.png"
    try:
        png_data = app.get_graph().draw_mermaid_png()
        png_path.write_bytes(png_data)
        logger.info(f"PNG saved: {png_path}")
    except Exception as e:
        logger.warning(f"Graph render failed: {e}")

    return app, png_path


# ==============================
# Run
# ==============================
if __name__ == "__main__":
    query = "Simple Interest: P=1000, R=3%, T=2 years"

    logger.info("Pipeline started")
    print("=" * 60)

    app, png_path = create_workflow()

    with mlflow.start_run(run_name="ToolAgent_debug"):
        result = app.invoke(
            {"messages": [HumanMessage(content=query)], "iteration": 0},
            {"recursion_limit": 100},  # helpful while debugging loops
        )

        mlflow.log_text(query, "input/query.txt")
        mlflow.log_dict(result.get("tool_trace", []), "debug/tool_trace.json")
        if png_path.exists():
            mlflow.log_artifact(str(png_path), artifact_path="debug/graph")

    print("=" * 60)

    trace = result.get("tool_trace", [])
    if trace:
        print("\nTOOL EXECUTION:")
        for t in trace:
            inp = t.get("input", {})
            print(f"Step {t.get('step')}: {t.get('op')}({inp.get('a')},{inp.get('b')}) = {t.get('output')}")

    print("\nRESULT:")
    print("-" * 40)
    print(result["messages"][-1].content.strip())
