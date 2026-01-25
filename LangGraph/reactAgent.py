"""
reactAgent.py 

What it does
- A small LangGraph "tool loop": agent -> tools -> agent -> ... -> end
- Agent is an Ollama chat model with tool calling enabled
- Tools included:
  - search_tool (DuckDuckGo)
  - calculator_tool (safe AST eval)
  - recommend_clothing (simple rules)
  - news_summarizer_tool (summarize DDG results)

Logging
- Uses logger_config.setup_logger()
- Writes rotating logs to logs/react-agent.log (per your logger_config)

Common gotcha fixed
- Curly braces in ChatPromptTemplate must be escaped: {{ and }} for literal JSON examples.
"""

from __future__ import annotations

import os
import json
import ast
import math
import re
from pathlib import Path
from typing import Annotated, Sequence, Optional, TypedDict, Any, List, Dict

import mlflow
from pydantic import BaseModel, Field

from langchain_ollama import ChatOllama
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchResults

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from logger_config import setup_logger


# =============================================================================
# CONFIG
# =============================================================================
class Config:
    DEBUG = os.getenv("DEBUG", "1") == "1"

    LOG_DIR = os.getenv("LOG_DIR", "logs")
    LOG_NAME = os.getenv("LOG_NAME", "react-agent")

    GRAPH_DIR = Path(os.getenv("GRAPH_DIR", "Graphs"))
    SAVE_GRAPH_PNG = os.getenv("SAVE_GRAPH_PNG", "1") == "1"

    MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    MLFLOW_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT", "react-agent")

    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "granite4:350m")
    TEMPERATURE = float(os.getenv("TEMPERATURE", "0.0"))

    DDG_RESULTS = int(os.getenv("DDG_RESULTS", "5"))

    RECURSION_LIMIT = int(os.getenv("RECURSION_LIMIT", "30"))


# =============================================================================
# LOGGER (logger_config)
# =============================================================================
logger = setup_logger(
    debug_mode=Config.DEBUG,
    log_name=Config.LOG_NAME,
    log_dir=Config.LOG_DIR,
)
logger.info("Logger configured")


# =============================================================================
# MLFLOW
# =============================================================================
mlflow.set_tracking_uri(Config.MLFLOW_TRACKING_URI)
mlflow.set_experiment(Config.MLFLOW_EXPERIMENT)
mlflow.langchain.autolog()


# =============================================================================
# JSON helpers
# =============================================================================
def safe_json_dumps(x: Any) -> str:
    try:
        return json.dumps(x, ensure_ascii=False)
    except TypeError:
        return json.dumps(str(x), ensure_ascii=False)


def safe_json_loads(x: Any) -> Any:
    if isinstance(x, (list, dict)):
        return x
    if not isinstance(x, str):
        return x
    try:
        return json.loads(x)
    except Exception:
        return x


# =============================================================================
# TOOLS
# =============================================================================
ddg = DuckDuckGoSearchResults(num_results=Config.DDG_RESULTS)


@tool
def search_tool(query: str) -> str:
    """Search the web using DuckDuckGo and return results as JSON text."""
    raw = ddg.invoke(query)
    return safe_json_dumps(raw)


class ClothingInput(BaseModel):
    temperature_c: Optional[float] = Field(default=None, description="Temperature in Celsius")
    description: Optional[str] = Field(default=None, description="Weather description text (optional)")


@tool
def recommend_clothing(input: ClothingInput) -> str:
    """Return clothing advice based on temperature (°C) or description."""
    desc = (input.description or "").lower()

    if input.temperature_c is not None:
        t = float(input.temperature_c)
        if t <= 5:
            return "Heavy jacket/coat, warm layers, gloves recommended."
        if t <= 15:
            return "Light jacket or sweater is good."
        if t <= 25:
            return "T-shirt or light shirt; carry a thin layer for evening."
        return "Light, breathable clothes; stay hydrated."

    if "snow" in desc or "freezing" in desc:
        return "Wear a heavy coat, gloves, and boots."
    if "rain" in desc or "wet" in desc:
        return "Bring a raincoat and waterproof shoes."
    if "hot" in desc:
        return "T-shirt/shorts and sunscreen recommended."
    if "cold" in desc:
        return "Wear a warm jacket or sweater."
    return "A light jacket should be fine."


SAFE_NAMES = {
    "pi": math.pi,
    "e": math.e,
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "abs": abs,
    "round": round,
}

ALLOWED_NODES = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Call,
    ast.Name,
    ast.Load,
    ast.Constant,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod, ast.FloorDiv,
    ast.UAdd, ast.USub,
)


class UnsafeExpressionError(ValueError):
    pass


def _validate_ast(node: ast.AST) -> None:
    for n in ast.walk(node):
        if not isinstance(n, ALLOWED_NODES):
            raise UnsafeExpressionError(f"Disallowed syntax: {type(n).__name__}")

        if isinstance(n, ast.Call):
            if not isinstance(n.func, ast.Name):
                raise UnsafeExpressionError("Only simple function calls allowed (e.g., sqrt(16)).")
            if n.func.id not in SAFE_NAMES:
                raise UnsafeExpressionError(f"Unknown function: {n.func.id}")

        if isinstance(n, ast.Name) and n.id not in SAFE_NAMES:
            raise UnsafeExpressionError(f"Unknown identifier: {n.id}")


@tool
def calculator_tool(expression: str) -> str:
    """Safely evaluate math like '2+3*4', 'sqrt(16)', 'sin(pi/2)'."""
    expr = expression.strip().replace("π", "pi")
    try:
        tree = ast.parse(expr, mode="eval")
        _validate_ast(tree)
        result = eval(
            compile(tree, filename="<calculator_tool>", mode="eval"),
            {"__builtins__": {}},
            SAFE_NAMES,
        )
        return str(result)
    except (SyntaxError, UnsafeExpressionError, ValueError, TypeError, ZeroDivisionError) as e:
        return f"Error: {e}"


class NewsSummarizeInput(BaseModel):
    search_results_json: str = Field(..., description="JSON string returned by search_tool.")
    top_k: int = Field(3, ge=1, le=10)
    focus: Optional[str] = Field(None, description="Optional topic focus.")


def _normalize_results(obj: Any) -> List[Dict[str, Any]]:
    if isinstance(obj, list):
        return [x for x in obj if isinstance(x, dict)]
    if isinstance(obj, dict):
        for key in ("results", "items", "data"):
            if key in obj and isinstance(obj[key], list):
                return [x for x in obj[key] if isinstance(x, dict)]
        return []
    return []


def _extract_date(text: str) -> Optional[str]:
    if not text:
        return None
    patterns = [
        r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},\s+\d{4}\b",
        r"\b\d{4}-\d{2}-\d{2}\b",
        r"\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}\b",
    ]
    for p in patterns:
        m = re.search(p, text)
        if m:
            return m.group(0)
    return None


def _make_main_points(snippet: str, focus: Optional[str] = None) -> List[str]:
    if not snippet:
        return []
    s = re.sub(r"\s+", " ", snippet).strip()
    parts = re.split(r"[.;•\-]\s+", s)
    parts = [p.strip() for p in parts if len(p.strip()) >= 25]

    if focus:
        f = focus.lower()
        focused = [p for p in parts if f in p.lower()]
        parts = focused + [p for p in parts if p not in focused]

    return parts[:3]


@tool
def news_summarizer_tool(input: NewsSummarizeInput) -> str:
    """Summarize top-k articles from DuckDuckGo search results JSON."""
    raw_obj = safe_json_loads(input.search_results_json)
    items = _normalize_results(raw_obj)

    if not items:
        return "Could not parse structured search results. Try adjusting the query."

    top_items = items[: input.top_k]
    blocks: List[str] = []

    for i, it in enumerate(top_items, start=1):
        title = it.get("title") or it.get("heading") or "Untitled"
        link = it.get("link") or it.get("url") or ""
        snippet = it.get("snippet") or it.get("body") or ""

        date = _extract_date(snippet)
        points = _make_main_points(snippet, input.focus)

        lines: List[str] = []
        lines.append(f"{i}. {title}")
        if date:
            lines.append(f"   - Date (from snippet): {date}")
        if link:
            lines.append(f"   - Link: {link}")
        if points:
            lines.append("   - Main points:")
            for p in points:
                lines.append(f"     - {p}")
        else:
            lines.append("   - Main points: (not enough snippet text to extract)")

        blocks.append("\n".join(lines))

    return "Top news summaries:\n\n" + "\n\n".join(blocks)


TOOLS = [search_tool, recommend_clothing, calculator_tool, news_summarizer_tool]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}


# =============================================================================
# MODEL + PROMPT (IMPORTANT: escape braces {{ }} )
# =============================================================================
llm = ChatOllama(model=Config.OLLAMA_MODEL, temperature=Config.TEMPERATURE)

SYSTEM_PROMPT = """
You are a helpful assistant. Use tools when needed.

Rules:
- For web/current updates: call search_tool.
- For calculations: call calculator_tool with {{"expression": "2 + 3 * 4"}}.
- For clothing: call recommend_clothing with {{"temperature_c": 25, "description": "Sunny"}}.
- For news summaries:
  1) Call search_tool first.
  2) Then call news_summarizer_tool with:
     {{"search_results_json": "<output from search_tool>", "top_k": 3}}.

Return the final answer after tools finish.
""".strip()

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

agent_runnable = prompt | llm.bind_tools(TOOLS)


# =============================================================================
# LANGGRAPH STATE
# =============================================================================
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


def agent_node(state: AgentState) -> dict:
    last = state["messages"][-1]
    logger.info("AGENT | last_message_type=%s", type(last).__name__)

    response = agent_runnable.invoke({"messages": state["messages"]})

    tool_calls = getattr(response, "tool_calls", None) or []
    if tool_calls:
        logger.info("AGENT | tool_calls=%s", [tc.get("name") for tc in tool_calls])
    else:
        logger.info("AGENT | no tool calls -> ready to answer")

    return {"messages": [response]}


def tools_node(state: AgentState) -> dict:
    last = state["messages"][-1]
    tool_calls = getattr(last, "tool_calls", None) or []
    logger.info("TOOLS | executing %d tool call(s)", len(tool_calls))

    outputs: List[ToolMessage] = []

    for tc in tool_calls:
        name = tc.get("name")
        args = tc.get("args", {}) or {}
        tc_id = tc.get("id")

        logger.info("TOOLS | call name=%s args=%s", name, args)

        tool_obj = TOOLS_BY_NAME.get(name)
        if tool_obj is None:
            err = {"error": f"Unknown tool: {name}", "tool": name, "args": args}
            logger.error("TOOLS | %s", err)
            outputs.append(ToolMessage(content=safe_json_dumps(err), name=str(name), tool_call_id=tc_id))
            continue

        try:
            result = tool_obj.invoke(args)
        except Exception as e:
            logger.exception("TOOLS | tool failed: %s", name)
            result = {"error": str(e), "tool": name, "args": args}

        outputs.append(
            ToolMessage(
                content=safe_json_dumps(result),
                name=name,
                tool_call_id=tc_id,
            )
        )

    return {"messages": outputs}


def route(state: AgentState) -> str:
    last = state["messages"][-1]
    return "tools" if getattr(last, "tool_calls", None) else "end"


def build_graph():
    builder = StateGraph(AgentState)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", tools_node)

    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", route, {"tools": "tools", "end": END})
    builder.add_edge("tools", "agent")

    return builder.compile()


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    graph = build_graph()

    Config.GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    mermaid = graph.get_graph().draw_mermaid()
    (Config.GRAPH_DIR / "reactAgent.mmd").write_text(mermaid, encoding="utf-8")
    logger.info("Saved graph mermaid: %s", str(Config.GRAPH_DIR / "reactAgent.mmd"))

    if Config.SAVE_GRAPH_PNG:
        try:
            png_bytes = graph.get_graph().draw_mermaid_png()
            (Config.GRAPH_DIR / "reactAgent.png").write_bytes(png_bytes)
            logger.info("Saved graph PNG: %s", str(Config.GRAPH_DIR / "reactAgent.png"))
        except Exception as e:
            logger.warning("Graph PNG render failed: %r", e)

    questions = [
        "What's the weather like in Delhi today, what should I wear?",
        "Calculate 2 + 3 * 4 and sin(pi/2).",
        "What is the latest update on Patna, Kankarbagh Sambhu Girls Hostel Case?",
    ]

    for q in questions:
        logger.info("QUERY | %s", q)

        inputs: AgentState = {"messages": [HumanMessage(content=q)]}

        # One clean run (invoke). If you want step-by-step, you can switch to stream().
        final_state = graph.invoke(inputs, config={"recursion_limit": Config.RECURSION_LIMIT})
        final_msg = final_state["messages"][-1]

        logger.info("FINAL_MESSAGE_TYPE | %s", type(final_msg).__name__)
        logger.info("FINAL_CONTENT | %s", getattr(final_msg, "content", "")[:2000])

        print("\n" + "=" * 80)
        print("USER:", q)
        print("-" * 80)
        print(getattr(final_msg, "content", ""))
