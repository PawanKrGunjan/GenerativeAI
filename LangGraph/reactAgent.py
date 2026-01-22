import json
import logging
import os
import ast
import math
import re
from typing import Annotated, Sequence, Optional, TypedDict, Any, List, Dict

from pydantic import BaseModel, Field

from langchain_ollama import ChatOllama
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchResults

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages


# ----------------------------
# Logging setup
# ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger("react-agent")


def safe_json_dumps(x: Any) -> str:
    """Always return a JSON string (fallback to str for non-serializable objects)."""
    try:
        return json.dumps(x, ensure_ascii=False)
    except TypeError:
        return json.dumps(str(x), ensure_ascii=False)

def safe_json_loads(x: Any) -> Any:
    """Parse JSON safely; if already a Python object, return as-is."""
    if isinstance(x, (list, dict)):
        return x
    if not isinstance(x, str):
        return x
    try:
        return json.loads(x)
    except Exception:
        return x

# ----------------------------
# Tools
# ----------------------------
ddg = DuckDuckGoSearchResults(num_results=5)

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


# ----------------------------
# Calculator tool (safe AST)
# ----------------------------
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


# ----------------------------
# News summarizer tool
# ----------------------------
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
        return (
            "Could not parse structured search results. "
            "Try adjusting the query or switching search provider."
        )

    top_items = items[: input.top_k]
    blocks: List[str] = []

    for i, it in enumerate(top_items, start=1):
        title = it.get("title") or it.get("heading") or "Untitled"
        link = it.get("link") or it.get("url") or ""
        snippet = it.get("snippet") or it.get("body") or ""

        date = _extract_date(snippet)
        points = _make_main_points(snippet, input.focus)

        md: List[str] = []
        md.append(f"{i}. {title}")
        if date:
            md.append(f"   - Date (from snippet): {date}")
        if link:
            md.append(f"   - Link: {link}")
        if points:
            md.append("   - Main points:")
            for p in points:
                md.append(f"     - {p}")
        else:
            md.append("   - Main points: (not enough snippet text to extract)")

        blocks.append("\n".join(md))

    return "Top news summaries:\n\n" + "\n\n".join(blocks)



TOOLS = [search_tool, recommend_clothing, calculator_tool, news_summarizer_tool]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}


# ----------------------------
# Model + prompt
# ----------------------------
MODEL_NAME = "granite4:350m"
llm = ChatOllama(model=MODEL_NAME, temperature=0.0)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """
You are a helpful assistant. Use tools when needed.

Rules:
- For web/current updates: call search_tool.
- For calculations: call calculator_tool with {{"expression": "2 + 3 * 4"}}.
- For clothing: call recommend_clothing with {{"temperature_c": 25, "description": "Sunny"}}.
- For news summaries:
  1) Call search_tool first.
  2) Then call news_summarizer_tool with:
     {{"search_results_json": "<output from search_tool>", "top_k": 3}}.
"""),
        MessagesPlaceholder(variable_name="messages"),
    ]
)


agent_runnable = prompt | llm.bind_tools(TOOLS)


# ----------------------------
# LangGraph state
# ----------------------------
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


def agent_node(state: AgentState) -> dict:
    last = state["messages"][-1]
    logger.info("AGENT | last_message_type=%s", type(last).__name__)

    response = agent_runnable.invoke({"messages": state["messages"]})

    if getattr(response, "tool_calls", None):
        logger.info("AGENT | tool_calls=%s", [tc["name"] for tc in response.tool_calls])
    else:
        logger.info("AGENT | no tool calls; final answer ready")

    return {"messages": [response]}


def tools_node(state: AgentState) -> dict:
    last = state["messages"][-1]
    tool_calls = getattr(last, "tool_calls", []) or []
    logger.info("TOOLS | executing %d tool call(s)", len(tool_calls))

    outputs = []
    for tc in tool_calls:
        name = tc["name"]
        args = tc.get("args", {}) or {}
        tc_id = tc.get("id")

        logger.info("TOOLS | call name=%s args=%s", name, args)

        try:
            result = TOOLS_BY_NAME[name].invoke(args)
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
    if getattr(last, "tool_calls", None):
        return "tools"
    return "end"


builder = StateGraph(AgentState)
builder.add_node("agent", agent_node)
builder.add_node("tools", tools_node)

builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", route, {"tools": "tools", "end": END})
builder.add_edge("tools", "agent")

graph = builder.compile()


# ----------------------------
# MAIN
# ----------------------------
if __name__ == "__main__":
    os.makedirs("Graphs", exist_ok=True)

    questions = [
        "What's the weather like in Delhi today, what should I wear",
        "calculate 2 + 3 * 4 and sin(pi/2).",
        "What is the latest update on Patna, Kankarbagh Sambhu Girls Hostel Case",
    ]

    # Mermaid graph once
    mermaid = graph.get_graph().draw_mermaid()
    print("\n================= MERMAID GRAPH =================")
    print(mermaid)
    with open("Graphs/reactAgent.mmd", "w", encoding="utf-8") as f:
        f.write(mermaid)

    for user_query in questions:
        inputs = {"messages": [HumanMessage(content=user_query)]}

        for event in graph.stream(inputs, stream_mode="values"):
            msg = event["messages"][-1]
            logger.info("STREAM | %s", type(msg).__name__)
            if getattr(msg, "content", None):
                print(f"\n--- {type(msg).__name__} ---\n{msg.content}")
