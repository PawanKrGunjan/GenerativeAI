import json,re
from typing import List, Dict, Any, Optional
from datetime import datetime
from zoneinfo import ZoneInfo

from langgraph.graph import StateGraph, END
from langgraph.store.memory import InMemoryStore
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    ToolMessage,
)

from pydantic import BaseModel, Field
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)

# Local imports
from agents.llm import get_llm, embed_text, EMBEDDING_DIM
from tools.tool_registry import TOOLS
from utils.logger import LOGGER
from utils.config import GRAPH_DIR
from utils.db_connect import get_connection
from memory.store_memory import update_reflection


IST = ZoneInfo("Asia/Kolkata")

# =============================
# LLM & Tools
# =============================
LLM = get_llm(temperature=0.0)
LLM_WITH_TOOLS = LLM.bind_tools(
    TOOLS,
    tool_choice="required"
)
TOOL_MAP = {tool.name: tool for tool in TOOLS}

store = InMemoryStore(index={"embed": embed_text, "dims": EMBEDDING_DIM})
db_conn = get_connection(LOGGER)

# =============================
# Constants
# =============================
MAX_MSG_HISTORY = 12         # Prevent context overflow
MAX_MEMORY = 10
MAX_ATTEMPTS = 7

# =============================
# Main Agent Prompt (final optimized version)
# =============================
system_template = """\
You are a disciplined Indian stock market advisor (NSE & BSE only).

Current time (IST): {current_datetime}
Attempt number: {attempt_count}

───────────────────────────────────────────────
STATE SNAPSHOT

Known companies:      {company_name}
Known symbols:        {symbols}
Available price data: {prices}
Recent news context:  {news}
Past analyses memory: {memory}
Tool calls so far:    {tool_history}
───────────────────────────────────────────────

STRICT RULES - YOU MUST FOLLOW THESE EVERY TIME

1. NEVER guess or hallucinate any of the following:
   • ticker symbols
   • current/last traded price
   • change %, volume, market cap, etc.
   • technical indicator values
   • financial ratios

2. You MUST use tools to obtain any missing data.

3. You are ONLY allowed to output the final advice block when ALL of these are true:
   ✓ At least one company name is clearly identified
   ✓ The corresponding NSE/BSE ticker symbol is known
   ✓ Fresh price/market data for that symbol exists in {prices}
   ✓ You have sufficient information to give a reasoned opinion

   If ANY condition is missing → call the appropriate tool.
4. When you need data, call the appropriate tool. Do not write the tool name in text. Use the tool calling mechanism.
4. When you are ready to give advice, output **ONLY** the following exact structure — nothing else before or after:

\n
Company:        {{full company name}}
Ticker:         {{NSE or BSE symbol}}
Price:          ₹{{last traded price, use comma for thousands}}
Updated:        {{DD-MMM-YYYY HH:MM IST  or "latest available"}}
Signal:         BUY | HOLD | SELL | NEUTRAL
Confidence:     XX/100

Pointwise Reasoning:
• first concise analytical point
• second point
• ...
• 5-10 bullet points max

You are **strictly forbidden** for guessing the price or ticker, Always use Tool Call to fetch Symbol and Price.

───────────────────────────────────────────────
MANDATORY STEP-BY-STEP WORKFLOW

1. Read the user message → identify the main company or index mentioned.
2. If company name is mentioned but no symbol is known yet → call lookup_stock_symbol(...)
3. Once you have a symbol → call get_stock_info(symbol="SYMBOL")
4. If more context is needed → use relevant tools.
5. ONLY when price data is available → output the final block above.

───────────────────────────────────────────────
IMPORTANT REMINDERS
• The correct next action when data is missing is almost always to call a tool.
• Do NOT try to be helpful by giving advice too early.
• Keep all reasoning inside the bullet points.
"""

PROMPT = ChatPromptTemplate.from_messages([
    ("system", system_template),
    MessagesPlaceholder(variable_name="messages"),
])

LLM_CHAIN = PROMPT | LLM_WITH_TOOLS

# =============================
# Agent State
# =============================
class InvestmentAgentState(BaseModel):
    messages: List[BaseMessage] = Field(default_factory=list)

    company_name: List[str] = Field(default_factory=list)
    symbols: Dict[str, str] = Field(default_factory=dict)

    prices: Dict[str, Any] = Field(default_factory=dict)
    news: Dict[str, Any] = Field(default_factory=dict)

    memory: List[str] = Field(default_factory=list)
    tool_history: List[Dict[str, Any]] = Field(default_factory=list)

    result: Optional[str] = None

    attempt_count: int = 0
    current_datetime: datetime = Field(default_factory=lambda: datetime.now(IST))


# =============================
# Helper Functions
# =============================
def format_tool_history(results: list) -> str:
    if not results:
        return "None"
    lines = [
        f"Step {r['step']}:\n"
        f"Tool: {r['tool']}\n"
        f"Args: {r['args']}\n"
        f"Result: {r['result']}\n"
        for r in results
    ]
    return "\n".join(lines)


def compact_tool_result(result: Any) -> str:
    """Compact tool output to save tokens (critical for long conversations)."""
    if isinstance(result, dict):
        if result.get("status") == "success":
            data = result.get("data", {})
            short = {
                "symbol": result.get("symbol"),
                "last_price": data.get("last_price"),
                "change_percent": data.get("change_percent"),
                "volume": data.get("volume"),
                "timestamp": data.get("timestamp"),
            }
            return json.dumps({k: v for k, v in short.items() if v is not None}, default=str)
        else:
            return json.dumps(result, default=str)[:1500]
    return str(result)[:1500]


def format_state_for_prompt(state: InvestmentAgentState) -> Dict[str, Any]:
    """Clean, readable formatting for the system prompt."""
    return {
        "current_datetime": state.current_datetime.strftime("%Y-%m-%d %H:%M IST"),
        "attempt_count": state.attempt_count,
        "company_name": ", ".join(state.company_name) if state.company_name else "None",
        "symbols": json.dumps(state.symbols, indent=2) if state.symbols else "None",
        "prices": json.dumps(
            {k: {kk: vv for kk, vv in v.items() if kk in ["last_price", "change_percent", "volume"]}
             for k, v in state.prices.items()},
            default=str
        ) if state.prices else "None",
        "news": json.dumps(state.news, default=str) if state.news else "None",
        "memory": "\n".join(state.memory[-5:]) if state.memory else "None",  # last 5 only
        "tool_history": format_tool_history(state.tool_history),
    }


def clean_final_advice(content: str) -> Optional[str]:
    """Safety net: ensures the exact format and removes any junk."""
    content = content.strip()
    if not content.startswith("Stock-Specific Advice"):
        return None

    # Keep only the block (in case model added extra text)
    lines = content.splitlines()
    output = []
    in_block = False
    for line in lines:
        if line.strip().startswith("Stock-Specific Advice"):
            in_block = True
            output.append(line)
            continue
        if in_block:
            output.append(line)
    return "\n".join(output) if len(output) >= 6 else None


# =============================================================
# Nodes
# =============================================================
def reasoning_node(state: InvestmentAgentState):
    LOGGER.info(f"NODE → REACT (attempt {state.attempt_count + 1})")

    # Fresh timestamp
    state.current_datetime = datetime.now(IST)
    new_attempt = state.attempt_count + 1

    # Prepare clean input for prompt
    input_dict = format_state_for_prompt(state)
    input_dict["messages"] = state.messages[-MAX_MSG_HISTORY:]

    # Call LLM
    response = LLM_CHAIN.invoke(input_dict)

    LOGGER.info(f"LLM RESPONSE → {response.content[:200]}...")

    updates = {
        "attempt_count": new_attempt,
        "messages": state.messages + [response],
    }

    tool_calls = getattr(response, "tool_calls", None) or []
    if tool_calls:
        LOGGER.info(f"LLM requested {len(tool_calls)} tool(s)")
        updates["messages"] = updates["messages"][-MAX_MSG_HISTORY:]
        return updates

    # Detect final advice (new format - no "Stock-Specific Advice" header)
    content = (response.content or "").strip()
    if "Company:" in content and "Ticker:" in content and "Price:" in content and "Signal:" in content:

        prices = state.prices or {}
        symbols = state.symbols or {}

        # === SIMPLE CHECK (all symbols are now guaranteed .NS) ===
        missing_prices = [
            (company, symbol)
            for company, symbol in symbols.items()
            if symbol not in prices
        ]

        # Strong loop protection
        recent_msgs = [m.content for m in state.messages[-5:] if isinstance(m, HumanMessage)]
        if missing_prices and any("STOP. You attempted" in msg for msg in recent_msgs):
            LOGGER.warning("Loop detected – advice without price data after reminder")
            return {
                "attempt_count": new_attempt,
                "result": "ERROR: Missing price data after repeated attempts. Analysis blocked.",
                "messages": state.messages + [AIMessage(content="Analysis blocked due to missing price data.")],
            }

        if missing_prices:
            company, symbol = missing_prices[0]
            LOGGER.warning(f"Advice generated without price → forcing tool: {symbol}")

            reminder = HumanMessage(
                content=(
                    f"STOP. You attempted to give advice without price data.\n\n"
                    f"First call the tool:\n"
                    f"get_stock_info(symbol='{symbol}')\n\n"
                    f"Do NOT give advice until the tool result is available."
                )
            )
            updates["messages"] = (state.messages + [response, reminder])[-MAX_MSG_HISTORY:]
            return updates

        # Valid final advice
        cleaned = clean_final_advice(content)
        final_content = cleaned or content

        memory_entry = (
            f"[{state.current_datetime.strftime('%Y-%m-%d %H:%M')}] "
            f"{final_content[:350]}..."
        )

        LOGGER.info("Final advice generated successfully")
        return {
            "attempt_count": new_attempt,
            "messages": (state.messages + [response])[-MAX_MSG_HISTORY:],
            "result": final_content,
            "memory": (state.memory + [memory_entry])[-MAX_MEMORY:],
        }

    # Normal continuation
    updates["messages"] = updates["messages"][-MAX_MSG_HISTORY:]
    return updates


def execute_tool_calls(state: InvestmentAgentState):
    last_msg = state.messages[-1]
    tool_calls = getattr(last_msg, "tool_calls", []) or []

    LOGGER.info(f"Executing {len(tool_calls)} tool calls")

    new_messages = []
    new_symbols = dict(state.symbols)
    new_prices = dict(state.prices)
    company_name = list(state.company_name)
    tool_history = list(state.tool_history)

    step = len(tool_history) + 1

    for call in tool_calls:
        name = call.get("name") or call.get("function", {}).get("name")
        args = (
            call.get("args")
            or call.get("arguments")
            or call.get("function", {}).get("arguments", {})
            or {}
        )
        call_id = call.get("id") or f"tool_call_{step}"

        LOGGER.info(f"Tool → {name} | Args → {args}")

        try:
            result = TOOL_MAP[name].invoke(args)
        except Exception as e:
            LOGGER.error(f"Tool {name} failed: {e}")
            result = {"error": str(e)}

        # Symbol lookup handling (lookup now ALWAYS returns .NS)
        if name == "lookup_stock_symbol" and isinstance(result, list):
            for sym in result:
                company = sym.get("company_name")
                ticker = sym.get("symbol")          # ← guaranteed .NS
                if company and ticker:
                    if company not in company_name:
                        company_name.append(company)
                    new_symbols[company] = ticker   # stored with .NS

        # Price data handling (standard .NS only)
        if name == "get_stock_info":
            symbol = result.get("symbol", "")
            if symbol and result.get("status") == "success":
                data = result.get("data", {})
                new_prices[symbol] = data                    # store under .NS
                LOGGER.info(f"Stored price for {symbol}")

        # Tool history
        tool_history.append({
            "step": step,
            "tool": name,
            "args": args,
            "result": result,
            "time": datetime.now(IST).isoformat()
        })
        step += 1

        # Compact ToolMessage
        new_messages.append(
            ToolMessage(
                tool_call_id=call_id,
                name=name,
                content=compact_tool_result(result)
            )
        )

    return {
        "messages": (state.messages + new_messages)[-MAX_MSG_HISTORY:],
        "symbols": new_symbols,
        "company_name": company_name,
        "prices": new_prices,
        "tool_history": tool_history,
    }


def reflection_node(state: InvestmentAgentState):
    if not state.result or not state.symbols:
        return {}

    LOGGER.info("NODE → REFLECT")

    reflection = f"""
Time: {state.current_datetime.strftime('%Y-%m-%d %H:%M:%S')}
Companies: {state.company_name}
Symbols: {state.symbols}

Final Advice:
{state.result[:600]}...
"""
    try:
        update_reflection(state.symbols, reflection)
        LOGGER.info(f"Reflection stored for {list(state.symbols.keys())}")
    except Exception:
        LOGGER.exception("Reflection storage failed")

    return {}


# =============================================================
# Router
# =============================================================
def router(state: InvestmentAgentState):
    if not state.messages:
        return "REACT"

    last_msg = state.messages[-1]

    if getattr(last_msg, "tool_calls", None):
        return "TOOLS"

    if state.result:
        return "REFLECT"

    if state.attempt_count >= MAX_ATTEMPTS:
        LOGGER.warning("Max attempts reached → END")
        return END

    return "REACT"


# =============================================================
# Build Graph
# =============================================================
def build_graph():
    workflow = StateGraph(InvestmentAgentState)

    workflow.add_node("REACT", reasoning_node)
    workflow.add_node("TOOLS", execute_tool_calls)
    workflow.add_node("REFLECT", reflection_node)

    workflow.set_entry_point("REACT")

    workflow.add_conditional_edges(
        "REACT",
        router,
        {
            "TOOLS": "TOOLS",
            "REFLECT": "REFLECT",
            "REACT": "REACT",
            END: END,
        },
    )

    workflow.add_edge("TOOLS", "REACT")
    workflow.add_edge("REFLECT", END)

    graph = workflow.compile(store=store)

    # Save visualization
    try:
        GRAPH_DIR.mkdir(parents=True, exist_ok=True)
        graph_name = "investment_agent_optimized"
        path = GRAPH_DIR / f"{graph_name}.md"
        mermaid = graph.get_graph().draw_mermaid()
        path.write_text(f"```mermaid\n{mermaid}\n```")
        LOGGER.info(f"Graph saved → {path}")

        png_path = GRAPH_DIR / f"{graph_name}.png"
        png_bytes = graph.get_graph().draw_mermaid_png()
        png_path.write_bytes(png_bytes)
        LOGGER.info(f"PNG saved: {png_path}")
    except Exception as e:
        LOGGER.warning(f"Graph visualization failed: {e}")

    return graph


gr = build_graph()

if __name__ == "__main__":
    graph = build_graph()
    print("💬 Investment Advisor Ready! (Type 'exit' to quit)\n")

    while True:
        query = input("You: ").strip()
        if query.lower() in ["exit", "quit"]:
            break
        if not query:
            continue

        result = graph.invoke({"messages": [HumanMessage(content=query)]})
        print("\nAdvisor:")
        print(result.get("result") or "No final advice yet.")
        print("-" * 60)