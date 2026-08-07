import json,re, asyncio
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

# Local imports
from agents.llm import get_llm, embed_text, EMBEDDING_DIM
from agents.agent_state import InvestmentAgentState
from agents.system_prompt import REACT_PROMPT
from agents.helper import format_tool_history, compact_tool_result, format_state_for_prompt, clean_final_advice, extract_python_dict
from tools.tool_registry import TOOLS
from utils.logger import LOGGER
from utils.config import GRAPH_DIR
from utils.db_connect import get_connection
from agents.store_memory import (
    load_symbol_memory,
    save_symbol_memory
)


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
MAX_MSG_HISTORY = 15         # Prevent context overflow
MAX_MEMORY = 10
MAX_ATTEMPTS = 7

LLM_CHAIN = REACT_PROMPT | LLM_WITH_TOOLS


# =============================================================
# Nodes
# =============================================================
async def reasoning_node(state: InvestmentAgentState):
    LOGGER.info(f"NODE → REACT (attempt {state.attempt_count + 1})")

    state.current_datetime = datetime.now(IST)
    new_attempt = state.attempt_count + 1

    input_dict = format_state_for_prompt(state)
    input_dict["messages"] = state.messages[-MAX_MSG_HISTORY:]

    response = LLM_CHAIN.invoke(input_dict)

    LOGGER.info(
        f"LLM RESPONSE → {str(response.content)[:200]}..."
        )

    updates = {
        "attempt_count": new_attempt,
        "messages": state.messages + [response],
    }

    # -----------------------------
    # TOOL CALL HANDLING
    # -----------------------------
    tool_calls = getattr(response, "tool_calls", None) or []
    if tool_calls:
        LOGGER.info(f"LLM requested {len(tool_calls)} tool(s)")
        updates["messages"] = updates["messages"][-MAX_MSG_HISTORY:]
        return updates

    # -----------------------------
    # CODE OUTPUT PARSING
    # -----------------------------
    content = (response.content or "").strip()
    parsed = extract_python_dict(content)
    def is_valid_final_answer(result):
        required_keys = ["company", "ticker", "price", "signal", "confidence","reasoning"]
        return all(k in result for k in required_keys)

    if parsed and is_valid_final_answer(parsed):
        prices = state.prices or {}
        symbols = state.symbols or {}

        missing_prices = [
            (company, symbol)
            for company, symbol in symbols.items()
            if symbol not in prices
        ]

        # Loop protection
        recent_msgs = [m.content for m in state.messages[-5:] if isinstance(m, HumanMessage)]
        if missing_prices and any("STOP." in msg for msg in recent_msgs):
            LOGGER.warning("Loop detected - missing price after retries")
            return {
                "attempt_count": new_attempt,
                "result": "ERROR: Missing price data.",
                "messages": state.messages + [AIMessage(content="Blocked due to missing data")],
            }

        # Force tool if missing price
        if missing_prices:
            company, symbol = missing_prices[0]
            LOGGER.warning(f"Forcing tool call → {symbol}")

            reminder = HumanMessage(
                content=f"Call tool: get_stock_info(symbol='{symbol}')"
            )

            return {
                "attempt_count": new_attempt,
                "messages": (state.messages + [response, reminder])[-MAX_MSG_HISTORY:]
            }

        # ✅ FINAL RESULT
        LOGGER.info("Final advice (CodeAgent) generated")

        memory_entry = f"[{state.current_datetime}] {parsed}"

        return {
            "attempt_count": new_attempt,
            "messages": (state.messages + [response])[-MAX_MSG_HISTORY:],
            "result": parsed,   # structured output
            "memory": ((state.memory or []) + [memory_entry])[-MAX_MEMORY:],
        }

    # fallback loop
    updates["messages"] = updates["messages"][-MAX_MSG_HISTORY:]
    return updates

async def execute_tool_calls(state: InvestmentAgentState):
    last_msg = state.messages[-1]
    tool_calls = getattr(last_msg, "tool_calls", []) or []

    LOGGER.info(f"Executing {len(tool_calls)} tool calls")

    new_messages = []
    new_symbols = dict(state.symbols)
    new_prices = dict(state.prices)
    company_name = list(state.company_name or [])
    tool_history = list(state.tool_history or [])

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

async def reflection_node(state: InvestmentAgentState):
    if not state.result or not state.symbols:
        return {}

    LOGGER.info("NODE → REFLECT (CodeAgent self-learning)")

    try:
        symbols = list(state.symbols.values())
        content = state.result

        reflections = []

        for sym in symbols:

            mem = load_symbol_memory(sym)

            past_bias = mem.get("key_facts", {}).get("bias", "unknown")
            last_signal = mem.get("last_signals", {})

            # -----------------------------
            # CODEAGENT PROMPT
            # -----------------------------
            reflection_prompt = f"""
You are a self-improving investment agent.

=== CURRENT DECISION ===
{content}

=== PAST MEMORY ===
Bias: {past_bias}
Last Signal: {last_signal}

You MUST return ONLY a Python dictionary.

STRICT RULES:
- DO NOT write functions
- DO NOT write code blocks
- DO NOT use ```python
- DO NOT explain anything
- DO NOT modify input
- ONLY return a dictionary

VALID FORMAT:

return {{
    "quality": "strong" | "average" | "weak",
    "true_confidence": 0-100,
    "bias": "bullish" | "bearish" | "sideways",
    "risks": ["risk1", "risk2"],
    "improvements": ["improve1"],
    "notes": "short text"
}}
"""

            llm_response = LLM.invoke(reflection_prompt)

            parsed = extract_python_dict(llm_response.content)

            if not parsed:
                LOGGER.warning(f"Reflection parse failed → {llm_response.content}")
                continue

            # -----------------------------
            # VALIDATION
            # -----------------------------
            quality = parsed.get("quality", "unknown")

            confidence = parsed.get("true_confidence")
            if not isinstance(confidence, int):
                confidence = 50

            bias = parsed.get("bias")
            if bias not in ["bullish", "bearish", "sideways"]:
                bias = "unknown"

            # -----------------------------
            # ATOMIC MEMORY UPDATE
            # -----------------------------
            mem.setdefault("key_facts", {})
            mem["key_facts"]["bias"] = bias
            mem["key_facts"]["true_confidence"] = confidence

            mem.setdefault("notes", []).append({
                "timestamp": datetime.now(IST).isoformat(),
                "note": parsed.get("notes", "")
            })

            mem.setdefault("reflections", []).append({
                "timestamp": datetime.now(IST).isoformat(),
                "quality": quality,
                "confidence": confidence,
                "data": parsed
            })

            save_symbol_memory(sym, mem)

            reflections.append((sym, quality))

        LOGGER.info(f"Reflections stored: {reflections}")

    except Exception as e:
        LOGGER.warning(f"Reflection failed: {e}")

    return {}

# =============================================================
# Router
# =============================================================
async def router(state: InvestmentAgentState):
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
async def build_graph():
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


class LocalInvestmentAgent:
    """Wrapper to execute the compiled investment agent graph synchronously."""

    def __init__(self):
        self.graph = asyncio.run(build_graph())

    def run(self, query: str) -> Dict[str, Any]:
        state = InvestmentAgentState(
            messages=[HumanMessage(content=query)],
            company_name=[],
            symbols={},
            prices={},
            news={},
            memory=[],
            tool_history=[],
            result=None,
            attempt_count=0,
            current_datetime=datetime.now(IST),
        )

        try:
            output = asyncio.run(self.graph.ainvoke(state))
        except Exception:
            LOGGER.exception("Agent execution failed")
            return {
                "answer": "Agent execution failed due to a system error.",
                "messages": [],
                "current_time_ist": datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S IST"),
            }

        if isinstance(output, dict):
            try:
                final_state = InvestmentAgentState(**output)
            except Exception:
                LOGGER.exception("State reconstruction failed")
                return {
                    "answer": "Internal agent state error.",
                    "messages": [],
                    "current_time_ist": datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S IST"),
                }
        else:
            final_state = output

        final_answer_msg = next(
            (
                m for m in reversed(final_state.messages)
                if isinstance(m, AIMessage) and not getattr(m, "tool_calls", None)
            ),
            None,
        )

        answer = (
            final_answer_msg.content
            if final_answer_msg
            else final_state.result or "No final answer generated."
        )

        return {
            "answer": answer,
            "messages": final_state.messages,
            "current_time_ist": datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S IST"),
        }


agent = LocalInvestmentAgent()


if __name__ == "__main__":
    graph = asyncio.run(build_graph())

    print("💬 Investment Advisor Ready! (Type 'exit' to quit)\n")

    while True:
        query = input("You: ").strip()

        if query.lower() in ("exit", "quit"):
            break

        if not query:
            continue

        result = asyncio.run(
            graph.ainvoke(
                {"messages": [HumanMessage(content=query)]}
            )
        )

        print("\nAdvisor:")
        print(result.get("result") or "No final advice yet.")
        print("-" * 60)