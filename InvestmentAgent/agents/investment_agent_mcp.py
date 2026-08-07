import json
import os
import urllib.request
import urllib.error
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence
from zoneinfo import ZoneInfo

from langgraph.graph import StateGraph, END
from langgraph.store.memory import InMemoryStore
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    ToolMessage,
)

from agents.llm import get_llm, embed_text, EMBEDDING_DIM
from agents.agent_state import InvestmentAgentState
from agents.system_prompt import REACT_PROMPT
from agents.helper import (
    format_tool_history,
    compact_tool_result,
    format_state_for_prompt,
    clean_final_advice,
    extract_python_dict,
)
from chat.response_formatter import format_final_response
from tools.tool_registry import TOOLS
from utils.logger import LOGGER
from utils.config import GRAPH_DIR
from utils.db_connect import get_connection
from agents.store_memory import load_symbol_memory, save_symbol_memory

IST = ZoneInfo("Asia/Kolkata")
MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://127.0.0.1:8000")
MCP_EXECUTE_URL = f"{MCP_SERVER_URL}/mcp/execute"


class MCPToolWrapper:
    """Wrap a local tool definition and execute it via the MCP endpoint."""

    def __init__(self, tool: Any):
        self._tool = tool
        self.name = tool.name
        self.description = getattr(tool, "description", "") or ""
        self.args_schema = getattr(tool, "args_schema", None)
        self.return_direct = getattr(tool, "return_direct", False)

    def invoke(self, args: Dict[str, Any]) -> Any:
        return execute_mcp_tool(self.name, args)

    def __getattr__(self, item: str) -> Any:
        return getattr(self._tool, item)

    def __call__(self, *args, **kwargs) -> Any:
        """Allow the wrapper instance to be used as a callable for bind_tools.

        Accepts either a single dict positional argument or keyword args.
        """
        if args and isinstance(args[0], dict) and not kwargs:
            return self.invoke(args[0])
        return self.invoke(kwargs or {})


def execute_mcp_tool(tool_name: str, args: Dict[str, Any]) -> Any:
    """Invoke the MCP execute endpoint for a single tool."""
    payload = json.dumps({"tool_name": tool_name, "args": args}).encode("utf-8")
    request = urllib.request.Request(
        MCP_EXECUTE_URL,
        data=payload,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )

    # Retries for transient failures (short per-attempt timeout + backoff)
    attempts = 2
    per_attempt_timeout = 30
    for attempt in range(1, attempts + 1):
        try:
            LOGGER.debug("MCP execute attempt %s for tool %s", attempt, tool_name)
            with urllib.request.urlopen(request, timeout=per_attempt_timeout) as response:
                body = response.read().decode("utf-8")
                data = json.loads(body)
                break
        except urllib.error.HTTPError as exc:
            message = exc.read().decode("utf-8")
            logger_msg = f"MCP tool execution HTTP {exc.code}: {message}"
            LOGGER.error(logger_msg)
            return {"status": "error", "message": logger_msg}
        except Exception as exc:
            LOGGER.warning("MCP tool execution attempt %s failed: %s", attempt, exc)
            data = {"status": "error", "message": str(exc)}
            if attempt < attempts:
                backoff = 2 ** attempt
                LOGGER.info("Retrying MCP execute after %s seconds backoff", backoff)
                try:
                    time.sleep(backoff)
                except Exception:
                    pass
                continue
            LOGGER.error("MCP tool execution failed after %s attempts: %s", attempts, exc)
            return {"status": "error", "message": str(exc), "timed_out": True}

    if isinstance(data, dict):
        if data.get("status") == "error":
            return {"status": "error", "message": data.get("error") or data.get("result") or data.get("message")}
        return data.get("result", data)

    return data


MCP_TOOLS = [MCPToolWrapper(tool) for tool in TOOLS]
MCP_TOOL_MAP = {tool.name: tool for tool in MCP_TOOLS}

# Create lightweight callables for LangChain bind_tools (functions with __name__)
def make_mcp_callable(tool_name: str):
    def _call(**kwargs):
        # LangChain may pass named args; consolidate into an args dict
        if not kwargs:
            args = {}
        elif "args" in kwargs and isinstance(kwargs["args"], dict) and len(kwargs) == 1:
            args = kwargs["args"]
        else:
            args = kwargs

        try:
            return execute_mcp_tool(tool_name, args or {})
        except Exception as exc:
            LOGGER.exception("MCP tool call failed: %s %s", tool_name, exc)
            return {"status": "error", "message": str(exc)}

    # set function metadata for langchain conversion
    _call.__name__ = tool_name
    _call.__doc__ = f"Remote MCP tool wrapper for {tool_name}"
    return _call


MCP_CALLABLES = [make_mcp_callable(t.name) for t in TOOLS]

LLM = get_llm(temperature=0.0)
if MCP_CALLABLES:
    LLM_WITH_TOOLS = LLM.bind_tools(MCP_CALLABLES, tool_choice="required")
else:
    LLM_WITH_TOOLS = LLM.bind_tools([], tool_choice="required")

TOOL_MAP = MCP_TOOL_MAP

store = InMemoryStore(index={"embed": embed_text, "dims": EMBEDDING_DIM})
db_conn = get_connection(LOGGER)

MAX_MSG_HISTORY = 15
MAX_MEMORY = 10
MAX_ATTEMPTS = 7

LLM_CHAIN = REACT_PROMPT | LLM_WITH_TOOLS


async def reasoning_node(state: InvestmentAgentState):
    LOGGER.info(f"NODE → REACT (attempt {state.attempt_count + 1})")

    state.current_datetime = datetime.now(IST)
    new_attempt = state.attempt_count + 1

    input_dict = format_state_for_prompt(state)
    input_dict["messages"] = state.messages[-MAX_MSG_HISTORY:]

    response = LLM_CHAIN.invoke(input_dict)

    LOGGER.info(f"LLM RESPONSE → {str(response.content)[:200]}...")

    updates = {
        "attempt_count": new_attempt,
        "messages": state.messages + [response],
    }

    tool_calls = getattr(response, "tool_calls", None) or []
    if tool_calls:
        LOGGER.info(f"LLM requested {len(tool_calls)} tool(s)")
        updates["messages"] = updates["messages"][-MAX_MSG_HISTORY:]
        return updates

    content = (response.content or "").strip()
    parsed = extract_python_dict(content)

    def is_valid_final_answer(result: Any) -> bool:
        required_keys = ["company", "ticker", "price", "signal", "confidence", "reasoning"]
        return isinstance(result, dict) and all(k in result for k in required_keys)

    if parsed and is_valid_final_answer(parsed):
        prices = state.prices or {}
        symbols = state.symbols or {}

        missing_prices = [
            (company, symbol)
            for company, symbol in symbols.items()
            if symbol not in prices
        ]

        recent_msgs = [m.content for m in state.messages[-5:] if isinstance(m, HumanMessage)]
        if missing_prices and any("STOP." in msg for msg in recent_msgs):
            LOGGER.warning("Loop detected - missing price after retries")
            return {
                "attempt_count": new_attempt,
                "result": "ERROR: Missing price data.",
                "messages": state.messages + [AIMessage(content="Blocked due to missing data")],
            }

        if missing_prices:
            company, symbol = missing_prices[0]
            LOGGER.warning(f"Forcing tool call → {symbol}")
            reminder = HumanMessage(content=f"Call tool: get_stock_info(symbol='{symbol}')")
            return {
                "attempt_count": new_attempt,
                "messages": (state.messages + [response, reminder])[-MAX_MSG_HISTORY:],
            }

        LOGGER.info("Final advice (CodeAgent) generated")
        memory_entry = f"[{state.current_datetime}] {parsed}"

        return {
            "attempt_count": new_attempt,
            "messages": (state.messages + [response])[-MAX_MSG_HISTORY:],
            "result": parsed,
            "memory": ((state.memory or []) + [memory_entry])[-MAX_MEMORY:],
        }

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

        # --- Ensure symbol exists for non-lookup tools ---
        if name != "lookup_stock_symbol":
            # Inject symbol from state if available
            if "symbol" not in args or not args.get("symbol"):
                if new_symbols:
                    # pick the first known symbol
                    args["symbol"] = next(iter(new_symbols.values()))
                elif company_name:
                    # try remote lookup via MCP tool if available
                    try:
                        lookup_tool = TOOL_MAP.get("lookup_stock_symbol")
                        if lookup_tool:
                            lookup_res = lookup_tool.invoke({"company": company_name[0]})
                            if isinstance(lookup_res, list) and lookup_res:
                                chosen = lookup_res[0]
                                sym = chosen.get("symbol") or chosen.get("ticker")
                                if sym:
                                    args["symbol"] = sym
                                    # record lookup into tool history/messages
                                    tool_history.append({
                                        "step": step,
                                        "tool": "lookup_stock_symbol",
                                        "args": {"company": company_name[0]},
                                        "result": lookup_res,
                                        "time": datetime.now(IST).isoformat(),
                                    })
                                    new_messages.append(
                                        ToolMessage(
                                            tool_call_id=f"lookup_auto_{step}",
                                            name="lookup_stock_symbol",
                                            content=compact_tool_result(lookup_res),
                                        )
                                    )
                                    step += 1
                    except Exception as e:
                        LOGGER.warning("Auto symbol lookup failed: %s", e)

        # --- Normalize symbol suffix to .NS when appropriate ---
        if isinstance(args, dict) and "symbol" in args and isinstance(args.get("symbol"), str):
            symv = args["symbol"].strip()
            if symv and not symv.endswith(".NS") and len(symv) <= 10:
                args["symbol"] = symv if "." in symv else f"{symv}.NS"

        # --- Normalize get_price_history period inputs ---
        if name == "get_price_history" and isinstance(args, dict):
            period = args.get("period") or args.get("Period") or args.get("p")
            if isinstance(period, str):
                p = period.strip().lower()
                if p in {"d", "day"}:
                    args["period"] = "1d"
                elif p == "1d":
                    args["period"] = "1d"
                else:
                    args["period"] = p

        # --- Skip if identical previous call failed ---
        prev_failure = None
        for h in reversed(tool_history):
            try:
                if h.get("tool") == name and h.get("args") == args:
                    r = h.get("result")
                    if isinstance(r, dict) and (r.get("status") == "error" or r.get("error") is not None):
                        prev_failure = r
                        break
            except Exception:
                continue

        if prev_failure is not None:
            LOGGER.warning("Skipping repeated failing tool call %s with args %s", name, args)
            result = {"error": "previous failure", "detail": prev_failure}
        else:
            try:
                result = TOOL_MAP[name].invoke(args)
            except Exception as e:
                LOGGER.error(f"Tool {name} failed: {e}")
                result = {"error": str(e)}

        # --- If tool errored, attempt fallback to get_stock_info when possible ---
        if isinstance(result, dict) and (result.get("status") == "error" or result.get("error") is not None):
            if name != "get_stock_info" and "symbol" in args:
                try:
                    fallback = TOOL_MAP.get("get_stock_info")
                    if fallback:
                        fb_res = fallback.invoke({"symbol": args.get("symbol")})
                        tool_history.append({
                            "step": step,
                            "tool": "get_stock_info",
                            "args": {"symbol": args.get("symbol")},
                            "result": fb_res,
                            "time": datetime.now(IST).isoformat(),
                        })
                        new_messages.append(
                            ToolMessage(
                                tool_call_id=f"fallback_info_{step}",
                                name="get_stock_info",
                                content=compact_tool_result(fb_res),
                            )
                        )
                        step += 1
                        result = fb_res
                except Exception as e:
                    LOGGER.warning("Fallback get_stock_info failed: %s", e)

        if name == "lookup_stock_symbol" and isinstance(result, list):
            for sym in result:
                company = sym.get("company_name")
                ticker = sym.get("symbol")
                if company and ticker:
                    if company not in company_name:
                        company_name.append(company)
                    new_symbols[company] = ticker

        if name == "get_stock_info" and isinstance(result, dict):
            symbol = result.get("symbol", "")
            if symbol and result.get("status") == "success":
                data = result.get("data", {})
                new_prices[symbol] = data
                LOGGER.info(f"Stored price for {symbol}")

        tool_history.append({
            "step": step,
            "tool": name,
            "args": args,
            "result": result,
            "time": datetime.now(IST).isoformat(),
        })
        step += 1

        new_messages.append(
            ToolMessage(
                tool_call_id=call_id,
                name=name,
                content=compact_tool_result(result),
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

            quality = parsed.get("quality", "unknown")
            confidence = parsed.get("true_confidence")
            if not isinstance(confidence, int):
                confidence = 50

            bias = parsed.get("bias")
            if bias not in ["bullish", "bearish", "sideways"]:
                bias = "unknown"

            mem.setdefault("key_facts", {})
            mem["key_facts"]["bias"] = bias
            mem["key_facts"]["true_confidence"] = confidence

            mem.setdefault("notes", []).append({
                "timestamp": datetime.now(IST).isoformat(),
                "note": parsed.get("notes", ""),
            })

            mem.setdefault("reflections", []).append({
                "timestamp": datetime.now(IST).isoformat(),
                "quality": quality,
                "confidence": confidence,
                "data": parsed,
            })

            save_symbol_memory(sym, mem)
            reflections.append((sym, quality))

        LOGGER.info(f"Reflections stored: {reflections}")
    except Exception as e:
        LOGGER.warning(f"Reflection failed: {e}")

    return {}


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

    try:
        GRAPH_DIR.mkdir(parents=True, exist_ok=True)
        graph_name = "investment_agent_mcp"
        path = GRAPH_DIR / f"{graph_name}.md"
        mermaid = graph.get_graph().draw_mermaid()
        path.write_text("```mermaid\n" + mermaid + "\n```")
        LOGGER.info(f"Graph saved → {path}")

        png_path = GRAPH_DIR / f"{graph_name}.png"
        png_bytes = graph.get_graph().draw_mermaid_png()
        png_path.write_bytes(png_bytes)
        LOGGER.info(f"PNG saved: {png_path}")
    except Exception as e:
        LOGGER.warning(f"Graph visualization failed: {e}")

    return graph


# Lightweight compiled graph cache for quick runs from API
GRAPH: Optional[Any] = None


async def get_graph():
    global GRAPH
    if GRAPH is None:
        GRAPH = await build_graph()
    return GRAPH


async def run_mcp_agent(message: str) -> Dict[str, Any]:
    """Run the MCP-backed agent for a single user message and return formatted response."""

    graph = await get_graph()

    state = InvestmentAgentState(
        messages=[HumanMessage(content=message)],
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

    final_state_dict = await graph.ainvoke(state)

    if isinstance(final_state_dict, dict):
        final_state = InvestmentAgentState(**final_state_dict)
    else:
        final_state = final_state_dict

    answer = format_final_response(final_state)

    memory_summary = (
        f"symbols={len(final_state.symbols)}, news={len(final_state.news)}, memory={len(final_state.memory)}"
    )

    return {"answer": answer, "memory_summary": memory_summary, "thread_id": "mcp_agent"}


if __name__ == "__main__":
    import asyncio

    graph = asyncio.run(build_graph())

    print("💬 MCP-backed Investment Advisor Ready! (Type 'exit' to quit)")

    while True:
        query = input("You: ").strip()
        if query.lower() in ("exit", "quit"):
            break
        if not query:
            continue

        result = asyncio.run(graph.ainvoke({"messages": [HumanMessage(content=query)]}))
        print("\nAdvisor:")
        print(result.get("result") or "No final advice yet.")
        print("-" * 60)
