"""
agents/investment_agent.py

Self-learning investment agent for NSE / NIFTY50
LangGraph workflow with persistent symbol memory
"""
import json
from typing import Dict, Any
from datetime import datetime
from zoneinfo import ZoneInfo
from langgraph.graph import StateGraph, END

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    ToolMessage
)

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from utils.config import GRAPH_DIR
from agents.llm import get_llm, embed_text
from tools.tool_registry import TOOLS
from concurrent.futures import ThreadPoolExecutor, as_completed

from utils.symbol_resolver import resolve_symbols
from utils.logger import LOGGER
from memory.store_memory import (
    load_symbol_memory,
    update_reflection,
    add_key_fact
)
from agents.agent_state import InvestmentAgentState as AgentState

IST = ZoneInfo("Asia/Kolkata")



# ─────────────────────────────────────────────
# Investment Agent
# ─────────────────────────────────────────────
class InvestmentAgent:
    def __init__(self, temperature: float = 0.0, max_attempts: int = 4):
        self.logger = LOGGER

        self.temperature = temperature
        self.max_attempts = max_attempts
        self.tools = TOOLS
        self.tool_map = {t.name: t for t in self.tools}
        self.llm = get_llm(temperature=self.temperature)
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        self.embed = embed_text
        self.prompt = ChatPromptTemplate.from_messages([(
"system",
"""
You are a professional Investment Advisor specializing in the Indian Stock Market.

Your goal is to provide well-researched, data-driven investment advice that helps users maximize returns while balancing risk.

Current IST time: {current_time}

Available Context:
News Sentiment: {sentiment}
Symbol Memory: {memory}

CRITICAL TOOL PIPELINE - Follow EXACTLY in this order:

1. **SYMBOL FIRST**: ALWAYS call 'lookup_stock_symbol' FIRST with company_name from user query. 
   - Do NOT guess symbols like TATAMOTORS, TMCV, etc.
   - Wait for tool result before any price/news calls.

2. **PRICE**: Call 'get_stock_info' with symbol from lookup result.
3. **NEWS**: Call 'get_stock_news' with confirmed symbol.
4. **ANALYSIS**: Only after ALL 3 tools succeed → provide Stock-Specific Advice.

RETRY LOGIC:
- If get_stock_info fails (error like "delisted", "no data", 404): 
  → IMMEDIATELY call 'lookup_stock_symbol' to get correct symbol → retry price.
- If lookup_stock_symbol fails: Say "Symbol lookup failed. Please provide exact NSE symbol."

Core Rules:
1. NEVER guess company symbols. lookup_stock_symbol is your ONLY source.
2. If ANY tool returns error/no data → say "Data unavailable for [price/symbol]."
3. Use historical memory only AFTER symbol confirmed.

Output ONLY when price tool succeeds with valid data:
[Your existing output format...]

If ANY step fails → "Data unavailable for price/symbol. Cannot analyze."
"""),
MessagesPlaceholder(variable_name="messages")
])


        # ─────────────────────────────
        # Chain
        # ─────────────────────────────
        self.chain = self.prompt | self.llm_with_tools

        # ─────────────────────────────
        # Graph
        # ─────────────────────────────
        self.graph = self._build_graph()
        self._save_graph_visualization(save_png=True)

    def _log_state(self, node, state):

        tools = list(state.tool_results.keys()) if state.tool_results else []

        self.logger.info(
            f"[{node}] "
            f"symbols={state.symbols} "
            f"tools={tools} "
            f"attempt={state.attempt_count}"
        )

    def _resolve_symbols_node(self, state: AgentState):
        self.logger.info("NODE → resolve_symbols")

        if state.symbols:
            self.logger.info(f"Symbols already present: {state.symbols}")
            return {}

        try:
            user_msg = next(
                m.content for m in reversed(state.messages)
                if isinstance(m, HumanMessage)
            )
        except StopIteration:
            self.logger.warning("No HumanMessage found in state.messages")
            return {}

        try:
            symbols = resolve_symbols(user_msg) or []
        except Exception:
            self.logger.exception("Symbol resolution failed")
            symbols = []

        self.logger.info(f"Resolved symbols → {symbols}")
        self._log_state("resolve_symbols", state)
        return {"symbols": symbols}

    def _load_memory_node(self, state: AgentState):
        self.logger.info("NODE → load_memory")

        symbols = state.symbols or []
        memory = {}

        for sym in symbols:
            try:
                memory[sym] = load_symbol_memory(sym)
                self.logger.info(f"Loaded memory → {sym}")
            except Exception:
                self.logger.exception(f"Memory load failed for {sym}")
                memory[sym] = {}

        self._log_state("load_memory", state)
        return {"memory": memory}

    def _agent_node(self, state: AgentState):
        self.logger.info("NODE → agent")

        symbols = state.symbols
        current_time = datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S IST")

        memory = json.dumps(state.memory or {}, indent=2)
        sentiment = json.dumps(state.sentiment or {}, indent=2)

        try:
            response = self.chain.invoke({
                "messages": state.messages,
                "memory": memory,
                "sentiment": sentiment,
                "current_time": current_time,
                "symbols": symbols
            })

            self.logger.info(
                f"LLM response received | tool_calls={bool(response.tool_calls)}"
            )

        except Exception:
            self.logger.exception("LLM invocation failed")

            response = AIMessage(
                content="Data unavailable due to system error."
            )
        self._log_state("agent", state)
        return {
            "messages": [response],
            "attempt_count": state.attempt_count + 1
        }

    def _tool_node(self, state: AgentState):
        self.logger.info("NODE → tools")

        outputs = []
        last_msg = state.messages[-1]

        if not getattr(last_msg, "tool_calls", None):
            self.logger.info("No tool calls found")
            return {"messages": outputs}

        tool_results = dict(state.tool_results or {})

        def execute_tool(tool_call):

            tool_name = tool_call["name"]
            raw_args = tool_call.get("args", {}) or {}

            self.logger.info(f"Executing tool → {tool_name} | args={raw_args}")

            if tool_name not in self.tool_map:

                self.logger.error(f"Tool not found → {tool_name}")

                return ToolMessage(
                    content=json.dumps(
                        {"error": f"Tool '{tool_name}' not found"}
                    ),
                    tool_call_id=tool_call["id"]
                ), None

            args = raw_args.copy()

            # Inject missing symbol
            if (
                tool_name != "lookup_stock_symbol"
                and "symbol" not in args
                and state.symbols
            ):
                args["symbol"] = state.symbols[0]

            # Inject query if needed
            if "query" not in args or not args.get("query"):
                if state.symbols:
                    args["query"] = f"{state.symbols[0]} stock news"

            try:

                result = self.tool_map[tool_name].invoke(args)
                self.logger.debug(
                    f"Tool result → {tool_name}: {json.dumps(result, default=str)[:800]}"
                )

            except Exception:

                self.logger.exception(
                    f"Tool execution failed → {tool_name}"
                )

                result = {
                    "status": "error",
                    "message": "Tool execution failed",
                    "tool": tool_name
                }

            message = ToolMessage(
                content=json.dumps(result, default=str),
                tool_call_id=tool_call["id"]
            )

            return message, (tool_name, args.get("symbol"), result)

        # Parallel execution
        max_workers = min(4, len(last_msg.tool_calls))

        with ThreadPoolExecutor(max_workers=max_workers) as executor:

            futures = [
                executor.submit(execute_tool, tc)
                for tc in last_msg.tool_calls
            ]

            for future in as_completed(futures):

                msg, result = future.result()

                outputs.append(msg)

                if result:

                    tool_name, symbol, data = result

                    if tool_name not in tool_results:
                        tool_results[tool_name] = {}

                    if isinstance(symbol, list):
                        for sym in symbol:
                            tool_results[tool_name][sym] = data
                    elif symbol:
                        tool_results[tool_name][symbol] = data
                    else:
                        tool_results[tool_name] = data

        self.logger.debug(
            f"Tool results snapshot → {json.dumps(tool_results, default=str)[:800]}"
        )
        self._log_state("tools", state)
        return {
            "messages": outputs,
            "tool_results": tool_results
        }     

    def _indicator_cache_node(self, state: AgentState):
        self.logger.info("NODE → indicator_cache")

        indicators = state.indicator_cache or {}
        tool_results = state.tool_results or {}

        for tool_name, result in tool_results.items():

            if tool_name not in ["compute_rsi", "compute_macd", "compute_sma"]:
                continue

            if not isinstance(result, dict):
                continue

            for symbol, data in result.items():

                if symbol not in indicators:
                    indicators[symbol] = {}

                indicators[symbol][tool_name] = data

                self.logger.info(
                    f"Indicator cached → {symbol} {tool_name}"
                )

        self._log_state("indicator_cache", state)
        return {"indicator_cache": indicators}

    def _news_sentiment_node(self, state: AgentState):

        self.logger.info("NODE → sentiment")

        news_data = (state.tool_results or {}).get("get_stock_news")

        if not news_data:
            self.logger.info("No news data found")
            return {}

        headlines = []

        for symbol_data in news_data.values():

            if not isinstance(symbol_data, dict):
                continue

            articles = symbol_data.get("articles", [])

            for article in articles:

                title = article.get("title")

                if isinstance(title, list):
                    title = " ".join(str(x) for x in title)

                if isinstance(title, str):
                    title = title.strip()

                if title:
                    headlines.append(title)

        self.logger.info(f"Headlines extracted → {len(headlines)}")

        if not headlines:
            return {}

        headlines = headlines[:10]

        headline_text = "\n".join(f"- {h}" for h in headlines)

        prompt = f"""
    Classify the overall sentiment of the following stock market headlines.

    Respond with ONLY ONE WORD:
    positive
    negative
    neutral

    Headlines:
    {headline_text}
    """

        try:

            response = self.llm.invoke(prompt)

            content = response.content

            if isinstance(content, list):
                content = " ".join(str(x) for x in content)

            sentiment_text = str(content).strip().lower()

            if "positive" in sentiment_text:
                sentiment = "positive"
            elif "negative" in sentiment_text:
                sentiment = "negative"
            else:
                sentiment = "neutral"

            self.logger.info(f"Sentiment result → {sentiment}")

        except Exception:

            self.logger.exception("Sentiment analysis failed")

            sentiment = "neutral"
        self._log_state("sentiment", state)
        return {
            "sentiment": {
                "overall": sentiment,
                "headlines": headlines
            }
        }

    def _reflection_node(self, state: AgentState):

        self._log_state("reflect", state)
        self.logger.info("NODE → reflection")

        if not state.symbols:
            self.logger.info("No symbols found → skipping reflection")
            return {}

        if not state.messages:
            self.logger.warning("No messages in state → skipping reflection")
            return {}

        symbol = state.symbols[0]
        last_msg = state.messages[-1]

        if not isinstance(last_msg, AIMessage):
            self.logger.info("Last message not AIMessage → skipping reflection")
            return {}

        reflection = f"""
    Agent Decision:

    {last_msg.content}

    Reflection Task:
    1. Track price movement after this signal.
    2. Evaluate if the recommendation was profitable.
    3. Store lessons for future signals.

    Timestamp:
    {datetime.now(IST)}
    """

        try:
            update_reflection(symbol, reflection)
            self.logger.info(f"Reflection stored → {symbol}")

        except Exception:
            self.logger.exception("Reflection storage failed")

        return {}

    def _memory_update_node(self, state: AgentState):

        self._log_state("memory_update", state)
        self.logger.info("NODE → memory_update")

        if not state.symbols:
            self.logger.info("No symbols found → skipping memory update")
            return {}

        if not state.messages:
            self.logger.warning("No messages found → skipping memory update")
            return {}

        symbol = state.symbols[0]
        last_msg = state.messages[-1]

        if not isinstance(last_msg, AIMessage):
            self.logger.info("Last message not AIMessage → skipping memory update")
            return {}

        text = (last_msg.content or "").lower()

        try:

            if "buy" in text:
                add_key_fact(symbol, "last_signal", "BUY")

            elif "sell" in text:
                add_key_fact(symbol, "last_signal", "SELL")

            elif "hold" in text:
                add_key_fact(symbol, "last_signal", "HOLD")

            self.logger.info(f"Memory updated → {symbol}")

        except Exception:
            self.logger.exception("Memory update failed")

        return {}

    def _should_continue(self, state: AgentState):

        if not state.messages:
            self.logger.warning("No messages in state → ending")
            return END

        last_msg = state.messages[-1]

        if not isinstance(last_msg, AIMessage):
            return END

        # If tools were requested
        if getattr(last_msg, "tool_calls", None):
            self.logger.info("Tool calls detected → routing to tools")
            return "tools"

        # Force tools if symbol exists but no data yet
        if state.symbols and not state.tool_results and state.attempt_count < 2:
            self.logger.info("Forcing tool execution")
            return "tools"

        # Reflection trigger
        if state.attempt_count >= self.max_attempts:
            self.logger.info("Max attempts reached → reflection")
            return "reflect"

        return END

    def _build_graph(self):

        workflow = StateGraph(AgentState)

        workflow.add_node("agent", self._agent_node)
        workflow.add_node("tools", self._tool_node)
        workflow.add_node("load_memory", self._load_memory_node)
        workflow.add_node("indicator_cache", self._indicator_cache_node)
        workflow.add_node("sentiment", self._news_sentiment_node)
        workflow.add_node("reflect", self._reflection_node)
        workflow.add_node("memory_update", self._memory_update_node)

        # Start with the agent
        workflow.set_entry_point("agent")

        workflow.add_conditional_edges(
            "agent",
            self._should_continue,
            {
                "tools": "tools",
                "reflect": "reflect",
                END: END
            }
        )

        workflow.add_edge("tools", "load_memory")
        workflow.add_edge("load_memory", "indicator_cache")
        workflow.add_edge("indicator_cache", "sentiment")
        workflow.add_edge("sentiment", "agent")

        workflow.add_edge("reflect", "memory_update")
        workflow.add_edge("memory_update", END)

        return workflow.compile()

    def _save_graph_visualization(self, save_png: bool = False):
        try:
            GRAPH_DIR.mkdir(parents=True, exist_ok=True)
            graph_name= f"investment_agent_v3"

            path = GRAPH_DIR / f"{graph_name}.md"
            mermaid = self.graph.get_graph().draw_mermaid()
            path.write_text(f"```mermaid\n{mermaid}\n```")
            self.logger.info(f"Graph saved → {path}")

            if save_png:
                png_path = GRAPH_DIR / f"{graph_name}.png"
                png_bytes = self.graph.get_graph().draw_mermaid_png()
                png_path.write_bytes(png_bytes)
                self.logger.info(f"PNG saved: {png_path}")

        except Exception as e:
            self.logger.warning(f"Graph visualization failed: {e}")

    def run(self, query: str) -> Dict[str, Any]:

        current_time = datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S IST")

        self.logger.info(f"Agent run → {query}")

        # Initialize full state safely
        state = AgentState(
            messages=[HumanMessage(content=query)],
            attempt_count=0,
            symbols=[],
            memory={},
            tool_results={},
            sentiment={},
            indicator_cache={}
        )

        try:

            result = self.graph.invoke(state)

        except Exception:

            self.logger.exception("Agent execution failed")

            return {
                "answer": "Agent execution failed due to a system error.",
                "messages": [],
                "current_time_ist": current_time
            }

        # Convert dict → AgentState safely
        try:
            result_state = AgentState(**result)
        except Exception:

            self.logger.exception("State reconstruction failed")

            return {
                "answer": "Internal agent state error.",
                "messages": [],
                "current_time_ist": current_time
            }

        final_messages = result_state.messages or []

        self.logger.info(f"Total messages generated → {len(final_messages)}")

        # Find final AI answer
        final_answer_msg = next(
            (
                m for m in reversed(final_messages)
                if isinstance(m, AIMessage)
                and not getattr(m, "tool_calls", None)
            ),
            None
        )

        if final_answer_msg:
            final_answer = final_answer_msg.content
        else:
            final_answer = "No final answer generated."

        self.logger.info("Agent run completed")

        return {
            "answer": final_answer,
            "messages": final_messages,
            "current_time_ist": current_time
        }

# ─────────────────────────────────────────────
# Singleton Agent
# ─────────────────────────────────────────────

agent = InvestmentAgent()


# ─────────────────────────────────────────────
# CLI Test
# ─────────────────────────────────────────────

if __name__ == "__main__":

    result = agent.run(
        "Compare Hindustan Aeronautics past year return with Nifty 50"
    )

    print("\nCurrent IST:", result["current_time_ist"])

    print("\nFinal Answer:\n")

    print(result["answer"])