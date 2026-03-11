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

Current IST time:
{current_time}

Available Context

News Sentiment:
{sentiment}

Symbol Memory:
{memory}


Core Rules

1. NEVER guess stock prices, company symbols, or financial data.
2. ALWAYS use available tools to fetch:
   • stock price
   • company symbol
   • latest market news
3. Fetch the latest market data BEFORE performing any analysis.
4. If the user mentions a company name, resolve its symbol using tools first.
5. Use historical memory and reflections when available to improve decision quality.
6. Do NOT hallucinate tools or data.
7. If data is unavailable, clearly say: "Data unavailable".
8. Your advice affects real financial decisions — be precise and cautious.


Analysis Requirements

Before giving a signal you must verify:

• Current stock price
• Price timestamp
• Latest relevant news
• Sentiment context
• Risk factors


Output Format:

Stock-Specific Advice

Company: <Company Name>  
Symbol: <Stock Symbol>  
Current Price: ₹<price>  
Price Time: <datetime IST>

Signal: BUY / HOLD / SELL / NEUTRAL  
Confidence: <0-100>/100


Explanation:
Provide clear point-wise reasoning:
1. Current price analysis
2. Relevant news impact
3. Sentiment interpretation
4. Risk considerations
5. Short-term vs long-term outlook

Important
• Do NOT fabricate financial information.
• Always rely on tool outputs.
• If multiple signals conflict, explain the uncertainty.
"""
        ),
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
        self._save_graph_visualization(save_png=False)

    # ─────────────────────────────────────────────
    # Symbol Resolve Node
    # ─────────────────────────────────────────────
    def _resolve_symbols_node(self, state: AgentState):

        if state.symbols:
            return {}
        user_msg = next(
            m.content for m in reversed(state.messages)
            if isinstance(m, HumanMessage)
        )
        symbols = resolve_symbols(user_msg)

        if symbols:
            self.logger.info(f"Resolved symbol → {symbols}")

        return {"symbols": symbols}

    # ─────────────────────────────────────────────
    # Memory Load Node
    # ─────────────────────────────────────────────
    def _load_memory_node(self, state: AgentState):

        symbols = state.symbols
        memory = {}

        for sym in symbols:
            memory[sym] = load_symbol_memory(sym)
            self.logger.info(f"Loaded memory for {sym}")

        return {"memory": memory}
    # ─────────────────────────────────────────────
    # Agent Node
    # ─────────────────────────────────────────────
    def _agent_node(self, state: AgentState):

        symbols = state.symbols
        current_time = datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S IST")
        memory = json.dumps(state.memory, indent=2)
        sentiment = json.dumps(state.sentiment, indent=2)

        response = self.chain.invoke({
            "messages": state.messages,
            "memory": memory,
            "sentiment": sentiment,
            "current_time": current_time,
            "symbols": symbols
        })

        return {
            "messages": [response],
            "attempt_count": state.attempt_count + 1
        }

    def _tool_node(self, state: AgentState):

        outputs = []
        last_msg = state.messages[-1]

        if not getattr(last_msg, "tool_calls", None):
            return {"messages": outputs}

        # preserve previous results
        tool_results = dict(state.tool_results)

        def execute_tool(tool_call):

            tool_name = tool_call["name"]
            raw_args = tool_call.get("args", {}) or {}

            if tool_name not in self.tool_map:
                return ToolMessage(
                    content=json.dumps({"error": f"Tool '{tool_name}' not found"}),
                    tool_call_id=tool_call["id"]
                ), None

            args = raw_args.copy()

            # Inject missing arguments safely
            if "symbol" not in args and state.symbols:
                args["symbol"] = state.symbols[0]

            if "query" not in args or not args.get("query"):
                if state.symbols:
                    args["query"] = f"{state.symbols[0]} stock news"

            try:
                result = self.tool_map[tool_name].invoke(args)

            except Exception as e:

                result = {
                    "status": "error",
                    "message": str(e),
                    "tool": tool_name
                }

            message = ToolMessage(
                content=json.dumps(result, default=str),
                tool_call_id=tool_call["id"]
            )

            return message, (tool_name, args.get("symbol"), result)

        # PARALLEL EXECUTION
        #with ThreadPoolExecutor(max_workers=len(last_msg.tool_calls)) as executor:
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

                    # If symbol is a list → store result per symbol
                    if isinstance(symbol, list):
                        for sym in symbol:
                            tool_results[tool_name][sym] = data
                    elif symbol:
                        tool_results[tool_name][symbol] = data
                    else:
                        tool_results[tool_name] = data

        return {
            "messages": outputs,
            "tool_results": tool_results
        }        

    def _indicator_cache_node(self, state: AgentState):

        indicators = state.indicator_cache
        tool_results = state.tool_results

        for tool_name, result in tool_results.items():
            if tool_name in ["compute_rsi", "compute_macd", "compute_sma"]:
                for symbol, data in result.items():

                    if symbol not in indicators:
                        indicators[symbol] = {}

                    indicators[symbol][tool_name] = data

                    self.logger.info(f"Indicator cached → {symbol} {tool_name}")

        return {"indicator_cache": indicators}

    def _news_sentiment_node(self, state: AgentState):
        news_data = state.tool_results.get("get_stock_news")
        if not news_data:
            return {}
        headlines = []

        for symbol_data in news_data.values():
            for article in symbol_data.get("articles", []):
                title = article.get("title")
                if title:
                    headlines.append(title)

        if not headlines:
            return {}
        prompt = f"""
        Classify the sentiment of these headlines.

        Return ONLY one word:
        positive
        negative
        neutral

        Headlines:
        {headlines}
        """
        sentiment = (
            self.llm.invoke(prompt)
            .content
            .strip()
            .lower()
        )

        if sentiment not in ["positive", "negative", "neutral"]:
            sentiment = "neutral"

        self.logger.info(f"News sentiment → {sentiment}")

        return {
            "sentiment": {
                "overall": sentiment,
                "headlines": headlines
            }
        }
    # ─────────────────────────────────────────────
    # Reflection Node
    # ─────────────────────────────────────────────
    def _reflection_node(self, state: AgentState):

        if not state.symbols:
            return {}

        symbol = state.symbols[0]

        last_msg = state.messages[-1]

        if not isinstance(last_msg, AIMessage):
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
        update_reflection(symbol, reflection)
        self.logger.info(f"Reflection stored for {symbol}")

        return {}

    # ─────────────────────────────────────────────
    # Memory Update Node
    # ─────────────────────────────────────────────
    def _memory_update_node(self, state: AgentState):

        if not state.symbols:
            return {}

        symbol = state.symbols[0]

        last_msg = state.messages[-1]

        if not isinstance(last_msg, AIMessage):
            return {}

        text = last_msg.content.lower()
        if "buy" in text:
            add_key_fact(symbol, "last_signal", "BUY")

        elif "sell" in text:
            add_key_fact(symbol, "last_signal", "SELL")

        elif "hold" in text:
            add_key_fact(symbol, "last_signal", "HOLD")

        self.logger.info(f"Memory updated for {symbol}")

        return {}

    # ─────────────────────────────────────────────
    # Continue Logic
    # ─────────────────────────────────────────────
    def _should_continue(self, state: AgentState):

        last_msg = state.messages[-1]

        if not isinstance(last_msg, AIMessage):
            return END

        # FORCE tools if symbols exist but no tool results yet
        if state.symbols and not state.tool_results and state.attempt_count < 2:
            return "tools"
        if last_msg.tool_calls:
            return "tools"

        if state.attempt_count >= self.max_attempts:
            return "reflect"

        return END

    # ─────────────────────────────────────────────
    # Build Graph
    # ─────────────────────────────────────────────
    def _build_graph(self):

        workflow = StateGraph(AgentState)

        workflow.add_node("resolve_symbols", self._resolve_symbols_node)
        workflow.add_node("load_memory", self._load_memory_node)
        workflow.add_node("agent", self._agent_node)
        workflow.add_node("tools", self._tool_node)
        workflow.add_node("indicator_cache", self._indicator_cache_node)
        workflow.add_node("sentiment", self._news_sentiment_node)
        workflow.add_node("reflect", self._reflection_node)
        workflow.add_node("memory_update", self._memory_update_node)

        workflow.set_entry_point("resolve_symbols")
        workflow.add_edge("resolve_symbols", "load_memory")
        workflow.add_edge("load_memory", "agent")
        workflow.add_conditional_edges(
            "agent",
            self._should_continue,
            {
                "tools": "tools",
                "reflect": "reflect",
                END: END
            }
        )
        # tools → indicator cache
        workflow.add_edge("tools", "indicator_cache")
        # indicator → sentiment
        workflow.add_edge("indicator_cache", "sentiment")
        # sentiment → agent
        workflow.add_edge("sentiment", "agent")
        workflow.add_edge("reflect", "memory_update")
        workflow.add_edge("memory_update", END)

        return workflow.compile()

    # ─────────────────────────────────────────────
    # Graph Visualization
    # ─────────────────────────────────────────────

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

    # ─────────────────────────────────────────────
    # Run Agent
    # ─────────────────────────────────────────────
    def run(self, query: str) -> Dict[str, Any]:

        current_time = datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S IST")
        self.logger.info(f"Agent run → {query}")

        state = AgentState(
            messages=[HumanMessage(content=query)],
            attempt_count=0,
            symbols=[],
            memory={}
        )

        result = self.graph.invoke(state)

        # convert dict → AgentState
        result_state = AgentState(**result)

        final_messages = result_state.messages

        final_answer_msg = next(
            (
                m for m in reversed(final_messages)
                if isinstance(m, AIMessage) and not m.tool_calls
            ),
            None
        )

        final_answer = (
            final_answer_msg.content
            if final_answer_msg
            else "No final answer generated."
        )

        return {
            "answer": final_answer,
            "messages": final_messages,
            "current_time_ist": current_time
        }

# ─────────────────────────────────────────────
# Singleton Agent
# ─────────────────────────────────────────────

agent = InvestmentAgent()


# # ─────────────────────────────────────────────
# # CLI Test
# # ─────────────────────────────────────────────

# if __name__ == "__main__":

#     result = agent.run(
#         "Compare Hindustan Aeronautics past year return with Nifty 50"
#     )

#     print("\nCurrent IST:", result["current_time_ist"])

#     print("\nFinal Answer:\n")

#     print(result["answer"])