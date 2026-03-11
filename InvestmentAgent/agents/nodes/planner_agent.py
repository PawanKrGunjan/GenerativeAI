# agents/planner_agent.py

import json
import re

from agents.llm import get_llm
from tools.tool_registry import SEARCH_TOOLS, MARKET_TOOLS
from agents.agent_state import InvestmentState
from utils.symbol_resolver import resolve_symbols
from utils.logger import LOGGER


class PlannerAgent:

    def __init__(self, InvestmentState):

        self.llm = get_llm()

        # Planner should only select DATA tools
        self.tools = SEARCH_TOOLS + MARKET_TOOLS
        self.tool_map = {tool.name: tool for tool in self.tools}

    def _resolve_symbols(self):
        if state.get("symbols"):
            return {}
        user_msg = state["messages"][0].content
        symbols = resolve_symbols(user_msg)
        if symbols:
            self.logger.info(f"Resolved symbol → {symbols}")
        return {"symbols": symbols}
    
    def run(self, state):

        LOGGER.info("Planner started")

        query = state.get("query", "")
        tool_names = list(self.tool_map.keys())

        prompt = f"""
You are a financial data planning AI.

User Query:
{query}

Available Data Tools:
{tool_names}

Your job is to decide if market data must be fetched before analysis.

Rules:

• If NO tools are needed return:
[]

• If tools are needed return JSON list like:

[
  {{"name":"get_current_stock_price","args":{{"symbol":"RELIANCE"}}}},
  {{"name":"get_current_stock_price","args":{{"symbol":"INFY"}}}}
]

Rules:
- Each tool call must contain ONE symbol string
- Never return a list of symbols
- Only use tools from the available list
- Return ONLY valid JSON
"""

        response = self.llm.invoke(prompt)

        tool_calls = []

        try:

            content = response.content.strip()

            match = re.search(r"\[.*\]", content, re.DOTALL)

            if match:
                parsed = json.loads(match.group())

                if isinstance(parsed, list):
                    tool_calls = parsed

        except Exception as e:

            LOGGER.warning("Planner returned invalid JSON: %s", e)

        validated_calls = []

        for call in tool_calls:

            if not isinstance(call, dict):
                continue

            name = call.get("name")
            args = call.get("args", {})

            if name not in self.tool_map:
                LOGGER.warning("Planner requested unknown tool: %s", name)
                continue

            if not isinstance(args, dict):
                args = {}

            # Prevent symbol list issue
            symbol = args.get("symbol")
            if isinstance(symbol, list):
                for s in symbol:
                    validated_calls.append({
                        "name": name,
                        "args": {"symbol": str(s)}
                    })
                continue

            validated_calls.append({
                "name": name,
                "args": args
            })

        # Reset results if new plan generated
        if validated_calls:
            state["tool_results"] = []

        state["tool_calls"] = validated_calls
        state["invalid_data"] = False
        state["analysis_tools_needed"] = False

        LOGGER.info("Planner created %d tool calls", len(validated_calls))

        return state