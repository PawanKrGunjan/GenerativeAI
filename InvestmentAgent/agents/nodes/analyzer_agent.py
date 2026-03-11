# agents/analyzer_agent.py

import json
import re
from agents.llm import get_llm
from utils.logger import LOGGER


class AnalyzerAgent:

    def __init__(self):
        self.llm = get_llm()

    def run(self, state):

        LOGGER.info("Analyzer started")

        query = state.get("query", "")
        tool_data = state.get("tool_results", [])

        # Prevent extremely large prompts
        tool_data_str = str(tool_data)[:4000]

        prompt = f"""
You are a financial analysis AI.

User Query:
{query}

Market Data:
{tool_data_str}

Tasks:

1. Check if the provided data is invalid or incomplete
2. Decide if additional analysis tools are required
3. If enough information exists, produce final investment analysis

Return ONLY valid JSON.

Example format:

{{
 "invalid_data": false,
 "analysis_tools_needed": false,
 "analysis": "Detailed financial insight"
}}
"""

        response = self.llm.invoke(prompt)

        try:

            content = response.content.strip()

            # Extract JSON block
            match = re.search(r"\{.*\}", content, re.DOTALL)

            if not match:
                raise ValueError("No JSON found")

            result = json.loads(match.group())

            state["invalid_data"] = bool(result.get("invalid_data", False))
            state["analysis_tools_needed"] = bool(result.get("analysis_tools_needed", False))
            state["analysis"] = result.get("analysis")

        except Exception as e:

            LOGGER.warning("Analyzer JSON parse failed: %s", e)

            state["invalid_data"] = False
            state["analysis_tools_needed"] = False
            state["analysis"] = response.content

        LOGGER.info(
            "Analyzer result → invalid_data=%s, tools_needed=%s",
            state["invalid_data"],
            state["analysis_tools_needed"]
        )

        return state