# agents/reflection_agent.py

from agents.llm import get_llm
from memory.memory_store import save_memory
from utils.logger import LOGGER


class ReflectionAgent:

    def __init__(self):

        self.llm = get_llm()

    def run(self, state):

        LOGGER.info("Reflection started")

        analysis = state.get("analysis")

        if not analysis:

            LOGGER.warning("No analysis to reflect on")

            return state

        prompt = f"""
Evaluate the investment analysis below.

Analysis:
{analysis}

Provide lessons for improving future investment decisions.
"""

        result = self.llm.invoke(prompt)

        reflection = result.content

        try:

            save_memory({
                "query": state["query"],
                "analysis": analysis,
                "reflection": reflection
            })

        except Exception:
            LOGGER.warning("Memory saving failed")

        state["reflection"] = reflection

        LOGGER.info("Reflection stored")

        return state