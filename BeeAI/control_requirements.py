import asyncio
import logging
from beeai_framework.agents.experimental import RequirementAgent
from beeai_framework.agents.experimental.requirements.conditional import ConditionalRequirement
from beeai_framework.memory import UnconstrainedMemory
from beeai_framework.backend import ChatModel, ChatModelParameters
from beeai_framework.tools.think import ThinkTool
from beeai_framework.tools.search.wikipedia import WikipediaTool
from beeai_framework.middleware.trajectory import GlobalTrajectoryMiddleware
from beeai_framework.tools import Tool
# from dotenv import load_dotenv
# import os

# load_dotenv()
# os.environ["OPENAI_API_KEY"] = os.getenv("PERPLEXITY_API_KEY")
# print("Key loaded:", bool(os.getenv("OPENAI_API_KEY")))
# print("Base set:", os.getenv("OPENAI_API_BASE"))

async def controlled_execution_example():
    """
    RequirementAgent with Controlled Execution - Requirements System
    
    Requirements provide precise control over tool execution order and behavior.
    Same query, same tracking - but now with strict execution rules.
    """
    #llm = ChatModel.from_name("watsonx:meta-llama/llama-4-maverick-17b-128e-instruct-fp8", ChatModelParameters(temperature=0))
    #llm = ChatModel.from_name("openai:sonar-pro",ChatModelParameters(temperature=0))
    llm = ChatModel.from_name("ollama:llama3.2", ChatModelParameters(temperature=0, max_tokens=300))

    # SAME SYSTEM PROMPT as previous examples
    SYSTEM_INSTRUCTIONS = """You are an expert cybersecurity analyst specializing in threat assessment and risk analysis.

Your methodology:
1. Analyze the threat landscape systematically
2. Research authoritative sources when available
3. Provide comprehensive risk assessment with actionable recommendations
4. Focus on practical, implementable security measures"""
    
    # RequirementAgent with strict execution control
    controlled_agent = RequirementAgent(
        llm=llm,
        tools=[ThinkTool(), WikipediaTool()],
        memory=UnconstrainedMemory(),
        instructions=SYSTEM_INSTRUCTIONS,
        middlewares=[GlobalTrajectoryMiddleware(included=[Tool])],
        
        # REQUIREMENTS: Declarative control over execution flow
        requirements=[
            # MUST start with systematic thinking
            ConditionalRequirement(
                ThinkTool,
                force_at_step=1,  # Thinking required first
                min_invocations=1,  # At least once
                max_invocations=3,  # Max number of invocations
                consecutive_allowed=False  # No repeated thinking
            ),
            # Wikipedia research only after thinking
            ConditionalRequirement(
                WikipediaTool,
                only_after=[ThinkTool],  # Only after thinking
                min_invocations=1,  # Must research at least once
                max_invocations=2  # Max number of research calls
            )
        ]
    )
    
    # SAME QUERY as all previous examples
    ANALYSIS_QUERY = """Analyze the cybersecurity risks of quantum computing for financial institutions. 
    What are the main threats, timeline for concern, and recommended preparation strategies?"""
    
    result = await controlled_agent.run(ANALYSIS_QUERY)
    print(f"\n🔧 Controlled Execution Analysis:\n{result.answer.text}")

async def main() -> None:
    logging.getLogger('asyncio').setLevel(logging.CRITICAL)
    await controlled_execution_example()

if __name__ == "__main__":
    asyncio.run(main())