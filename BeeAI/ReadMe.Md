# BeeAI Quick Reference Notes  

This practical cheat-sheet covers **BeeAI** (also known as BeeAI Framework or Bee Agent Framework), an open-source toolkit from IBM Research and the Linux Foundation AI & Data program. It helps build production-ready, autonomous AI agents and multi-agent systems in **Python** and **TypeScript**.

BeeAI focuses on developer experience, interoperability (via Agent2Agent - A2A Protocol), observability, and production features like sandboxed code execution, memory optimization, and error handling.

**Key Highlights :**
- Supports single agents (ReAct, Tool Calling) and multi-agent workflows
- Integrates with Ollama, OpenAI, Groq, watsonx.ai, DeepSeek, Llama 3.x, Granite models
- Built-in tools (search, weather, Wikipedia, etc.) + custom tools (LangChain compatible)
- Excellent traceability, telemetry, and OpenInference instrumentation
- Part of a broader ecosystem: BeeAI Platform for discovery/sharing, Agent Stack for deployment

**Main Repo:** https://github.com/i-am-bee/beeai-framework  
**Docs:** https://docs.beeai.dev or https://framework.beeai.dev  
**Python Package:** `pip install beeai-framework`

---

### 1. Basic Setup (Python)

```python
# Install
pip install beeai-framework

# Optional: For specific backends or tools
pip install beeai-framework[ollama]  # or [openai], [groq], etc.

import asyncio
import logging
from dotenv import load_dotenv

from beeai_framework.backend import ChatModel
from beeai_framework.agents.react import ReActAgent  # Main agent type
from beeai_framework.tools.search.wikipedia import WikipediaTool
from beeai_framework.tools.weather import OpenMeteoTool

load_dotenv()  # Load API keys from .env
logging.basicConfig(level=logging.INFO)
```

**LLM Backend Examples:**

```python
# Ollama (local)
llm = ChatModel.from_name("ollama:llama3.2")

# OpenAI
llm = ChatModel.from_name("openai:gpt-4o-mini")

# Groq
llm = ChatModel.from_name("groq:llama3-70b-8192")

# watsonx.ai
llm = ChatModel.from_name("watsonx:granite-3.3-8b-instruct")
```

---

### 2. Creating a Single Agent

#### ReAct Agent (Recommended – modern reasoning + tool use)

```python
agent = ReActAgent(
    llm=llm,
    tools=[WikipediaTool(), OpenMeteoTool()],
    # memory=TokenMemory(max_tokens=4000),  # Optional: limit context
)

async def run_agent():
    result = await agent.run("What is the weather in Paris today? Explain the Eiffel Tower history.")
    print(result.final_output)

asyncio.run(run_agent())
```

**Other Agent Types:**
- `ToolCallingAgent` → Optimized for heavy tool use (parallel calls) – legacy but still supported

---

### 3. Multi-Agent Workflows

BeeAI shines at orchestrating specialized agents.

```python
from beeai_framework.workflows.agent import AgentWorkflow, AgentWorkflowInput

# Define specialized agents
researcher = ReActAgent(llm=llm, tools=[WikipediaTool()], name="Researcher")
analyst = ReActAgent(llm=llm, tools=[], name="Analyst")  # Pure reasoning

# Create workflow
workflow = AgentWorkflow(
    llm=llm,  # Coordinator LLM
    agents=[researcher, analyst]
)

async def run_workflow():
    input_data = AgentWorkflowInput(message="Analyze recent trends in AI agents in 2026.")
    result = await workflow.run(input_data)
    print("Final Report:", result.final_output)

asyncio.run(run_workflow())
```

**When to use:**
- **Single Agent** — Simple tasks with tools (search, weather, calculations)
- **Multi-Agent Workflow** — Complex problems needing specialization (research → analysis → summarization)

---

### 4. Tools & Custom Tools

Built-in tools: Wikipedia, DuckDuckGo, OpenMeteo (weather), etc.

**Custom Tool Example (LangChain compatible):**

```python
from langchain_core.tools import StructuredTool

def multiply(a: int, b: int) -> int:
    """Multiply two numbers"""
    return a * b

multiply_tool = StructuredTool.from_function(multiply)

agent = ReActAgent(llm=llm, tools=[multiply_tool])
```

Supports Model Context Protocol (MCP) for advanced integrations.

---

### 5. Memory & Observability

```python
from beeai_framework.memory import TokenMemory, UnconstrainedMemory

# Token-limited (cost-efficient)
agent = ReActAgent(llm=llm, memory=TokenMemory(max_tokens=8000))

# Unlimited history
agent = ReActAgent(llm=llm, memory=UnconstrainedMemory())
```

**Observability:** Built-in event emitter + OpenInference tracing → full visibility into thoughts, tool calls, and decisions.

---

### 6. Advanced Features

- **Sandboxed Code Execution** — Safe Python code running
- **Structured Outputs** — Pydantic models for reliable JSON
- **Emitter Events** — Real-time streaming and debugging
- **Error Handling** — Clear FrameworkError exceptions
- **Multi-Provider Support** — Switch LLMs easily
- **Agent2Agent (A2A) Protocol** — Cross-framework collaboration (LangChain, CrewAI, etc.)

---

### 7. Best Practices Summary

| Goal                                | Recommended Approach                                      |
|-------------------------------------|-----------------------------------------------------------|
| Quick local agent                   | Ollama + ReActAgent                                       |
| Production multi-agent              | AgentWorkflow + multiple specialized agents               |
| Tool-heavy tasks                    | ReActAgent or ToolCallingAgent + custom tools             |
| Cost control                        | TokenMemory + smaller models (Llama 3.x, Granite)         |
| Debugging/Tracing                   | Use built-in emitter + OpenInference                      |
| Cross-framework deployment          | BeeAI Platform + Agent Stack                              |

---

### 8. Recommended Models (2026)

| Provider   | Model Example                  | Use Case                          |
|------------|--------------------------------|-----------------------------------|
| Ollama     | llama3.2, granite3.3:8b        | Local/fast prototyping            |
| OpenAI     | gpt-4o-mini, gpt-4o            | High quality                      |
| Groq       | llama3-70b-8192                | Speed + quality                   |
| watsonx    | granite-3.3-8b-instruct        | Enterprise/open models            |

---

**You now have a solid BeeAI foundation!**  
BeeAI is excellent for production-grade, observable, multi-agent systems with great interoperability.

Explore examples: https://github.com/i-am-bee/beeai-framework/tree/main/python/examples  
Full docs: https://docs.beeai.dev

**Prepared By: Pawan Kumar Gunjan**  
**Date: January 09, 2026**