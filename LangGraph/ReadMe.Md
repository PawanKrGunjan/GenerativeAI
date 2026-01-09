# LangGraph Quick Reference Notes  

**LangGraph** is a leading framework for building **stateful, controllable, multi-actor agent workflows** with **LangChain**. It extends LangChain expressions (LCEL) by adding **cycles, branching, persistence, and human-in-the-loop** capabilities — essential for production-grade agentic systems.

LangGraph models applications as **graphs**: nodes = agents/tools/functions, edges = message passing with state. It excels at long-running, interruptible, cyclic workflows (e.g., ReAct loops, multi-agent collaboration, planning → execution → reflection).

**Key Highlights (2026 status):**
- Part of LangChain ecosystem (langgraph package)
- Built-in persistence (checkpoints), streaming, moderation, time-travel debugging
- Supports single-agent loops and complex multi-agent graphs
- Integrates seamlessly with LangChain tools, memory, RAG
- Production-ready: Used in LangSmith, many enterprise deployments
- Main Repo: https://github.com/langchain-ai/langgraph
- Docs: https://langchain-ai.github.io/langgraph/
- Install: `pip install langgraph`

---

### 1. Basic Setup (Python)

```python
# Install
pip install langgraph langchain langchain-openai  # or langchain-ollama, etc.

import os
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from typing import Annotated, TypedDict
import operator

# LLM (example with OpenAI)
os.environ["OPENAI_API_KEY"] = "your_key"
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Optional: Local model via Ollama
# from langchain_ollama import ChatOllama
# llm = ChatOllama(model="llama3.2")
```

---

### 2. Defining State

All graphs require a **state schema** (usually a TypedDict).

```python
class AgentState(TypedDict):
    messages: Annotated[list, operator.add]  # Accumulates messages
    next: str                                       # For conditional routing
    # Add custom fields: research_data: dict, etc.
```

`Annotated[list, operator.add]` → automatically appends new messages.

---

### 3. Building a Simple ReAct Agent Graph

```python
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, tools_condition

@tool
def search(query: str) -> str:
    """Search the web"""
    return f"Results for {query}: ..."

tools = [search]
tool_node = ToolNode(tools)
model_with_tools = llm.bind_tools(tools)

# Node functions
def chatbot(state: AgentState):
    return {"messages": [model_with_tools.invoke(state["messages"])]}

# Build graph
builder = StateGraph(AgentState)

builder.add_node("chatbot", chatbot)
builder.add_node("tools", tool_node)

builder.add_edge(START, "chatbot")
builder.add_conditional_edges(
    "chatbot",
    tools_condition,  # Routes to "tools" if tool call, else END
    {"tools": "tools", END: END}
)
builder.add_edge("tools", "chatbot")  # Loop back

memory = MemorySaver()  # For persistence & time-travel
graph = builder.compile(checkpointer=memory)

# Run
config = {"configurable": {"thread_id": "123"}}  # Required for persistence
for chunk in graph.stream(
    {"messages": [("human", "What is LangGraph?")]},
    config,
    stream_mode="values"
):
    chunk["messages"][-1].pretty_print()
```

---

### 4. Multi-Agent Collaboration Graph

```python
researcher = llm.with_config(tags=["research"])
critic = llm.with_config(tags=["critic"])

def researcher_node(state):
    return {"messages": [researcher.invoke(state["messages"])]}

def critic_node(state):
    return {"messages": [critic.invoke(state["messages"])]}

builder = StateGraph(AgentState)
builder.add_node("researcher", researcher_node)
builder.add_node("critic", critic_node)

builder.add_edge(START, "researcher")
builder.add_edge("researcher", "critic")
builder.add_edge("critic", END)  # Or loop back for refinement

graph = builder.compile()
```

**Conditional routing example:**

```python
def route(state):
    last_msg = state["messages"][-1]
    if "FINAL" in last_msg.content:
        return END
    return "critic"

builder.add_conditional_edges("researcher", route)
```

---

### 5. Human-in-the-Loop & Persistence

```python
memory = MemorySaver()
graph = builder.compile(checkpointer=memory, interrupt_before=["tools"])

# Run until interruption
config = {"configurable": {"thread_id": "abc"}}
events = graph.stream(inputs, config, stream_mode="values")

# Later resume after human approval
graph.update_state(config, {"messages": [("human", "Approved!")]})
for event in graph.stream(None, config):
    ...
```

**Use cases:**
- Wait for human approval before tool execution
- Time-travel debugging in LangSmith

---

### 6. Advanced Features

- **Checkpoints** → Save/resume state anytime
- **Streaming** → `stream_mode="values"`, `"updates"`, `"debug"`
- **Moderation** → Add moderation node before LLM calls
- **Memory** → Built-in or custom (e.g., conversation summary)
- **Pre-built Templates** → ReAct, Planner-Executor, Supervisor patterns
- **LangGraph Platform** → Cloud deployment, tracing, scaling

---

### 7. Best Practices Summary

| Goal                                | Recommended Approach                                      |
|-------------------------------------|-----------------------------------------------------------|
| Simple tool-using agent             | ReAct pattern with ToolNode + tools_condition             |
| Multi-agent collaboration           | Multiple nodes + conditional edges                        |
| Human oversight                     | `interrupt_before` + resume with human input              |
| Long-running workflows              | Checkpointer + thread_id persistence                      |
| Debugging                           | LangSmith tracing + stream_mode="debug"                   |
| Production reliability              | Moderation nodes, error handling, retries                 |

---

### 8. Recommended Models (2026)

| Provider   | Model Example             | Use Case                          |
|------------|---------------------------|-----------------------------------|
| OpenAI     | gpt-4o, gpt-4o-mini       | Best reasoning & tool calling     |
| Anthropic  | claude-3.5-sonnet         | Strong planning & safety          |
| Ollama     | llama3.2, granite-3.3     | Local/privacy-focused             |
| Groq       | llama3-70b-8192           | Speed-critical applications       |

---

**You now have a solid LangGraph foundation!**  
LangGraph is the gold standard for **controllable, stateful, production-grade agent workflows** with cycles and persistence.

Explore examples: https://github.com/langchain-ai/langgraph/tree/main/examples  
Tutorials: https://langchain-ai.github.io/langgraph/tutorials/

**Prepared By: Pawan Kumar Gunjan**  
**Date: January 09, 2026**