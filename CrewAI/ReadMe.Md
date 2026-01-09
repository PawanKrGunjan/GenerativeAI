# CrewAI Quick Reference Notes  

CrewAI is built from scratch (independent of LangChain), emphasizing simplicity, speed, reliability, and production-readiness. It supports hierarchical processes, delegation, memory, tools, and now **Flows** (event-driven workflows) alongside classic Crews.

**Key Highlights (2026 status):**
- 42k+ GitHub stars, thriving community
- Supports OpenAI, Groq, Anthropic, Ollama/local models, watsonx.ai, etc.
- Built-in + custom tools (including crewai-tools package)
- Processes: Sequential, Hierarchical (manager delegates)
- Advanced: Memory, caching, observability, enterprise integrations
- Main Repo: https://github.com/crewAIInc/crewAI
- Docs: https://docs.crewai.com
- Install: `pip install crewai` (or `crewai[tools]` for extras)

---

### 1. Basic Setup (Python)

```python
# Install
pip install crewai          # Core framework
pip install 'crewai[tools]' # Includes popular tools (Serper, Browser, etc.)

import os
from crewai import Agent, Task, Crew, Process

# Set API keys (OpenAI example)
os.environ["OPENAI_API_KEY"] = "your_openai_key"

# Optional: For local models via Ollama
# pip install crewai[ollama]
# os.environ["OPENAI_API_BASE"] = "http://localhost:11434/v1"
```

---

### 2. Creating Agents

Agents have **role**, **goal**, and **backstory** for strong role-playing behavior.

```python
researcher = Agent(
    role="Senior Researcher",
    goal="Discover groundbreaking insights about AI agents",
    backstory="You're a seasoned researcher with 15 years in AI trends.",
    allow_delegation=False,  # Can't delegate to others
    verbose=True,
    # llm="gpt-4o-mini"  # Optional: override default model
)

writer = Agent(
    role="Professional Writer",
    goal="Craft engaging and insightful articles",
    backstory="You write for top tech publications like Wired.",
    allow_delegation=True
)
```

**When to customize:**
- `allow_delegation=False` → Specialist (does own work)
- `allow_delegation=True` → Can hand off subtasks (useful in hierarchical crews)

---

### 3. Defining Tasks

Tasks are explicit instructions with expected outputs.

```python
research_task = Task(
    description="Research the latest trends in multi-agent AI frameworks in 2026.",
    expected_output="A bullet-point summary with sources.",
    agent=researcher,
    # tools=[SerperDevTool()]  # Optional tools
)

write_task = Task(
    description="Write a compelling 800-word article based on the research.",
    expected_output="Markdown article with title, intro, sections, conclusion.",
    agent=writer,
    output_file="ai_trends_2026.md"  # Auto-saves result
)
```

---

### 4. Assembling & Running a Crew

```python
crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, write_task],
    process=Process.sequential,  # Or Process.hierarchical (manager agent delegates)
    verbose=2,                   # Detailed logging
    memory=True,                 # Enables conversation/memory persistence
    cache=True                   # Speeds up repeated runs
)

result = crew.kickoff(inputs={"topic": "Multi-agent AI in 2026"})  # Optional inputs
print(result)
```

**Hierarchical Example (with Manager):**

```python
manager = Agent(role="Project Manager", goal="Oversee and delegate effectively", ...)

crew = Crew(
    agents=[manager, researcher, writer],
    tasks=[research_task, write_task],
    process=Process.hierarchical,  # Manager decides who does what
    manager_agent=manager
)
```

**When to use processes:**
- **Sequential** — Simple, linear workflows (fast & predictable)
- **Hierarchical** — Complex tasks needing delegation & oversight (more autonomous)

---

### 5. Tools Integration

CrewAI has excellent built-in and custom tool support.

```python
from crewai_tools import SerperDevTool, ScrapeWebsiteTool, FileReadTool

# Add to agent
researcher = Agent(
    ...,
    tools=[SerperDevTool(), ScrapeWebsiteTool()]
)
```

**Custom Tool Example:**

```python
from crewai_tools import BaseTool

class MyCalculatorTool(BaseTool):
    name: str = "Calculator"
    description: str = "Performs basic math operations"

    def _run(self, expression: str) -> str:
        return str(eval(expression))  # Use safely!

calc_tool = MyCalculatorTool()
agent = Agent(..., tools=[calc_tool])
```

---

### 6. Advanced Features

- **Flows** → Event-driven workflows (new in recent versions) for stateful, reactive systems
- **Memory** → Short-term, long-term, entity memory options
- **Caching** → Avoid re-running expensive tasks
- **Observability** → Built-in logging, tracing
- **Enterprise** → CrewAI+ platform for deployment, monitoring, security

---

### 7. Best Practices Summary

| Goal                                | Recommended Approach                                      |
|-------------------------------------|-----------------------------------------------------------|
| Quick prototype                     | Sequential crew with 2-3 agents                           |
| Complex delegation                  | Hierarchical process + manager agent                      |
| Research/heavy tool use             | Agents with Serper, Browser, custom tools                 |
| Production reliability              | Enable memory=True, cache=True, verbose logging           |
| Local/offline models                | Ollama + local LLMs (Llama 3.x, Granite)                  |
| Scalable workflows                  | Combine Crews + Flows                                     |

---

### 8. Recommended Models (2026)

| Provider   | Model Example             | Use Case                          |
|------------|---------------------------|-----------------------------------|
| OpenAI     | gpt-4o, gpt-4o-mini       | Best quality / cost balance       |
| Groq       | llama3-70b-8192           | Ultra-fast inference              |
| Ollama     | llama3.2, granite-3.3     | Local/privacy-focused             |
| Anthropic  | claude-3.5-sonnet         | Strong reasoning                  |

---

**You now have a solid CrewAI foundation!**  
CrewAI is perfect for structured, role-based multi-agent automation with excellent developer experience and production features.

Explore examples: https://github.com/crewAIInc/crewAI-examples  
Community: Discord & Forum links in docs

**Prepared By: Pawan Kumar Gunjan**  
**Date: January 09, 2026**