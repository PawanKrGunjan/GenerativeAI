# AG2 Quick Reference Notes  

AG2 is an open-source **AgentOS** (Agent Operating System) for building AI agents and enabling multi-agent cooperation. It focuses on streamlining agentic AI development with features like conversable agents, LLM integration, tools, human-in-the-loop workflows, and multi-agent orchestration patterns.

Key differences from original AutoGen:  
- Rebranded under new governance (Apache 2.0 license for new contributions).  
- Added features: Remote agents, Bedrock support, cost tracking, structured outputs, RAG, tools with secrets, and a pattern cookbook.  
- Python 3.10+ required (dropped 3.9 support).  
- Community-driven: Join Discord at https://discord.gg/sNGSwQME3x for support.

---

### 1. Basic Setup

```python
# Install AG2 (alias 'autogen' on PyPI)
# For OpenAI support:
pip install ag2[openai]  # Or 'ag2[bedrock]' for AWS Bedrock, etc.

# Import core components
import logging
from autogen import ConversableAgent, LLMConfig, UserProxyAgent, register_function
from autogen.agentchat.group.patterns import AutoPattern
from autogen.agentchat.group import run_group_chat

# Setup logging (optional but recommended)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# LLM Configuration (load from JSON file: OAI_CONFIG_LIST)
# Example OAI_CONFIG_LIST.json content:
# [{"model": "gpt-4o-mini", "api_key": "your_openai_key", "base_url": "https://api.openai.com/v1"}]
llm_config = LLMConfig.from_json(path="OAI_CONFIG_LIST")
```

**Tips:**  
- Use extras like `[openai]`, `[bedrock]`, etc., based on your LLM provider.  
- For production: Enable cost tracking in group chats via config options.

---

### 2. Creating Agents

Agents in AG2 are based on `ConversableAgent` (core class for sending/receiving messages and generating replies via LLMs, tools, or humans).

#### Basic Agent

```python
# Simple Assistant Agent (uses LLM for replies)
assistant = ConversableAgent(
    name="assistant_agent",
    system_message="You are a helpful AI assistant that summarizes text.",
    llm_config=llm_config,
    human_input_mode="NEVER",  # Options: "NEVER", "ALWAYS", "TERMINATE"
    is_termination_msg=lambda msg: "DONE" in msg.get("content", "").upper()  # Custom termination condition
)

# Run a simple interaction
response = assistant.generate_reply(message="Summarize: AG2 is an AgentOS for AI agents.")
logger.info(response)
```

#### User Proxy Agent (Human-in-the-Loop)

```python
user_proxy = UserProxyAgent(
    name="user_proxy",
    system_message="You represent the human user and provide input when needed.",
    code_execution_config={"work_dir": "coding", "use_docker": False},  # For safe code execution
    human_input_mode="ALWAYS"  # Prompts human for every reply
)

# Simple single-agent flow
user_proxy.run(assistant, message="What is AG2?").process()
```

**When to use:**  
- `ConversableAgent`: For autonomous AI agents (e.g., planner, coder, reviewer).  
- `UserProxyAgent`: When human feedback is required (e.g., validation, oversight).  
- Customize `system_message` for role-specific behavior; `human_input_mode` for autonomy level.

---

### 3. Multi-Agent Conversations

AG2 excels at orchestrating multiple agents via built-in patterns (e.g., group chats, swarms, nested/sequential).

#### Simple Two-Agent Conversation

```python
coder = ConversableAgent(
    name="coder",
    system_message="You are a Python developer. Write short Python scripts.",
    llm_config=llm_config
)

reviewer = ConversableAgent(
    name="reviewer",
    system_message="You are a code reviewer. Suggest improvements only; no code generation.",
    llm_config=llm_config
)

# Run conversation
response = reviewer.run(
    recipient=coder,
    message="Write a Python function for Fibonacci numbers.",
    max_turns=10  # Limit turns to prevent infinite loops
)
response.process()
logger.info("Final summary: %s", response.summary)
```

#### Group Chat with AutoPattern

```python
# Define agents
planner = ConversableAgent(
    name="planner",
    system_message="Create lesson plans for 4th grade.",
    llm_config=llm_config
)

reviewer = ConversableAgent(
    name="reviewer",
    system_message="Suggest up to 3 improvements based on curriculum.",
    llm_config=llm_config
)

teacher = ConversableAgent(
    name="teacher",
    system_message="Decide topics and finalize plans. Output 'DONE!' when satisfied.",
    is_termination_msg=lambda msg: "DONE!" in msg.get("content", "").upper(),
    llm_config=llm_config
)

# AutoPattern for orchestration
auto_pattern = AutoPattern(
    agents=[teacher, planner, reviewer],
    initial_agent=planner,  # Start with planner
    group_manager_args={"name": "group_manager", "llm_config": llm_config}
)

# Run group chat
response = run_group_chat(
    pattern=auto_pattern,
    messages="Introduce kids to the solar system.",  # Initial message
    max_rounds=20  # Safety limit
)
response.process()
logger.info("Final lesson plan: %s", response.summary)
```

#### With Human Validator

Add `user_agent=human_validator` to `AutoPattern` for human-in-the-loop:

```python
human_validator = UserProxyAgent(
    name="human_validator",
    system_message="Approve or revise lesson plans."
)

auto_pattern = AutoPattern(
    agents=[teacher, planner, reviewer],
    initial_agent=teacher,
    user_agent=human_validator,  # Injects human oversight
    group_manager_args={"name": "group_manager", "llm_config": llm_config}
)
```

**When to use patterns:**  
- **Two-agent (run())**: Simple back-and-forth (e.g., coder-reviewer).  
- **Group (run_group_chat + AutoPattern)**: Complex collaboration (e.g., planning workflows).  
- **Human-in-loop**: When accuracy/oversight is critical (e.g., education, code review).  
- Limit `max_turns` or `max_rounds` to control costs and prevent loops.

---

### 4. Tools Integration

Agents can register and invoke tools (functions) for external capabilities.

```python
from typing import Annotated
from datetime import datetime

# Define a tool function
def get_weekday(date_string: Annotated[str, "Format: YYYY-MM-DD"]) -> str:
    dt = datetime.strptime(date_string, "%Y-%m-%d")
    return dt.strftime("%A")

# Agents
date_agent = ConversableAgent(
    name="date_agent",
    system_message="You find the day of the week for dates.",
    llm_config=llm_config
)

executor_agent = ConversableAgent(
    name="executor_agent",
    human_input_mode="NEVER",
    llm_config=llm_config
)

# Register tool
register_function(
    get_weekday,
    caller=date_agent,     # Who can call it
    executor=executor_agent,  # Who executes it
    description="Get the day of the week for a given date"
)

# Run with tool
chat_result = executor_agent.initiate_chat(
    recipient=date_agent,
    message="What day was 1995-03-25?",
    max_turns=2
)
logger.info("Result: %s", chat_result.chat_history[1]["content"])
```

**When to use tools:**  
- For non-LLM tasks (e.g., APIs, calculations, file I/O).  
- Supports secrets (e.g., API keys) and structured outputs (e.g., JSON).  
- Integrate with RAG for knowledge retrieval.

---

### 5. Advanced Features

- **Structured Outputs:** Use `llm_config` with schemas for JSON responses.  
- **RAG (Retrieval-Augmented Generation):** Built-in for agent memory/knowledge bases.  
- **Remote Agents:** Run agents on different machines via config.  
- **Cost Tracking:** Enabled in group chats; logs token usage.  
- **Pattern Cookbook:** 9 built-in orchestration patterns (e.g., swarms, nested chats).  
- **Code Execution:** Safe via `code_execution_config` (with/without Docker).

---

### 6. Best Practices Summary

| Goal                                | Recommended Approach                                      |
|-------------------------------------|-----------------------------------------------------------|
| Quick start                         | Single `ConversableAgent` + `run()`                       |
| Multi-agent workflows               | `AutoPattern` + `run_group_chat()`                        |
| Human oversight                     | `UserProxyAgent` with `human_input_mode="ALWAYS"`         |
| External capabilities               | `register_function()` for tools                           |
| Safety/Control                      | Set `max_turns`, termination conditions                   |
| Config/Logging                      | Use `OAI_CONFIG_LIST.json` and basic logging              |
| Extensibility                       | Custom reply methods, remote agents, Bedrock integration  |

---

### 7. Common LLM Models & Providers

- **OpenAI:** `gpt-4o-mini` (fast/cheap), `gpt-4o` (advanced).  
- **Bedrock:** Install `ag2[bedrock]`; supports AWS models.  
- **Others:** Any LLM via config (e.g., custom APIs).

---

**You now have a clean, working AG2 setup!**  
This framework is ideal for agentic AI research and apps. Explore examples at https://github.com/ag2ai/build-with-ag2.


**Prepared By: Pawan Kumar Gunjan**  
**Date: January 09, 2026**