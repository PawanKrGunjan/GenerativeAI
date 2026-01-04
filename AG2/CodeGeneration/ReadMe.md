## Built-in Agent Types

AG2 provides specialized agent classes built on `ConversableAgent` to streamline common workflows such as task-solving, tool use, and user interaction.

#### AssistantAgent — Task-solving LLM assistant

`AssistantAgent` is a subclass of `ConversableAgent` configured with a default system message tailored for solving tasks using LLMs. It can suggest Python code blocks, offer debugging suggestions, and provide structured responses.

- `human_input_mode`: Defaults to `"NEVER"` — the assistant operates autonomously.
- `code_execution_config`: Defaults to `False` — it does **not execute code** itself.
- Designed to work collaboratively with other agents (for example, `UserProxyAgent`) that handle execution.

This agent excels at reasoning, planning, and generating code — and expects others to handle the execution layer.

#### UserProxyAgent — Executing code on behalf of the user

`UserProxyAgent` is a subclass of `ConversableAgent` that acts as a proxy for the human user. It is designed to **execute code**, simulate user decisions, and provide execution-based feedback to other agents like `AssistantAgent`.

- `human_input_mode`: Defaults to `"ALWAYS"` — prompts the user at every turn.
- `llm_config`: Defaults to `False` — no LLM responses unless explicitly configured.
- **Code execution is enabled by default.**

You can customize its behavior by:
- Registering an auto-reply function via `.register_reply()`.
- Overriding `.get_human_input()` to change how user input is gathered.
- Overriding `.execute_code_blocks()`, `.run_code()`, or `.execute_function()` to control code execution behavior.

These two agents are often paired: the `AssistantAgent` writes code, and the `UserProxyAgent` executes it.
