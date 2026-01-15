"""
Multi-Agent Code Generation System (AG2/AutoGen)
================================================

Agents
------
- Manager (Human): Provides tasks and reviews progress.
- Developer (LLM): Writes code; must save via tools (no code pasted in chat).
- Tester (LLM): Runs validations/tests and reports PASS/FAIL; can only create NEW test files.
- Executor (Auto): Executes tools automatically (no human approval), except installs.

Logging
-------
- logs/agent.log      : orchestration + high-level tool events
- logs/workspace.log  : detailed workspace operations and subprocess output (from tool.py)

Workspace
---------
- CODE/ : all generated code/artifacts live here
"""

from dataclasses import dataclass
from typing import Callable

from dotenv import load_dotenv
from autogen import (
    ConversableAgent,
    UserProxyAgent,
    GroupChat,
    GroupChatManager,
    LLMConfig,
    register_function,
)

from tool import CodeWorkspace
from log_setup import setup_logging


# -----------------------------
# Runtime config
# -----------------------------

load_dotenv()

MODEL_CONFIG = {
    "model": "mistral",
    "api_type": "ollama",
    "client_host": "http://127.0.0.1:11434",
    "temperature": 0.0,
    "max_tokens": 500,
    "request_timeout": 90,
    "max_retries": 0,
}
llm_config = LLMConfig(MODEL_CONFIG)

CODE_DIR = "CODE"
LOGS_DIR = "logs"

TOOL_DESCRIPTIONS = {
    "ws_list_files": "List files under the CODE workspace directory (optionally under a subdirectory).",
    "ws_read_file": "Read a text file from the CODE workspace directory.",
    "ws_write_file": "Write/overwrite a text file inside CODE. Returns only the saved filename.",
    "ws_run_command": "Run a command inside CODE. Returns exit_code/cwd (full output in logs).",
    "ws_run_python_file": "Run a Python file inside CODE. Returns exit_code/cwd (full output in logs).",
    "ws_run_python_code": "Execute a temporary Python snippet inside CODE. Returns exit_code/cwd (full output in logs).",
    "ws_write_test_file": "Create a NEW test file only under CODE/tests/ (no overwrite). Returns only the saved filename.",
}


# -----------------------------
# System messages
# -----------------------------

MANAGER_SYSTEM_MESSAGE = """
You are the PROJECT MANAGER (human-led).
- Accept requirements from the human.
- Assign tasks to Developer and Tester.
- Human approval is required only for installing packages (pip/apt).
Keep replies short and directive.
"""

DEVELOPER_SYSTEM_MESSAGE = """
You are the DEVELOPER.

Hard rules:
- NEVER paste full code in chat.
- ALWAYS save code using ws_write_file, and reply with only the saved filename + short status.
- To validate, run using ws_run_python_file or ws_run_python_code.
- Do NOT ask permission for saving/running.
- If dependencies are missing, explicitly request installation and wait.

Quality rules:
- Prefer scripts that save artifacts (e.g., plots via savefig) instead of plt.show().
"""

TESTER_SYSTEM_MESSAGE = """
You are the TESTER.

Hard rules:
- NEVER paste full code in chat.
- When asked to check/test, you MUST:
  1) run the target using ws_run_python_file or ws_run_command,
  2) verify artifacts using ws_list_files,
  3) respond in exactly one of these formats:

TESTS PASSED
- exit_code: <int>
- artifacts: <comma-separated filenames>

TESTS FAILED
- exit_code: <int>
- BUG: <description>
- SUGGESTION: <fix>

Permissions:
- You may create NEW tests only using ws_write_test_file under CODE/tests/ (no overwrite).
"""

EXECUTOR_SYSTEM_MESSAGE = """
You are the TOOL EXECUTOR.
Execute tool calls automatically and return tool results.
"""


# -----------------------------
# Install gating (only human prompt)
# -----------------------------

def is_install_command(command: list[str]) -> bool:
    if not command:
        return False
    parts = [str(x).lower() for x in command]
    joined = " ".join(parts)
    return (
        "pip install" in joined
        or "python -m pip install" in joined
        or "apt install" in joined
        or "apt-get install" in joined
        or "conda install" in joined
        or "brew install" in joined
    )


def approve_install(command: list[str]) -> bool:
    print("\nINSTALL REQUEST:", command)
    ans = input("Approve install? (YES/NO): ").strip().lower()
    return ans in ("y", "yes")


# -----------------------------
# Agents
# -----------------------------

def create_agents():
    manager = UserProxyAgent(
        name="Manager",
        system_message=MANAGER_SYSTEM_MESSAGE,
        human_input_mode="ALWAYS",
        llm_config=llm_config,
        code_execution_config=False,
    )

    developer = ConversableAgent(
        name="Developer",
        system_message=DEVELOPER_SYSTEM_MESSAGE,
        llm_config=llm_config,
        max_consecutive_auto_reply=5,
    )

    tester = ConversableAgent(
        name="Tester",
        system_message=TESTER_SYSTEM_MESSAGE,
        llm_config=llm_config,
        max_consecutive_auto_reply=5,
    )

    # Executor runs tools automatically; no code-block execution.
    executor = UserProxyAgent(
        name="Executor",
        system_message=EXECUTOR_SYSTEM_MESSAGE,
        human_input_mode="NEVER",
        llm_config=False,
        code_execution_config=False,
    )

    return manager, developer, tester, executor


# -----------------------------
# Tools
# -----------------------------

@dataclass(frozen=True)
class ToolBundle:
    developer_tools: dict[str, Callable]
    tester_tools: dict[str, Callable]


def build_tools(workspace: CodeWorkspace, log) -> ToolBundle:
    """
    Build schema-friendly tool wrappers:
    - Only use simple runtime types (str, list[str], dict, bool, int).
    - Return minimal data to chat (filenames/exit codes), log details elsewhere.
    """

    def ws_list_files(rel_dir: str = ".") -> list[str]:
        log.info("TOOL ws_list_files rel_dir=%s", rel_dir)
        return workspace.list_files(rel_dir)

    def ws_read_file(rel_path: str, encoding: str = "utf-8") -> str:
        log.info("TOOL ws_read_file rel_path=%s", rel_path)
        return workspace.read_file(rel_path, encoding=encoding)

    def ws_write_file(
        rel_path: str,
        content: str,
        encoding: str = "utf-8",
        overwrite: bool = True,
        create_dirs: bool = True,
    ) -> str:
        log.info("TOOL ws_write_file rel_path=%s overwrite=%s chars=%d", rel_path, overwrite, len(content))
        return workspace.write_file(
            rel_path,
            content,
            encoding=encoding,
            overwrite=overwrite,
            create_dirs=create_dirs,
        )

    def ws_run_command(command: list[str]) -> dict:
        log.info("TOOL ws_run_command command=%s", command)
        if is_install_command(command) and not approve_install(command):
            log.warning("INSTALL denied command=%s", command)
            return {"exit_code": 126, "cwd": str(workspace.code_dir), "command": command}
        r = workspace.run_command(command)
        return {"exit_code": r.exit_code, "cwd": r.cwd, "command": r.command}

    def ws_run_python_file(rel_py_path: str, args: list[str] | None = None) -> dict:
        # NOTE: uses `list[str] | None` (not Optional[list[str]]) to avoid ForwardRef issues.
        log.info("TOOL ws_run_python_file rel_py_path=%s args=%s", rel_py_path, args)
        r = workspace.run_python_file(rel_py_path, args=args or [])
        return {"exit_code": r.exit_code, "cwd": r.cwd, "command": r.command}

    def ws_run_python_code(code: str, filename_hint: str = "snippet") -> dict:
        log.info("TOOL ws_run_python_code filename_hint=%s chars=%d", filename_hint, len(code))
        r = workspace.run_python_code(code, filename_hint=filename_hint)
        return {"exit_code": r.exit_code, "cwd": r.cwd, "command": r.command}

    def ws_write_test_file(rel_path: str, content: str) -> str:
        if not (rel_path.startswith("tests/") or rel_path.startswith("tests\\")):
            raise PermissionError("Tester can only write under CODE/tests/")
        log.info("TOOL ws_write_test_file rel_path=%s chars=%d", rel_path, len(content))
        return workspace.write_file(rel_path, content, overwrite=False, create_dirs=True)

    common = {
        "ws_list_files": ws_list_files,
        "ws_read_file": ws_read_file,
        "ws_run_command": ws_run_command,
        "ws_run_python_file": ws_run_python_file,
        "ws_run_python_code": ws_run_python_code,
    }

    return ToolBundle(
        developer_tools={**common, "ws_write_file": ws_write_file},
        tester_tools={**common, "ws_write_test_file": ws_write_test_file},
    )


def register_tools_for_agents(
    *,
    tools: ToolBundle,
    developer: ConversableAgent,
    tester: ConversableAgent,
    executor: UserProxyAgent,
):
    # description= is required by register_function in your AG2 build. [web:22]
    for name, fn in tools.developer_tools.items():
        register_function(fn, caller=developer, executor=executor, name=name, description=TOOL_DESCRIPTIONS[name])
    for name, fn in tools.tester_tools.items():
        register_function(fn, caller=tester, executor=executor, name=name, description=TOOL_DESCRIPTIONS[name])


# -----------------------------
# Main
# -----------------------------

def main():
    log = setup_logging(logs_dir=LOGS_DIR, log_file="agent.log")
    log.info("Booting multi-agent system")

    manager, developer, tester, executor = create_agents()
    log.info("Agents ready | manager=%s developer=%s tester=%s executor=%s",
             manager.name, developer.name, tester.name, executor.name)

    workspace = CodeWorkspace(
        code_dir=CODE_DIR,
        require_approval=False,  # only installs are gated (inside ws_run_command)
        logs_dir=LOGS_DIR,
        log_file="workspace.log",
    )

    tools = build_tools(workspace, log)
    register_tools_for_agents(tools=tools, developer=developer, tester=tester, executor=executor)

    groupchat = GroupChat(
        agents=[manager, developer, tester, executor],
        messages=[],
        max_round=40,
        speaker_selection_method="round_robin",
    )
    group_manager = GroupChatManager(groupchat=groupchat, llm_config=llm_config)

    msg = input("Manager> ").strip()
    log.info("MANAGER_INPUT: %s", msg)
    manager.initiate_chat(group_manager, message=msg)


if __name__ == "__main__":
    main()
