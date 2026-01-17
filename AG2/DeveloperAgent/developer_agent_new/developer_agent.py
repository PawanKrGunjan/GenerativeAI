"""
Multi-Agent Code Generation System (AutoGen AgentChat v0.4+)
===========================================================

Workspace layout:
- CODE_DIR/<project>/      (active workspace root)
- logs/agent.log           (rotating file logs)

Manager commands:
- files / :files [dir]     list files under workspace (dir optional)
- status / :status [dir]   workspace snapshot (dir optional)
- :new <name>              switch project (resets agents + new workspace)
- :ps                      open multiline problem-statement block (END to finish)
- help / :help
- EXIT / QUIT
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import FunctionalTermination, MaxMessageTermination
from autogen_agentchat.messages import ToolCallExecutionEvent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console

from autogen_core.tools import FunctionTool

# Prefer OpenAI-compatible client against Ollama /v1 endpoint for reliable tool calling.
# Falls back to OllamaChatCompletionClient if OpenAI client is unavailable.
try:
    from autogen_ext.models.openai import OpenAIChatCompletionClient  # type: ignore
except Exception:  # pragma: no cover
    OpenAIChatCompletionClient = None  # type: ignore

from autogen_ext.models.ollama import OllamaChatCompletionClient

from tool import CodeWorkspace
from log_setup import setup_logging


# -----------------------------
# Runtime config
# -----------------------------
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
LOGS_DIR = "logs"
PROJECTS_ROOT = "CODE_DIR"  # per-project root folder (fixes drift)

MODEL = "llama3.1:latest"
OLLAMA_HOST = "http://127.0.0.1:11434"
OLLAMA_OPENAI_BASE_URL = f"{OLLAMA_HOST}/v1"


def build_model_client():
    model_info = {
        "vision": False,
        "function_calling": True,
        "json_output": True,
        "family": "llama",
    }

    # Best-effort: use OpenAI-compatible API if available (more consistent tool calls).
    if OpenAIChatCompletionClient is not None:
        return OpenAIChatCompletionClient(
            model=MODEL,
            base_url=OLLAMA_OPENAI_BASE_URL,
            api_key="ollama",
            model_info=model_info,
        )

    # Fallback: native Ollama client.
    return OllamaChatCompletionClient(
        model=MODEL,
        host=OLLAMA_HOST,
        timeout=90,
        options={"temperature": 0.0, "num_predict": 500},
        model_info=model_info,
    )


TOOL_DESCRIPTIONS = {
    "ws_list_files": "List files under the workspace (optionally under a subdirectory).",
    "ws_read_file": "Read a text file from the workspace.",
    "ws_write_file": "Write/overwrite a text file inside workspace. Returns saved filename.",
    "ws_write_file_atomic": "Atomic write/overwrite inside workspace. Returns saved filename.",
    "ws_replace_lines": "Replace inclusive [start_line,end_line] (1-based) in a file. Returns saved filename.",
    "ws_insert_lines": "Insert text before at_line (1-based). Returns saved filename.",
    "ws_delete_lines": "Delete inclusive [start_line,end_line] (1-based) in a file. Returns saved filename.",
    "ws_code_status": "Return JSON status of workspace files (path/bytes/mtime/lines/hash).",
    "ws_run_command": "Run a command inside workspace. Returns exit_code/cwd/command.",
    "ws_run_python_file": "Run a Python file inside workspace. Returns exit_code/cwd/command.",
    "ws_run_python_code": "Execute a temporary Python snippet inside workspace. Returns exit_code/cwd/command.",
    "ws_write_test_file": "Create a NEW test file only under tests/ (no overwrite). Returns saved filename.",
    "approve": "Approve the run only if file exists, run exit_code==0, and required artifacts exist.",
}


DEVELOPER_SYSTEM_MESSAGE = """
You are the DEVELOPER.

Goal
- Implement exactly what the Manager requests in the workspace and work with Tester until the run is approved.

Hard rules
- Never paste full code in chat.
- Never print tool-call JSON (no {"name":..., "parameters":...}).
- Use tools directly for file edits and running code.
- Write code only via ws_write_file / ws_write_file_atomic (relative paths only).
- Run the entry script at least once via ws_run_python_file before handoff.
- Always end your final message with:
  FILE: <relative_path_under_workspace>

Process
1) Clarify assumptions briefly if needed.
2) Implement/update files.
3) Run the file once.
4) Hand off FILE to Tester.
"""

TESTER_SYSTEM_MESSAGE = """
You are the TESTER.

Goal
- Verify the Developer’s FILE exists, runs with exit_code 0, and required artifacts exist.
- Then report PASS/FAIL in the exact format below.

Hard rules
- Never paste full code in chat.
- Never print tool-call JSON.
- Use tools directly (ws_code_status/ws_list_files/ws_run_python_file).
- You may create NEW tests only via ws_write_test_file under tests/ (no overwrite).

Approval gate
- After verification, call the approve tool.
- Only report TESTS PASSED if approve returns approved=true.

Output format (exact)

TESTS PASSED
- file: <relative_path>
- exit_code: 0
- artifacts: <comma-separated filenames or NONE>
- summary: <1-2 lines for Manager>

OR

TESTS FAILED
- file: <relative_path or NONE>
- exit_code: <int or NONE>
- BUG: <short description>
- SUGGESTION: <precise fix instructions for Developer>
"""


# -----------------------------
# Helpers
# -----------------------------
def _safe_project_name(name: str) -> str:
    safe = "".join(c for c in name.strip() if c.isalnum() or c in ("-", "_"))
    return safe or "default"


def _project_code_dir(project: str) -> str:
    # Anchor to this script dir to avoid cwd drift.
    return str((BASE_DIR / PROJECTS_ROOT / _safe_project_name(project)).resolve())


def _strip_quotes(s: str) -> str:
    s = s.strip()
    if (len(s) >= 2) and ((s[0] == s[-1] == "'") or (s[0] == s[-1] == '"')):
        return s[1:-1].strip()
    return s


def _norm_local_dir_arg(arg: str) -> str:
    a = (arg or "").strip().replace("\\", "/")
    if not a or a == ".":
        return "."
    if a.lower() in ("code", "code_dir", "/workspace"):
        return "."
    for prefix in ("code/", "code_dir/", "/workspace/"):
        if a.lower().startswith(prefix):
            a = a[len(prefix) :]
            break
    return a or "."


def _parse_local_cmd_with_optional_arg(cmd: str) -> tuple[str, str]:
    raw = cmd.strip()
    if raw.startswith(":"):
        raw = raw[1:].strip()
    if not raw:
        return "", ""
    parts = raw.split(maxsplit=1)
    name = parts[0].lower()
    arg = _strip_quotes(parts[1]) if len(parts) > 1 else ""
    return name, arg


def read_problem_statement_block() -> str:
    print('{problem_statement: """')
    lines: list[str] = []
    while True:
        line = input()
        s = line.strip()

        if s in ('"""}', "END"):
            break

        if s.endswith('"""}'):
            prefix = line[: line.rfind('"""}')].rstrip()
            if prefix:
                lines.append(prefix)
            break

        lines.append(line)

    print('"""}')
    body = "\n".join(lines).rstrip()
    return '{problem_statement: """\n' + body + '\n"""}'


# -----------------------------
# Install gating (optional)
# -----------------------------
def is_install_command(command: list[str]) -> bool:
    if not command:
        return False
    joined = " ".join(str(x).lower() for x in command)
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
# Tools
# -----------------------------
@dataclass(frozen=True)
class ToolBundle:
    developer_tools: list[FunctionTool]
    tester_tools: list[FunctionTool]


def build_tools(workspace: CodeWorkspace, log) -> ToolBundle:
    # Normalize: accept "CODE/xyz.py", "code_dir/xyz.py", "/workspace/xyz.py" => "xyz.py"
    def _norm_rel(p: str) -> str:
        p = (p or "").strip().replace("\\", "/")

        if p.startswith("/workspace/"):
            p = p[len("/workspace/") :]
        elif p == "/workspace":
            p = ""

        p = p.lstrip("/")  # force relative

        if p.lower() in ("code", "code_dir"):
            return "."
        for prefix in ("code/", "code_dir/"):
            if p.lower().startswith(prefix):
                p = p[len(prefix) :]
                break
        if p.startswith("./"):
            p = p[2:]
        return p.strip()

    def _norm_dir(d: str) -> str:
        d = _norm_rel(d)
        return d or "."

    def _file_exists(rel_path: str) -> bool:
        rel_path = _norm_rel(rel_path)
        if rel_path in ("", "."):
            return False
        return (Path(workspace.code_dir) / rel_path).is_file()

    def _artifact_exists(rel_path: str) -> bool:
        rel_path = _norm_rel(rel_path)
        return (Path(workspace.code_dir) / rel_path).exists()

    def ws_list_files(rel_dir: str = ".") -> list[str]:
        rel_dir = _norm_dir(rel_dir)
        log.info("TOOL ws_list_files rel_dir=%s", rel_dir)
        return [str(x) for x in workspace.list_files(rel_dir)]

    def ws_read_file(rel_path: str, encoding: str = "utf-8") -> str:
        rel_path = _norm_rel(rel_path)
        log.info("TOOL ws_read_file rel_path=%s", rel_path)
        return str(workspace.read_file(rel_path, encoding=encoding))

    def ws_write_file(rel_path: str, content: str, encoding: str = "utf-8", overwrite: bool = True) -> str:
        rel_path = _norm_rel(rel_path)
        if overwrite is False:
            log.warning("FORCING overwrite=True for ws_write_file rel_path=%s", rel_path)
        log.info("TOOL ws_write_file rel_path=%s overwrite=True chars=%d", rel_path, len(content))
        return str(workspace.write_file(rel_path, content, encoding=encoding, overwrite=True, create_dirs=True))

    def ws_write_file_atomic(rel_path: str, content: str, encoding: str = "utf-8", overwrite: bool = True) -> str:
        rel_path = _norm_rel(rel_path)
        if overwrite is False:
            log.warning("FORCING overwrite=True for ws_write_file_atomic rel_path=%s", rel_path)
        log.info("TOOL ws_write_file_atomic rel_path=%s overwrite=True chars=%d", rel_path, len(content))
        return str(workspace.write_file_atomic(rel_path, content, encoding=encoding, overwrite=True, create_dirs=True))

    def ws_replace_lines(rel_path: str, start_line: int, end_line: int, replacement: str, encoding: str = "utf-8") -> str:
        rel_path = _norm_rel(rel_path)
        log.info("TOOL ws_replace_lines rel_path=%s start=%d end=%d chars=%d", rel_path, start_line, end_line, len(replacement))
        return str(workspace.replace_lines(rel_path, start_line, end_line, replacement, encoding=encoding))

    def ws_insert_lines(rel_path: str, at_line: int, text: str, encoding: str = "utf-8") -> str:
        rel_path = _norm_rel(rel_path)
        log.info("TOOL ws_insert_lines rel_path=%s at=%d chars=%d", rel_path, at_line, len(text))
        return str(workspace.insert_lines(rel_path, at_line, text, encoding=encoding))

    def ws_delete_lines(rel_path: str, start_line: int, end_line: int, encoding: str = "utf-8") -> str:
        rel_path = _norm_rel(rel_path)
        log.info("TOOL ws_delete_lines rel_path=%s start=%d end=%d", rel_path, start_line, end_line)
        return str(workspace.delete_lines(rel_path, start_line, end_line, encoding=encoding))

    def ws_code_status(rel_dir: str = ".") -> dict:
        rel_dir = _norm_dir(rel_dir)
        log.info("TOOL ws_code_status rel_dir=%s", rel_dir)
        try:
            return workspace.get_code_status(rel_dir, max_files=300)
        except TypeError:
            return workspace.get_code_status(rel_dir)

    def ws_run_command(command: list[str]) -> dict:
        command = [str(x) for x in command]
        log.info("TOOL ws_run_command command=%s", command)

        if is_install_command(command) and not approve_install(command):
            log.warning("INSTALL denied command=%s", command)
            return {"exit_code": 126, "cwd": str(workspace.code_dir), "command": command}

        r = workspace.run_command(command)
        cmd_out = r.command if isinstance(r.command, (list, tuple)) else str(r.command)
        if isinstance(cmd_out, (list, tuple)):
            cmd_out = [str(x) for x in cmd_out]
        return {"exit_code": int(r.exit_code), "cwd": str(r.cwd), "command": cmd_out}

    def ws_run_python_file(
        rel_py_path: str | None = None,
        rel_path: str | None = None,
        args: list[str] | None = None,
    ) -> dict:
        chosen = rel_py_path or rel_path
        if not chosen:
            raise ValueError("ws_run_python_file requires rel_py_path (or rel_path)")

        chosen = _norm_rel(chosen)
        args = [str(x) for x in (args or [])]
        log.info("TOOL ws_run_python_file rel_py_path=%s args=%s", chosen, args)

        r = workspace.run_python_file(chosen, args=args)
        cmd_out = r.command if isinstance(r.command, (list, tuple)) else str(r.command)
        if isinstance(cmd_out, (list, tuple)):
            cmd_out = [str(x) for x in cmd_out]
        return {"exit_code": int(r.exit_code), "cwd": str(r.cwd), "command": cmd_out}

    def ws_run_python_code(code: str, filename_hint: str = "snippet") -> dict:
        log.info("TOOL ws_run_python_code filename_hint=%s chars=%d", filename_hint, len(code))
        r = workspace.run_python_code(code, filename_hint=filename_hint)
        cmd_out = r.command if isinstance(r.command, (list, tuple)) else str(r.command)
        if isinstance(cmd_out, (list, tuple)):
            cmd_out = [str(x) for x in cmd_out]
        return {"exit_code": int(r.exit_code), "cwd": str(r.cwd), "command": cmd_out}

    def ws_write_test_file(rel_path: str, content: str) -> str:
        rel_path = _norm_rel(rel_path)
        if not rel_path.startswith("tests/"):
            raise PermissionError("Tester can only write under tests/")
        log.info("TOOL ws_write_test_file rel_path=%s chars=%d", rel_path, len(content))
        return str(workspace.write_file(rel_path, content, overwrite=False, create_dirs=True))

    def approve(file: str, required_artifacts: list[str] | None = None) -> str:
        """
        A hard gate that:
        - verifies file exists,
        - runs it (exit_code must be 0),
        - verifies required artifacts exist (if provided).
        Returns a JSON string with approved true/false.
        """
        file = _norm_rel(file)
        required_artifacts = required_artifacts or []

        if not _file_exists(file):
            return json.dumps({"approved": False, "file": file, "exit_code": None, "artifacts": [], "reason": "FILE_NOT_FOUND"})

        run = ws_run_python_file(rel_py_path=file)
        exit_code = int(run.get("exit_code", 1))

        missing = [a for a in required_artifacts if not _artifact_exists(a)]
        approved_ok = (exit_code == 0) and (len(missing) == 0)

        return json.dumps(
            {
                "approved": approved_ok,
                "file": file,
                "exit_code": exit_code,
                "artifacts": required_artifacts,
                "missing": missing,
            }
        )

    developer_tools = [
        FunctionTool(ws_list_files, name="ws_list_files", description=TOOL_DESCRIPTIONS["ws_list_files"]),
        FunctionTool(ws_read_file, name="ws_read_file", description=TOOL_DESCRIPTIONS["ws_read_file"]),
        FunctionTool(ws_write_file, name="ws_write_file", description=TOOL_DESCRIPTIONS["ws_write_file"]),
        FunctionTool(ws_write_file_atomic, name="ws_write_file_atomic", description=TOOL_DESCRIPTIONS["ws_write_file_atomic"]),
        FunctionTool(ws_replace_lines, name="ws_replace_lines", description=TOOL_DESCRIPTIONS["ws_replace_lines"]),
        FunctionTool(ws_insert_lines, name="ws_insert_lines", description=TOOL_DESCRIPTIONS["ws_insert_lines"]),
        FunctionTool(ws_delete_lines, name="ws_delete_lines", description=TOOL_DESCRIPTIONS["ws_delete_lines"]),
        FunctionTool(ws_code_status, name="ws_code_status", description=TOOL_DESCRIPTIONS["ws_code_status"]),
        FunctionTool(ws_run_command, name="ws_run_command", description=TOOL_DESCRIPTIONS["ws_run_command"]),
        FunctionTool(ws_run_python_file, name="ws_run_python_file", description=TOOL_DESCRIPTIONS["ws_run_python_file"]),
        FunctionTool(ws_run_python_code, name="ws_run_python_code", description=TOOL_DESCRIPTIONS["ws_run_python_code"]),
    ]

    tester_tools = [
        FunctionTool(ws_list_files, name="ws_list_files", description=TOOL_DESCRIPTIONS["ws_list_files"]),
        FunctionTool(ws_read_file, name="ws_read_file", description=TOOL_DESCRIPTIONS["ws_read_file"]),
        FunctionTool(ws_code_status, name="ws_code_status", description=TOOL_DESCRIPTIONS["ws_code_status"]),
        FunctionTool(ws_run_command, name="ws_run_command", description=TOOL_DESCRIPTIONS["ws_run_command"]),
        FunctionTool(ws_run_python_file, name="ws_run_python_file", description=TOOL_DESCRIPTIONS["ws_run_python_file"]),
        FunctionTool(ws_run_python_code, name="ws_run_python_code", description=TOOL_DESCRIPTIONS["ws_run_python_code"]),
        FunctionTool(ws_write_test_file, name="ws_write_test_file", description=TOOL_DESCRIPTIONS["ws_write_test_file"]),
        FunctionTool(approve, name="approve", description=TOOL_DESCRIPTIONS["approve"]),
    ]

    return ToolBundle(developer_tools=developer_tools, tester_tools=tester_tools)


def create_agents(*, tools: ToolBundle, model_client) -> tuple[AssistantAgent, AssistantAgent]:
    developer = AssistantAgent(
        name="Developer",
        system_message=DEVELOPER_SYSTEM_MESSAGE,
        model_client=model_client,
        tools=tools.developer_tools,
        max_tool_iterations=12,
    )
    tester = AssistantAgent(
        name="Tester",
        system_message=TESTER_SYSTEM_MESSAGE,
        model_client=model_client,
        tools=tools.tester_tools,
        max_tool_iterations=12,
    )
    return developer, tester


def print_help() -> None:
    print("\nManager commands:")
    print("  <text>                 Run a new task (Developer+Tester)")
    print("  files / :files [dir]   List workspace files (dir optional)")
    print("  status / :status [dir] Workspace snapshot (dir optional)")
    print("  :new <name>            Switch project (reset agents + new workspace)")
    print("  :ps                    Paste multiline problem statement (END to finish)")
    print("  help / :help           Show this help")
    print("  EXIT / QUIT            Exit program\n")


# -----------------------------
# Termination (tool-verified)
# -----------------------------
def _approve_success_termination():
    # Terminates only when the approve tool was executed AND returned approved=true.
    seen_approved_true = {"ok": False}

    def expression(messages) -> bool:
        if seen_approved_true["ok"]:
            return True

        for m in messages:
            if isinstance(m, ToolCallExecutionEvent):
                for ex in m.content:
                    if ex.name != "approve":
                        continue
                    try:
                        payload = json.loads(ex.content or "{}")
                    except Exception:
                        continue
                    if payload.get("approved") is True:
                        seen_approved_true["ok"] = True
                        return True
        return False

    return FunctionalTermination(expression)


# -----------------------------
# Main
# -----------------------------
async def amain() -> None:
    log = setup_logging(logs_dir=LOGS_DIR, log_file="agent.log", reconfigure=True)
    log.info("Booting multi-agent system")

    model_client = build_model_client()

    # Stop only when approve returned approved=true, or after a safety cap.
    termination = _approve_success_termination() | MaxMessageTermination(250)

    project = "default"
    code_dir = _project_code_dir(project)

    workspace = CodeWorkspace(
        code_dir=code_dir,
        require_approval=False,
        logs_dir=LOGS_DIR,
        log_file="agent.log",
        logger=log,
    )

    tools = build_tools(workspace, log)
    developer, tester = create_agents(tools=tools, model_client=model_client)
    team = RoundRobinGroupChat([developer, tester], termination_condition=termination)

    print_help()
    print(f"Active project: {project} | CODE dir: {code_dir}")

    try:
        while True:
            cmd = input("\nManager> ").strip()
            if cmd.upper() in ("EXIT", "QUIT"):
                break
            if not cmd:
                continue

            name, arg = _parse_local_cmd_with_optional_arg(cmd)

            if name == "help":
                print_help()
                continue

            if name == "files":
                rel = _norm_local_dir_arg(arg)
                print(workspace.list_files(rel))
                continue

            if name == "status":
                rel = _norm_local_dir_arg(arg)
                try:
                    print(workspace.get_code_status(rel, max_files=300))
                except TypeError:
                    print(workspace.get_code_status(rel))
                continue

            if name == "new":
                new_name = arg
                if not new_name:
                    print("Usage: :new <project_name>")
                    continue

                await team.reset()

                project = _safe_project_name(new_name)
                code_dir = _project_code_dir(project)

                log = setup_logging(logs_dir=LOGS_DIR, log_file="agent.log", reconfigure=True)
                log.info("Switched project | project=%s | code_dir=%s", project, code_dir)

                workspace = CodeWorkspace(
                    code_dir=code_dir,
                    require_approval=False,
                    logs_dir=LOGS_DIR,
                    log_file="agent.log",
                    logger=log,
                )

                tools = build_tools(workspace, log)
                developer, tester = create_agents(tools=tools, model_client=model_client)
                team = RoundRobinGroupChat([developer, tester], termination_condition=termination)

                print(f"Switched to project: {project} | CODE dir: {code_dir}")
                continue

            if name == "ps":
                block = read_problem_statement_block()
                await Console(
                    team.run_stream(
                        task=f"MANAGER TASK:\n{block}\n\nWork until Tester approves (approve tool returns approved=true)."
                    )
                )
                continue

            await Console(
                team.run_stream(
                    task=f"MANAGER TASK:\n{cmd}\n\nWork until Tester approves (approve tool returns approved=true)."
                )
            )

    finally:
        await team.reset()
        await model_client.close()


if __name__ == "__main__":
    asyncio.run(amain())
