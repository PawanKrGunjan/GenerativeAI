# Multi-Agent Code Generation (AutoGen AgentChat v0.4+)

This project runs a local **Manager → Developer → Tester** loop for code generation and testing using AutoGen AgentChat and an Ollama-hosted model.  
It supports multiple isolated projects under `CODE_projects/` and keeps logs in `logs/`.

---

## Folder structure

- `developer_agent.py` — the Manager CLI + multi-agent orchestration
- `tool.py` — `CodeWorkspace` (safe read/write/exec inside the active project folder)
- `log_setup.py` — shared logging setup
- `logs/agent.log` — runtime logs (app + workspace + selected AutoGen logs)
- `CODE_projects/<project_name>/` — workspace for each project (all generated code goes here)

---

## Requirements

1. Python 3.10+ recommended
2. Ollama running locally
3. A tool-calling capable model (example used: `llama3.1:latest`)

Install Python deps (example):
```bash
pip install -U autogen-agentchat autogen-core autogen-ext python-dotenv
```

Pull model:
```bash
ollama pull llama3.1:latest
```

Start Ollama (if not already running) and ensure it’s reachable at:
`http://127.0.0.1:11434`

---

## How to run

From your venv:
```powershell
python .\developer_agent.py
```

You will see:
- help menu (Manager commands)
- active project name and workspace directory

---

## Manager commands

These commands run **locally** (they do NOT go to the LLM agents):

- `help` or `:help` — print help
- `files` or `:files` — list all files in the active project folder
- `status` or `:status` — show a status snapshot of files (sizes, mtimes, hashes)
- `:new <project_name>` — reset agents and switch to a fresh workspace folder
- `EXIT` / `QUIT` — stop the program

Anything else you type is treated as a new **task** and will be sent to the Developer+Tester loop.

---

## Important usage rule (paths)

**Do not prefix paths with `CODE/`.**  
The active workspace root is already `CODE_projects/<project>/`, so writing `main.py` means:

`CODE_projects/<project>/main.py`

If you ask for `CODE/main.py`, the system will try to create:

`CODE_projects/<project>/CODE/main.py`

…and that will fail unless you explicitly allow creating that subfolder.

Recommended Manager style:
- Ask: “Create `main.py` and save it in the workspace.”
- Expect Developer to hand off: `FILE: main.py`

---

## Logging (how it works)

This project uses Python’s logging system and writes rotating logs to `logs/agent.log`.  
AutoGen exposes structured/event logs via `EVENT_LOGGER_NAME` and trace logs via `TRACE_LOGGER_NAME`, and these can be routed to handlers like any other Python logger. [web:175][web:181]

Your `log_setup.py` is configured to:
- keep normal app logs on console + file
- route AutoGen EVENT/TRACE logs to **file-only** by default (so the console stays readable) [web:175][web:181]
- redirect Python warnings into logging via `logging.captureWarnings(True)` (emits to the `py.warnings` logger) [web:170]

---

## Known issues (from your run) + recommended fixes

### 1) Tester declared “TESTS PASSED” even when exit_code != 0
In your run, `exit_code` was `2` (“can’t open file”), but the Tester still printed `TESTS PASSED`.  
Fix this by tightening the Tester system message:

- Add: “If exit_code is not 0, you MUST output TESTS FAILED.”

Optional stronger termination:
- Change termination text to require `exit_code: 0` (example):
  - `TextMentionTermination("- exit_code: 0", sources=["Tester"])`
  - AND also require “TESTS PASSED”

### 2) Fix tool wrapper arg name mismatch
Your tools correctly use `ws_run_python_file(rel_py_path=...)`.  
If the model sometimes sends `rel_path` instead, you can make the wrapper more forgiving:

In the tool wrapper, accept both names:
- Create a second tool name `ws_run_python_file_rel_path` (alias), OR
- Add a tiny normalization tool like `ws_normalize_path(...)` and force usage.

### 3) Make the system more robust against “CODE/” prefixes
To prevent agents from accidentally using `CODE/...`, normalize tool inputs in wrappers:

- If `rel_path` starts with `CODE/` or `CODE\`, strip that prefix before calling `workspace.write_file(...)`
- Do the same for `ws_run_python_file`

This single change eliminates most “No such file or directory: ...\CODE\...” errors.

---

## Example workflow

1. Start:
   ```powershell
   python .\developer_agent.py
   ```

2. Switch to a new project:
   ```
   Manager> :new trig-plots
   ```

3. Give a task:
   ```
   Manager> Write a python script that plots sin/cos/tan from -2π to 2π and saves plot.png.
   ```

4. Verify artifacts:
   ```
   Manager> files
   Manager> status
   ```

5. Exit:
   ```
   Manager> EXIT
   ```

---

