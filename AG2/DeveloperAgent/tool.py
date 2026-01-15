# tool.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence, Union, Any
import os
import sys
import subprocess
import uuid
import logging
from logging.handlers import RotatingFileHandler
import threading


@dataclass
class ExecResult:
    exit_code: int
    stdout: str
    stderr: str
    command: Union[str, Sequence[str]]
    cwd: str


class CodeWorkspace:
    """
    A small workspace helper that *only* reads/writes/executes inside CODE/.

    Adds logging:
    - Live console logs
    - Rotating log file under ./logs/
    """
    def __init__(
        self,
        code_dir: Union[str, Path] = "CODE",
        *,
        timeout: int = 60,
        require_approval: bool = True,
        approval_fn: Optional[Callable[[str, dict], bool]] = None,
        allow_shell: bool = False,
        logs_dir: Union[str, Path] = "logs",
        log_file: str = "workspace.log",
        log_level: int = logging.INFO,
        log_to_console: bool = True,
        rotate_max_bytes: int = 2_000_000,
        rotate_backup_count: int = 5,
    ) -> None:
        self.code_dir = Path(code_dir).resolve()
        self.code_dir.mkdir(parents=True, exist_ok=True)

        self.timeout = int(timeout)
        self.require_approval = bool(require_approval)
        self.approval_fn = approval_fn or self._default_approval
        self.allow_shell = bool(allow_shell)

        self.logs_dir = Path(logs_dir).resolve()
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.logs_dir / log_file

        self.logger = logging.getLogger(f"CodeWorkspace[{self.code_dir}]")
        self.logger.setLevel(log_level)

        # Avoid duplicate handlers if CodeWorkspace is instantiated multiple times
        if not self.logger.handlers:
            fmt = logging.Formatter(
                fmt="[%(asctime)s] [%(levelname)s] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )

            file_h = RotatingFileHandler(
                str(self.log_path),
                maxBytes=int(rotate_max_bytes),
                backupCount=int(rotate_backup_count),
                encoding="utf-8",
            )
            file_h.setFormatter(fmt)
            self.logger.addHandler(file_h)  # rotating file logs are supported via RotatingFileHandler [web:147]

            if log_to_console:
                console_h = logging.StreamHandler(sys.stdout)
                console_h.setFormatter(fmt)
                self.logger.addHandler(console_h)

        self.logger.info("Workspace initialized | code_dir=%s | logs=%s", self.code_dir, self.log_path)

    # -------------------------
    # Approval + path safety
    # -------------------------
    def _default_approval(self, operation: str, payload: dict) -> bool:
        short = dict(payload)
        if "content" in short and isinstance(short["content"], str) and len(short["content"]) > 200:
            short["content"] = short["content"][:200] + "...(truncated)"
        ans = input(f"\nREQUEST: {operation} {short}\nApprove? (YES/NO): ").strip().lower()
        return ans in ("y", "yes")

    def _approve(self, operation: str, payload: dict) -> None:
        self.logger.info("REQUEST %s | payload=%s", operation, self._safe_payload(payload))
        if not self.require_approval:
            self.logger.info("APPROVAL bypassed (require_approval=False) | op=%s", operation)
            return
        ok = bool(self.approval_fn(operation, payload))
        self.logger.info("APPROVAL %s | op=%s", "YES" if ok else "NO", operation)
        if not ok:
            raise PermissionError(f"Manager denied operation: {operation}")

    def _safe_payload(self, payload: dict) -> dict:
        # Keep logs readable; avoid dumping huge code blobs
        p = dict(payload)
        if "content" in p and isinstance(p["content"], str) and len(p["content"]) > 500:
            p["content"] = p["content"][:500] + "...(truncated)"
        return p

    def _resolve_inside_code(self, rel_path: Union[str, Path]) -> Path:
        p = Path(rel_path)
        if p.is_absolute():
            raise ValueError("Absolute paths are not allowed. Provide a path relative to CODE/.")
        resolved = (self.code_dir / p).resolve()
        if self.code_dir != resolved and self.code_dir not in resolved.parents:
            raise ValueError("Path escapes CODE/ directory (blocked).")
        return resolved

    # -------------------------
    # File operations
    # -------------------------
    def list_files(self, rel_dir: Union[str, Path] = ".") -> list[str]:
        target = self._resolve_inside_code(rel_dir)
        self._approve("list_files", {"rel_dir": str(rel_dir)})

        self.logger.info("LIST %s", target)
        if not target.exists():
            return []
        if target.is_file():
            return [str(Path(rel_dir))]

        out: list[str] = []
        for f in sorted(target.rglob("*")):
            if f.is_file():
                out.append(str(f.relative_to(self.code_dir)))
        self.logger.info("LIST done | count=%d", len(out))
        return out

    def read_file(self, rel_path: Union[str, Path], *, encoding: str = "utf-8") -> str:
        path = self._resolve_inside_code(rel_path)
        self._approve("read_file", {"rel_path": str(rel_path)})

        self.logger.info("READ %s", path)
        content = path.read_text(encoding=encoding)
        self.logger.info("READ done | chars=%d | file=%s", len(content), path)
        return content

    def write_file(
        self,
        rel_path: Union[str, Path],
        content: str,
        *,
        encoding: str = "utf-8",
        overwrite: bool = True,
        create_dirs: bool = True,
    ) -> str:
        path = self._resolve_inside_code(rel_path)
        self._approve("write_file", {"rel_path": str(rel_path), "overwrite": overwrite, "content": content})

        if create_dirs:
            path.parent.mkdir(parents=True, exist_ok=True)

        if path.exists() and not overwrite:
            raise FileExistsError(f"File exists and overwrite=False: {rel_path}")

        path.write_text(content, encoding=encoding)
        self.logger.info("WRITE done | file=%s | chars=%d", path, len(content))
        return str(path.relative_to(self.code_dir))

    # -------------------------
    # Execution (subprocess) - LIVE streaming
    # -------------------------
    def _stream_pipe(self, pipe, level: int, prefix: str, buffer: list[str]) -> None:
        try:
            for line in iter(pipe.readline, ""):
                if not line:
                    break
                buffer.append(line)
                self.logger.log(level, "%s %s", prefix, line.rstrip("\n"))
        finally:
            try:
                pipe.close()
            except Exception:
                pass

    def run_command(
        self,
        command: Union[str, Sequence[str]],
        *,
        env: Optional[dict[str, str]] = None,
    ) -> ExecResult:
        self._approve("run_command", {"command": command})

        self.logger.info("RUN start | cwd=%s | cmd=%s", self.code_dir, command)

        popen_kwargs = dict(
            cwd=str(self.code_dir),
            env=(os.environ | (env or {})),
            text=True,
            bufsize=1,  # line-buffering helps stream output line-by-line [web:154]
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # keep your original semantics: only allow shell when command is str and allow_shell=True
        use_shell = isinstance(command, str) and self.allow_shell
        proc = subprocess.Popen(command, shell=use_shell, **popen_kwargs)

        stdout_buf: list[str] = []
        stderr_buf: list[str] = []

        t_out = threading.Thread(target=self._stream_pipe, args=(proc.stdout, logging.INFO, "STDOUT:", stdout_buf))
        t_err = threading.Thread(target=self._stream_pipe, args=(proc.stderr, logging.ERROR, "STDERR:", stderr_buf))
        t_out.start()
        t_err.start()

        try:
            proc.wait(timeout=self.timeout)
        except subprocess.TimeoutExpired:
            self.logger.error("RUN timeout | killing process | cmd=%s", command)
            proc.kill()
            proc.wait()

        t_out.join(timeout=5)
        t_err.join(timeout=5)

        exit_code = int(proc.returncode) if proc.returncode is not None else -1
        stdout = "".join(stdout_buf)
        stderr = "".join(stderr_buf)

        self.logger.info("RUN end | exit_code=%s | cmd=%s", exit_code, command)

        return ExecResult(
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            command=command,
            cwd=str(self.code_dir),
        )

    def run_python_file(self, rel_py_path: Union[str, Path], args: Optional[Sequence[str]] = None) -> ExecResult:
        py = self._resolve_inside_code(rel_py_path)
        if py.suffix != ".py":
            raise ValueError("run_python_file expects a .py file")
        cmd = [sys.executable, str(py)] + (list(args) if args else [])
        self.logger.info("RUN_PY_FILE %s", py)
        return self.run_command(cmd)

    def run_python_code(self, code: str, *, filename_hint: str = "snippet") -> ExecResult:
        temp_name = f"__tmp_{filename_hint}_{uuid.uuid4().hex[:8]}.py"
        self.logger.info("RUN_PY_CODE -> temp file %s", temp_name)
        self.write_file(temp_name, code, overwrite=True, create_dirs=False)
        return self.run_python_file(temp_name)

    # -------------------------
    # Optional: AutoGen code executor
    # -------------------------
    def run_autogen_codeblock(self, language: str, code: str) -> dict[str, Any]:
        self._approve("run_autogen_codeblock", {"language": language})
        try:
            from autogen.coding import CodeBlock, LocalCommandLineCodeExecutor
        except Exception as e:
            raise RuntimeError("AutoGen coding executor not available. Install/upgrade autogen/pyautogen.") from e

        self.logger.info("AUTOGEN_EXEC start | language=%s", language)
        executor = LocalCommandLineCodeExecutor(work_dir=self.code_dir)
        result = executor.execute_code_blocks([CodeBlock(language=language, code=code)])
        out = {
            "exit_code": getattr(result, "exit_code", None),
            "output": getattr(result, "output", None),
            "code_file": getattr(result, "code_file", None),
        }
        self.logger.info("AUTOGEN_EXEC end | result=%s", out)
        return out
