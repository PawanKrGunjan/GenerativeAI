# tool.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Sequence, Union

import datetime as _dt
import hashlib
import logging
import os
import subprocess
import sys
import tempfile
import threading
import uuid

from log_setup import setup_logging


@dataclass
class ExecResult:
    exit_code: int
    stdout: str
    stderr: str
    command: Union[str, Sequence[str]]
    cwd: str


def _sha12(text: str, *, encoding: str = "utf-8") -> str:
    return hashlib.sha256(text.encode(encoding, errors="ignore")).hexdigest()[:12]


def sanitize_payload(payload: dict) -> dict:
    """
    Remove large/sensitive string fields from logs (e.g., code) and replace them
    with size + short hash.
    """
    p = dict(payload)

    def _strip(field: str) -> None:
        if field not in p:
            return
        v = p.get(field)
        if not isinstance(v, str):
            return
        p.pop(field, None)
        p[f"{field}_chars"] = len(v)
        p[f"{field}_sha256_12"] = _sha12(v)

    # Common fields that may contain full code
    _strip("content")
    _strip("replacement")
    _strip("text")
    return p


class CodeWorkspace:
    """
    Workspace helper that only reads/writes/executes inside code_dir.

    Features:
    - Safe path resolution (blocks escaping code_dir)
    - Optional approval gating
    - Streaming subprocess output into logs
    - Fast line-based edits (replace/insert/delete)
    - code_dir status snapshot for agents
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
        log_file: str = "agent.log",
        log_level: int = logging.INFO,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.code_dir = Path(code_dir).resolve()
        self.code_dir.mkdir(parents=True, exist_ok=True)

        self.timeout = int(timeout)
        self.require_approval = bool(require_approval)
        self.approval_fn = approval_fn or self._default_approval
        self.allow_shell = bool(allow_shell)

        self.logs_dir = Path(logs_dir).resolve()
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        self._file_lock = threading.Lock()

        base_logger = logger or setup_logging(
            logs_dir=str(self.logs_dir),
            log_file=log_file,
            level=log_level,
        )
        self.logger = base_logger.getChild("workspace")
        self.logger.setLevel(log_level)

        self.logger.info(
            "Workspace initialized | code_dir=%s | logs_dir=%s",
            self.code_dir,
            self.logs_dir,
        )

    # -------------------------
    # Approval + path safety
    # -------------------------
    def _default_approval(self, operation: str, payload: dict) -> bool:
        safe = sanitize_payload(payload)
        ans = input(f"\nREQUEST: {operation} {safe}\nApprove? (YES/NO): ").strip().lower()
        return ans in ("y", "yes")

    def _approve(self, operation: str, payload: dict) -> None:
        self.logger.info("REQUEST %s | payload=%s", operation, sanitize_payload(payload))
        if not self.require_approval:
            self.logger.info("APPROVAL bypassed (require_approval=False) | op=%s", operation)
            return
        ok = bool(self.approval_fn(operation, payload))
        self.logger.info("APPROVAL %s | op=%s", "YES" if ok else "NO", operation)
        if not ok:
            raise PermissionError(f"Manager denied operation: {operation}")

    def _resolve_inside_code(self, rel_path: Union[str, Path]) -> Path:
        p = Path(rel_path)
        if p.is_absolute():
            raise ValueError("Absolute paths are not allowed. Provide a path relative to the workspace root.")
        resolved = (self.code_dir / p).resolve()
        # Safer + clearer than manual parent checks. [web:346]
        if not resolved.is_relative_to(self.code_dir):
            raise ValueError("Path escapes workspace root (blocked).")
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
        self.logger.info("READ done | file=%s | chars=%d", path, len(content))
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
        self._approve(
            "write_file",
            {"rel_path": str(rel_path), "overwrite": bool(overwrite), "content": content},
        )

        if create_dirs:
            path.parent.mkdir(parents=True, exist_ok=True)

        if path.exists() and not overwrite:
            raise FileExistsError(f"File exists and overwrite=False: {rel_path}")

        with self._file_lock:
            path.write_text(content, encoding=encoding)

        rel_saved = str(path.relative_to(self.code_dir))
        self.logger.info(
            "WRITE done | saved=%s | chars=%d | sha12=%s",
            rel_saved,
            len(content),
            _sha12(content, encoding=encoding),
        )
        return rel_saved

    def _write_text_atomic_noapprove(
        self,
        path: Path,
        content: str,
        *,
        encoding: str,
    ) -> None:
        """
        Atomic write by writing to a temp file in the same directory and replacing.
        This reduces corruption risk during partial writes. [web:397]
        """
        path.parent.mkdir(parents=True, exist_ok=True)

        fd, tmp_name = tempfile.mkstemp(
            dir=str(path.parent),
            prefix=path.name + ".",
            suffix=".tmp",
        )
        tmp = Path(tmp_name)
        try:
            with os.fdopen(fd, "w", encoding=encoding, newline="") as f:
                f.write(content)
            tmp.replace(path)
        finally:
            # In case replace failed, try cleanup.
            try:
                if tmp.exists():
                    tmp.unlink()
            except Exception:
                pass

    def write_file_atomic(
        self,
        rel_path: Union[str, Path],
        content: str,
        *,
        encoding: str = "utf-8",
        overwrite: bool = True,
        create_dirs: bool = True,
    ) -> str:
        path = self._resolve_inside_code(rel_path)
        self._approve(
            "write_file_atomic",
            {"rel_path": str(rel_path), "overwrite": bool(overwrite), "content": content},
        )

        if create_dirs:
            path.parent.mkdir(parents=True, exist_ok=True)

        if path.exists() and not overwrite:
            raise FileExistsError(f"File exists and overwrite=False: {rel_path}")

        with self._file_lock:
            self._write_text_atomic_noapprove(path, content, encoding=encoding)

        rel_saved = str(path.relative_to(self.code_dir))
        self.logger.info(
            "WRITE_ATOMIC done | saved=%s | chars=%d | sha12=%s",
            rel_saved,
            len(content),
            _sha12(content, encoding=encoding),
        )
        return rel_saved

    # -------------------------
    # Fast line edits
    # -------------------------
    def replace_lines(
        self,
        rel_path: Union[str, Path],
        start_line: int,
        end_line: int,
        replacement: str,
        *,
        encoding: str = "utf-8",
        create_if_missing: bool = False,
    ) -> str:
        """
        Replace inclusive 1-based range [start_line, end_line] with replacement.

        Insert-only:
          start_line = N, end_line = N-1  (inserts before line N)
        """
        if start_line < 1:
            raise ValueError("start_line must be >= 1")
        if end_line < 0:
            raise ValueError("end_line must be >= 0")

        path = self._resolve_inside_code(rel_path)
        self._approve(
            "replace_lines",
            {
                "rel_path": str(rel_path),
                "start_line": int(start_line),
                "end_line": int(end_line),
                "replacement": replacement,
            },
        )

        with self._file_lock:
            if not path.exists():
                if not create_if_missing:
                    raise FileNotFoundError(f"File not found: {rel_path}")
                old_lines: list[str] = []
            else:
                old_text = path.read_text(encoding=encoding)
                old_lines = old_text.splitlines(keepends=True)

            start_i = max(0, min(start_line - 1, len(old_lines)))
            end_i = max(0, min(end_line, len(old_lines)))  # inclusive end -> exclusive slice already

            new_lines = replacement.splitlines(keepends=True)
            new_content = "".join(old_lines[:start_i] + new_lines + old_lines[end_i:])

            # Avoid double-approve by writing internally.
            self._write_text_atomic_noapprove(path, new_content, encoding=encoding)

        rel_saved = str(path.relative_to(self.code_dir))
        self.logger.info(
            "REPLACE_LINES done | saved=%s | start=%d end=%d | replacement_chars=%d | replacement_sha12=%s",
            rel_saved,
            start_line,
            end_line,
            len(replacement),
            _sha12(replacement, encoding=encoding),
        )
        return rel_saved

    def insert_lines(self, rel_path: Union[str, Path], at_line: int, text: str, *, encoding: str = "utf-8") -> str:
        return self.replace_lines(rel_path, at_line, at_line - 1, text, encoding=encoding)

    def delete_lines(self, rel_path: Union[str, Path], start_line: int, end_line: int, *, encoding: str = "utf-8") -> str:
        return self.replace_lines(rel_path, start_line, end_line, "", encoding=encoding)

    # -------------------------
    # Status snapshot
    # -------------------------
    def get_code_status(
        self,
        rel_dir: Union[str, Path] = ".",
        *,
        max_files: int = 300,
        include_sha256: bool = True,
        encoding: str = "utf-8",
    ) -> dict[str, Any]:
        target = self._resolve_inside_code(rel_dir)
        self._approve("get_code_status", {"rel_dir": str(rel_dir), "max_files": int(max_files)})

        if not target.exists():
            return {"root": str(Path(rel_dir)), "count": 0, "files": []}

        files: list[Path] = [p for p in sorted(target.rglob("*")) if p.is_file()]
        files = files[:max_files]

        out: list[dict[str, Any]] = []
        for p in files:
            rel = str(p.relative_to(self.code_dir))
            st = p.stat()

            info: dict[str, Any] = {
                "path": rel,
                "bytes": int(st.st_size),
                "mtime": _dt.datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds"),
            }

            try:
                txt = p.read_text(encoding=encoding)
                info["lines"] = int(txt.count("\n") + (1 if txt else 0))
                if include_sha256:
                    info["sha256"] = hashlib.sha256(txt.encode(encoding, errors="ignore")).hexdigest()
            except Exception:
                info["lines"] = None
                if include_sha256:
                    info["sha256"] = hashlib.sha256(p.read_bytes()).hexdigest()

            out.append(info)

        return {"root": str(Path(rel_dir)), "count": len(out), "files": out}

    # -------------------------
    # Execution (subprocess) - live streaming
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

        merged_env = dict(os.environ)
        if env:
            merged_env.update({str(k): str(v) for k, v in env.items()})

        popen_kwargs = dict(
            cwd=str(self.code_dir),
            env=merged_env,
            text=True,
            bufsize=1,  # line buffering
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        use_shell = isinstance(command, str) and self.allow_shell
        proc = subprocess.Popen(command, shell=use_shell, **popen_kwargs)

        stdout_buf: list[str] = []
        stderr_buf: list[str] = []

        t_out = threading.Thread(
            target=self._stream_pipe,
            args=(proc.stdout, logging.INFO, "STDOUT:", stdout_buf),
            daemon=True,
        )
        t_err = threading.Thread(
            target=self._stream_pipe,
            args=(proc.stderr, logging.ERROR, "STDERR:", stderr_buf),
            daemon=True,
        )
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
        if py.suffix.lower() != ".py":
            raise ValueError("run_python_file expects a .py file")
        cmd = [sys.executable, str(py)] + (list(args) if args else [])
        self.logger.info("RUN_PY_FILE %s", py)
        return self.run_command(cmd)

    def run_python_code(self, code: str, *, filename_hint: str = "snippet") -> ExecResult:
        temp_name = f"__tmp_{filename_hint}_{uuid.uuid4().hex[:8]}.py"
        self.logger.info(
            "RUN_PY_CODE -> temp file %s | chars=%d | sha12=%s",
            temp_name,
            len(code),
            _sha12(code),
        )
        self.write_file_atomic(temp_name, code, overwrite=True, create_dirs=False)
        return self.run_python_file(temp_name)

    # -------------------------
    # Optional: AutoGen executor (lazy import)
    # -------------------------
    def run_autogen_codeblock(self, language: str, code: str) -> dict[str, Any]:
        self._approve("run_autogen_codeblock", {"language": language})
        # Keep simple: run via our own python execution path.
        if language.lower() not in ("python", "py"):
            raise ValueError("Only python codeblocks are supported by this workspace helper.")
        r = self.run_python_code(code, filename_hint="autogen")
        return {"exit_code": r.exit_code, "output": r.stdout, "error": r.stderr, "cwd": r.cwd}


if __name__ == "__main__":
    # Local manual test for CodeWorkspace
    ws = CodeWorkspace(
        code_dir="CODE",
        logs_dir="logs",
        log_file="agent.log",
        require_approval=False,
        timeout=30,
    )

    print("\n1) list_files('.'):")
    print(ws.list_files("."))

    print("\n2) write_file('hello.py'):")
    hello_code = (
        "def main():\n"
        "    print('Hello from CodeWorkspace')\n"
        "\n"
        "if __name__ == '__main__':\n"
        "    main()\n"
    )
    saved = ws.write_file("hello.py", hello_code, overwrite=True)
    print("saved:", saved)

    print("\n3) read_file('hello.py'):")
    print(ws.read_file("hello.py"))

    print("\n4) replace_lines: change print text (line 2 only):")
    ws.replace_lines(
        "hello.py",
        start_line=2,
        end_line=2,
        replacement="    print('Hello UPDATED')\n",
    )
    print(ws.read_file("hello.py"))

    print("\n5) run_python_file('hello.py'):")
    r = ws.run_python_file("hello.py")
    print("exit_code:", r.exit_code)
    print("stdout (also in logs):", r.stdout.strip())
    print("stderr (also in logs):", r.stderr.strip())

    print("\n6) get_code_status('.'):")
    status = ws.get_code_status(".")
    print("count:", status["count"])
    for f in status["files"]:
        if f["path"].endswith(".py"):
            print(" -", f)

    print("\nDONE. Check logs/agent.log for streamed stdout/stderr.")
