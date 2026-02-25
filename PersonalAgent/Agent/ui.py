"""
Terminal user interface using prompt_toolkit + colorama enhancements
"""

import logging
import os
import re
import time
import uuid
from datetime import datetime
from typing import Optional

from colorama import Fore, Style, init
from prompt_toolkit import PromptSession
from prompt_toolkit.application import run_in_terminal
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding.bindings.basic import load_basic_bindings

from .config import HISTORY_DIR

init(autoreset=True)
# ────────────────────────────────────────────────
# Utility functions
# ────────────────────────────────────────────────


def normalize_user(raw: str) -> str:
    """Sanitize username → safe for filenames & display"""
    s = (raw or "").strip()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-zA-Z0-9_.-]", "", s)
    return s or "anonymous"


def make_thread_id(user_id: str) -> str:
    """Generate unique thread ID with timestamp + short uuid"""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = uuid.uuid4().hex[:8]
    return f"{user_id}:{ts}:{suffix}"


def clear_screen() -> None:
    """Cross-platform clear terminal"""
    os.system("cls" if os.name == "nt" else "clear")


def print_divider(length: int = 70, char: str = "─", color=Fore.MAGENTA) -> None:
    """Print a colored horizontal divider"""
    print(f"{color}{char * length}{Style.RESET_ALL}")


def print_welcome_banner(user_id: str) -> None:
    """Display startup banner"""
    print_divider(78, "═", Fore.CYAN)
    print(f"{Fore.CYAN}  Agent  •  Personal Memory + RAG Assistant  •  User: {user_id}")
    print_divider(78, "═", Fore.CYAN)
    print(
        f"{Fore.YELLOW}Commands:  /help  /sync  /docs  /new  /clear  /exit{Style.RESET_ALL}\n"
    )


def help_text() -> str:
    """Help message content (used by /help and F1)"""
    return f"""{Fore.GREEN}Available Commands:{Style.RESET_ALL}
  {Fore.BLUE}/help{Style.RESET_ALL}     Show this help message
  {Fore.BLUE}/sync{Style.RESET_ALL}     Re-index documents in ./data
  {Fore.BLUE}/docs{Style.RESET_ALL}      List all indexed documents
  {Fore.BLUE}/new{Style.RESET_ALL}       Start a new conversation thread
  {Fore.BLUE}/clear{Style.RESET_ALL}     Clear the screen
  {Fore.BLUE}/exit{Style.RESET_ALL}      Quit the application

{Fore.GREEN}Keyboard Shortcuts:{Style.RESET_ALL}
  Enter         → new line (multiline mode)
  Alt+Enter     → send message
  Up / Down     → browse history
  Ctrl+L        → clear screen
  F1            → show help
"""


def print_help() -> None:
    """Display formatted help panel"""
    print(f"{Fore.CYAN}┌{'─' * 60}┐{Style.RESET_ALL}")
    print(
        f"{Fore.CYAN}│{' ' * 18}Agent Commands & Shortcuts{' ' * 18}│{Style.RESET_ALL}"
    )
    print(f"{Fore.CYAN}└{'─' * 60}┘{Style.RESET_ALL}\n")
    print(help_text())
    print_divider()


# ────────────────────────────────────────────────
# Main Chat UI Class
# ────────────────────────────────────────────────


class TerminalChatUI:
    """Interactive terminal chat interface with history & keybindings"""

    def __init__(self, *, user_id: str, log: logging.Logger):
        self.user_id = user_id
        self.log = log

        history_file = HISTORY_DIR / f"{user_id}.history.txt"
        self.session = PromptSession(
            history=FileHistory(str(history_file)),
            auto_suggest=AutoSuggestFromHistory(),
            enable_history_search=True,
        )

        self.bindings = load_basic_bindings()

        # Help key (F1)
        @self.bindings.add("f1")
        def _(event):
            run_in_terminal(print_help)

        # Clear screen (Ctrl+L)
        @self.bindings.add("c-l")
        def _(event):
            run_in_terminal(clear_screen)

        # Submit multiline input with Alt+Enter
        @self.bindings.add("escape", "enter")
        def _(event):
            text = event.app.current_buffer.text
            self.session.history.append_string(text)
            event.app.exit(result=text)

        # Backspace / Delete fix for multiline mode
        @self.bindings.add("backspace")
        @self.bindings.add("c-h")  # Ctrl+H = backspace on many terminals
        def _(event):
            buffer = event.app.current_buffer
            if buffer.document.text_before_cursor:
                buffer.delete_before_cursor(count=1)

        @self.bindings.add("delete")
        def _(event):
            event.app.current_buffer.delete(count=1)

    def prompt(self, thread_id: str) -> Optional[str]:
        """Display prompt and return user input (stripped)"""
        try:
            prompt_str = f"You[{thread_id}] > "
            continuation = "... "

            raw = self.session.prompt(
                message=prompt_str,
                multiline=True,
                key_bindings=self.bindings,
                prompt_continuation=continuation,
            )
            return (raw or "").strip()
        except (EOFError, KeyboardInterrupt):
            return None


# ────────────────────────────────────────────────
# Output Helpers (used in run.py)
# ────────────────────────────────────────────────


def print_thinking() -> None:
    """Simple animated thinking indicator"""
    print(f"{Fore.YELLOW}Agent thinking", end="", flush=True)
    for _ in range(4):
        time.sleep(0.25)
        print(".", end="", flush=True)
    print("\r" + " " * 30 + "\r", end="", flush=True)


def print_user_message(text: str) -> None:
    """Format user message with box"""
    print(
        f"{Fore.BLUE}┌──── You ────────────────────────────────────────────────┐{Style.RESET_ALL}"
    )
    print(f"{Fore.BLUE}│{Style.RESET_ALL}  {text}")
    print(
        f"{Fore.BLUE}└─────────────────────────────────────────────────────────┘{Style.RESET_ALL}\n"
    )


def print_ai_reply(reply: str) -> None:
    """Format AI response with box"""
    print(
        f"{Fore.GREEN}┌──── Agent ───────────────────────────────────────────────┐{Style.RESET_ALL}"
    )
    wrapped = reply.replace("\n", f"\n{Fore.GREEN}│{Style.RESET_ALL}  ")
    print(f"{Fore.GREEN}│{Style.RESET_ALL}  {wrapped}")
    print(
        f"{Fore.GREEN}└──────────────────────────────────────────────────────────┘{Style.RESET_ALL}\n"
    )


def print_system_msg(msg: str, emoji: str = "ℹ️", color=Fore.YELLOW) -> None:
    """Print system/info/error message"""
    print(f"{color}{emoji} {msg}{Style.RESET_ALL}\n")
