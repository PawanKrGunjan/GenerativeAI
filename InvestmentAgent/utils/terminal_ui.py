"""
Terminal user interface using prompt_toolkit + colorama
Minimal version for Investment Agent
"""

import logging
import os
import time
import uuid
from datetime import datetime
from typing import Optional

from colorama import Fore, Style, init
from prompt_toolkit import PromptSession
from prompt_toolkit.application import run_in_terminal
from prompt_toolkit.key_binding.bindings.basic import load_basic_bindings

init(autoreset=True)


# ────────────────────────────────────────────────
# Utility
# ────────────────────────────────────────────────

def make_thread_id() -> str:
    """Generate unique thread ID"""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = uuid.uuid4().hex[:6]
    return f"{ts}-{suffix}"


def clear_screen() -> None:
    os.system("cls" if os.name == "nt" else "clear")


def print_divider(length: int = 70, char: str = "─", color=Fore.MAGENTA) -> None:
    print(f"{color}{char * length}{Style.RESET_ALL}")


def help_text() -> str:
    return f"""{Fore.GREEN}Available Commands:{Style.RESET_ALL}

  {Fore.BLUE}/help{Style.RESET_ALL}     Show help
  {Fore.BLUE}/new{Style.RESET_ALL}      Start new conversation thread
  {Fore.BLUE}/clear{Style.RESET_ALL}    Clear screen
  {Fore.BLUE}/exit{Style.RESET_ALL}     Quit application

{Fore.GREEN}Keyboard Shortcuts:{Style.RESET_ALL}

  Alt+Enter    → send message
  Enter        → newline
  Ctrl+L       → clear screen
  F1           → help
"""


def print_help() -> None:
    print(f"{Fore.CYAN}┌{'─' * 60}┐{Style.RESET_ALL}")
    print(f"{Fore.CYAN}│{' ' * 20}Agent Help{' ' * 20}│{Style.RESET_ALL}")
    print(f"{Fore.CYAN}└{'─' * 60}┘{Style.RESET_ALL}\n")
    print(help_text())
    print_divider()


# ────────────────────────────────────────────────
# Terminal UI
# ────────────────────────────────────────────────

class TerminalChatUI:

    def __init__(self, log: logging.Logger):

        self.log = log
        self.thread_id = make_thread_id()

        self.session = PromptSession()
        self.bindings = load_basic_bindings()

        # Help key
        @self.bindings.add("f1")
        def _(event):
            run_in_terminal(print_help)

        # Clear screen
        @self.bindings.add("c-l")
        def _(event):
            run_in_terminal(clear_screen)

        # Submit message
        @self.bindings.add("escape", "enter")
        def _(event):
            text = event.app.current_buffer.text
            event.app.exit(result=text)

    # ─────────────────────────────

    def prompt(self) -> Optional[str]:

        try:

            prompt_str = f"You[{self.thread_id}] > "

            raw = self.session.prompt(
                message=prompt_str,
                multiline=True,
                key_bindings=self.bindings,
                prompt_continuation="... ",
            )

            return (raw or "").strip()

        except (EOFError, KeyboardInterrupt):
            return None

    # ─────────────────────────────

    def new_thread(self):

        self.thread_id = make_thread_id()

        print_system_msg(
            f"Started new conversation: {self.thread_id}",
            "🧵",
            Fore.CYAN,
        )

        return self.thread_id


# ────────────────────────────────────────────────
# Output helpers
# ────────────────────────────────────────────────

def print_thinking() -> None:

    print(f"{Fore.YELLOW}Agent thinking", end="", flush=True)

    for _ in range(4):
        time.sleep(0.25)
        print(".", end="", flush=True)

    print("\r" + " " * 30 + "\r", end="", flush=True)


def print_user_message(text: str) -> None:

    print(
        f"{Fore.BLUE}┌──── You ────────────────────────────────────────────────┐{Style.RESET_ALL}"
    )

    print(f"{Fore.BLUE}│{Style.RESET_ALL}  {text}")

    print(
        f"{Fore.BLUE}└─────────────────────────────────────────────────────────┘{Style.RESET_ALL}\n"
    )


def print_ai_reply(reply: str) -> None:

    print(
        f"{Fore.GREEN}┌──── Agent ───────────────────────────────────────────────┐{Style.RESET_ALL}"
    )

    wrapped = reply.replace("\n", f"\n{Fore.GREEN}│{Style.RESET_ALL}  ")

    print(f"{Fore.GREEN}│{Style.RESET_ALL}  {wrapped}")

    print(
        f"{Fore.GREEN}└──────────────────────────────────────────────────────────┘{Style.RESET_ALL}\n"
    )


def print_system_msg(msg: str, emoji: str = "ℹ️", color=Fore.YELLOW) -> None:

    print(f"{color}{emoji} {msg}{Style.RESET_ALL}\n")