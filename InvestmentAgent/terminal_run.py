"""
terminal_run.py
Main entry point for Investment Agent CLI
"""

from colorama import Fore, Style, init
import asyncio
from utils.logger import LOGGER
from utils.db_connect import initialize_database
from utils.terminal_ui import (
    TerminalChatUI,
    clear_screen,
    help_text,
    print_ai_reply,
    print_divider,
    print_system_msg,
    print_thinking,
    print_user_message,
)

#from agents.investment_agent_AI import agent
#from agents.investment_agent_base import InvestmentAgentState, gr, IST
from agents.investment_agent import InvestmentAgentState, gr, IST
from chat.chat_run import handle_user_message

init(autoreset=True)

CURRENT_THREAD_ID = 'SriGanesh'
# ────────────────────────────────────────────────
# ENVIRONMENT SETUP
# ────────────────────────────────────────────────

def setup_environment():
    """Initialize required services"""
    LOGGER.info("Initializing environment")
    initialize_database(LOGGER)


# ────────────────────────────────────────────────
# COMMAND HANDLER
# ────────────────────────────────────────────────
def handle_command(cmd: str, ui: TerminalChatUI) -> bool:
    """
    Handle system commands.

    Returns True if command was handled.
    """

    if cmd in {"/exit", "exit", "quit", "q"}:
        print(f"{Fore.MAGENTA}Shutting down...{Style.RESET_ALL}")
        raise SystemExit

    if cmd in {"/help", "help", "?"}:
        print("\n" + help_text() + "\n")
        return True

    if cmd in {"/clear", "clear"}:
        clear_screen()
        return True

    if cmd in {"/new"}:
        ui.new_thread()
        return True

    return False


# ────────────────────────────────────────────────
# MAIN LOOP
# ────────────────────────────────────────────────
def main():
    setup_environment()
    print_divider(78, "═", Fore.CYAN)
    print(f"{Fore.CYAN}  Indian Stock Market AI Agent{Style.RESET_ALL}")
    print_divider(78, "═", Fore.CYAN)

    print(help_text())
    print_divider(78, "─", Fore.YELLOW)

    ui = TerminalChatUI(log=LOGGER)

    while True:

        try:
            user_input = ui.prompt()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{Fore.MAGENTA}Goodbye 👋{Style.RESET_ALL}")
            break

        if not user_input:
            continue

        text = user_input.strip()
        cmd = text.lower()

        # -------------------------------------------------
        # SYSTEM COMMANDS
        # -------------------------------------------------

        try:
            if handle_command(cmd, ui):
                continue
        except SystemExit:
            break

        # -------------------------------------------------
        # USER MESSAGE
        # -------------------------------------------------

        print_user_message(text)

        try:

            print_thinking()

            #result = agent.run(query=text)
            result = asyncio.run(handle_user_message(
                thread_id=CURRENT_THREAD_ID,
                user_query=text
            ))
        except Exception as e:

            LOGGER.exception("Agent execution failed")

            print_system_msg(
                f"Agent error: {str(e)}",
                "x",
                Fore.RED,
            )

            continue

        # -------------------------------------------------
        # RESPONSE
        # -------------------------------------------------

        if not result:

            LOGGER.warning("Empty agent response")

            print_system_msg(
                "No response generated",
                "!",
                Fore.YELLOW,
            )

            continue

        answer = result.get("answer")

        if answer:

            LOGGER.info("AI reply (%d chars)", len(answer))
            print_ai_reply(answer)

        memory_summary = result.get("memory_summary")

        if memory_summary:

            print_ai_reply(memory_summary)

        print_divider()


# ────────────────────────────────────────────────

if __name__ == "__main__":
    main()