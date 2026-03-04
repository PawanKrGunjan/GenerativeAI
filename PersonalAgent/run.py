"""
Offline Personal Identity Agent
Clean Main Entry Point
Auto-starts with default question: "Introduce yourself"
"""

import logging
import traceback
from pathlib import Path
from typing import Optional

import psycopg
from colorama import Fore, Style, init

from Agent.config import (
    DATA_DIR,
    GRAPH_DIR,
    DEFAULT_MAX_ITERATIONS,
)

from Agent.db_postgresql import (
    get_pg_connection,
    get_indexed_documents,
    setup_chat_history_table,
    setup_documents_table,
    setup_document_chunks_table,
    sync_documents_to_db,
)
#from Agent.react_agent import PersonalAssistanceAgent
from Agent.reflection_agent import PersonalAssistanceAgent
from Agent.llm import embeddings_model
from Agent.logger import setup_logger
from Agent.ui import (
    TerminalChatUI,
    clear_screen,
    help_text,
    make_thread_id,
    print_ai_reply,
    print_divider,
    print_system_msg,
    print_thinking,
    print_user_message,
)

init(autoreset=True)


# ============================================================
# ROLE SELECTION
# ============================================================

def select_role() -> str:
    roles = [
        "Friend",
        "Mentor",
        "Developer",
        "Researcher",
        "Strategist",
        "Business",
        "Therapist",
    ]

    print(f"\n{Fore.CYAN}Select Role:{Style.RESET_ALL}")
    for i, r in enumerate(roles, 1):
        print(f"  {i}. {r}")
    print("  0. Custom Role")

    while True:
        choice = input(f"\n{Fore.YELLOW}Enter choice number: {Style.RESET_ALL}").strip()

        if choice.isdigit():
            idx = int(choice)
            if 1 <= idx <= len(roles):
                return roles[idx - 1]
            elif idx == 0:
                custom = input(f"{Fore.YELLOW}Enter custom role: {Style.RESET_ALL}").strip()
                if custom:
                    return custom

        print(f"{Fore.RED}Invalid selection. Try again.{Style.RESET_ALL}")


# ============================================================
# STARTUP INTRO MESSAGE
# ============================================================

def run_startup_intro(agent, user_id, role, thread_id, log):
    """
    Runs default startup question once.
    """
    default_question = "Introduce yourself"

    print_divider()
    print_system_msg("Running startup prompt: Introduce yourself", "⚡", Fore.CYAN)

    print_thinking()

    reply = agent.run(
        user_id=user_id,
        role=role,
        text=default_question,
        thread_id=thread_id,
    )

    if reply:
        print_ai_reply(reply)
        log.info("Startup intro generated")
    else:
        print_system_msg("Startup intro failed", "✗", Fore.RED)
        log.warning("Startup intro failed")

    print_divider()


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print_divider(78, "═", Fore.CYAN)
    print(f"{Fore.CYAN}  Offline Personal Identity Agent  {Style.RESET_ALL}")
    print_divider(78, "═", Fore.CYAN)

    print(help_text())
    print_divider(78, "─", Fore.YELLOW)

    user_id = "PawanKrGunjan"
    print(f"{Fore.GREEN}User: {user_id}{Style.RESET_ALL}")

    role = select_role()
    print(f"{Fore.GREEN}Active Role: {role}{Style.RESET_ALL}")

    log = setup_logger(user_id)
    log.info("Starting Agent | user=%s | role=%s", user_id, role)

    pg_conn: Optional[psycopg.Connection] = None
    agent: Optional[PersonalAssistanceAgent] = None

    try:
        # -------------------------------------------------
        # DATABASE SETUP
        # -------------------------------------------------
        print_system_msg("Connecting to PostgreSQL...", "🛢️", Fore.YELLOW)
        pg_conn = get_pg_connection(log)

        setup_chat_history_table(pg_conn, log)
        setup_documents_table(pg_conn, log)
        setup_document_chunks_table(pg_conn, log)

        print_system_msg("Syncing documents...", "📚", Fore.YELLOW)
        sync_documents_to_db(pg_conn, log)

        print_system_msg("Database ready.", "✓", Fore.GREEN)

        # -------------------------------------------------
        # AGENT INITIALIZATION
        # -------------------------------------------------
        agent = PersonalAssistanceAgent(
            pg_conn=pg_conn,
            embeddings_model=embeddings_model,
            logger=log,
            max_iterations=DEFAULT_MAX_ITERATIONS
        )

        log.info("Agent initialized")

        # Save graph (optional, useful in dev)
        agent.save_graph_visualization(Path(GRAPH_DIR), save_png=False)

        # -------------------------------------------------
        # SESSION SETUP
        # -------------------------------------------------
        ui = TerminalChatUI(user_id=user_id, log=log)
        thread_id = make_thread_id(user_id)

        log.info("Session started | thread=%s | role=%s", thread_id, role)

        # -------------------------------------------------
        # AUTO INTRO MESSAGE
        # -------------------------------------------------
        run_startup_intro(agent, user_id, role, thread_id, log)

        # -------------------------------------------------
        # CHAT LOOP
        # -------------------------------------------------
        while True:

            try:
                user_input = ui.prompt(thread_id)
            except (EOFError, KeyboardInterrupt):
                print(f"\n{Fore.MAGENTA}Goodbye 👋{Style.RESET_ALL}")
                break

            if not user_input:
                continue

            text = user_input.strip()
            cmd = text.lower()

            # -------------------------------------------------
            # COMMANDS
            # -------------------------------------------------

            if cmd in {"/exit", "exit", "quit", "q"}:
                print(f"{Fore.MAGENTA}Shutting down...{Style.RESET_ALL}")
                break

            elif cmd in {"/help", "help", "?"}:
                print("\n" + help_text() + "\n")
                continue

            elif cmd in {"/clear", "clear"}:
                clear_screen()
                continue

            elif cmd in {"/new", "new"}:
                thread_id = make_thread_id(user_id)
                print(f"\n{Fore.CYAN}→ New session: {thread_id}{Style.RESET_ALL}\n")
                run_startup_intro(agent, user_id, role, thread_id, log)
                continue

            elif cmd.startswith("/role "):
                role = text.split("/role ", 1)[1].strip()
                print(f"{Fore.CYAN}→ Role changed to: {role}{Style.RESET_ALL}")
                continue

            elif cmd in {"/sync", "sync"}:
                print_system_msg("Re-indexing documents...", "🔄", Fore.YELLOW)
                sync_documents_to_db(pg_conn, log)
                print_system_msg("Done.", "✓", Fore.GREEN)
                continue

            elif cmd in {"/docs", "docs"}:
                paths = get_indexed_documents(pg_conn, log)
                if not paths:
                    print_system_msg("No documents indexed.", "📂", Fore.YELLOW)
                else:
                    print(f"{Fore.CYAN}Indexed documents:{Style.RESET_ALL}")
                    for p in paths:
                        print(f"  • {Path(p).name}")
                continue

            # -------------------------------------------------
            # NORMAL CHAT
            # -------------------------------------------------

            print_user_message(text)
            print_thinking()

            reply = agent.run(
                user_id=user_id,
                role=role,
                text=text,
                thread_id=thread_id,
            )

            if reply:
                print_ai_reply(reply)
                log.info("AI reply (%d chars)", len(reply))
            else:
                print_system_msg("No response generated", "✗", Fore.RED)
                log.warning("No reply generated")

            print_divider()

    except Exception:
        log.error("Fatal error:\n%s", traceback.format_exc())
        print_system_msg("Fatal error occurred", "✋", Fore.RED)

    finally:
        if pg_conn:
            try:
                pg_conn.close()
                log.info("PostgreSQL connection closed")
            except Exception:
                log.warning("Failed to close DB cleanly")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{Fore.MAGENTA}Interrupted. Bye!{Style.RESET_ALL}")
    except Exception:
        logging.error("Fatal:\n%s", traceback.format_exc())
        print(f"\n{Fore.RED}Fatal error. Check logs.{Style.RESET_ALL}")
        exit(1)