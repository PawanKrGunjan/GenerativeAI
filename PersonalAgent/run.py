#!/usr/bin/env python3
"""
Offline Personal Identity Agent
PostgreSQL + pgvector + Role-aware memory
Run: python run.py
"""

import logging
import traceback
from pathlib import Path
from typing import Optional

import psycopg
from colorama import Fore, Style, init
from langgraph.checkpoint.memory import MemorySaver

from Agent.config import DATA_DIR, MODEL_NAME, GRAPH_DIR
from Agent.db_postgresql import (
    get_pg_connection,
    get_indexed_documents,
    setup_chat_history_table,
    setup_documents_table,
    setup_document_chunks_table,
    sync_documents_to_db,
)
from Agent.graph import build_graph, run_turn, save_graph_visualization
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


# =====================================================
# ROLE SELECTION
# =====================================================

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
    print("  0. Type Custom Role")

    while True:
        choice = input(f"\n{Fore.YELLOW}Enter choice number: {Style.RESET_ALL}").strip()

        if choice.isdigit():
            choice = int(choice)

            if 1 <= choice <= len(roles):
                return roles[choice - 1]

            elif choice == 0:
                custom = input(
                    f"{Fore.YELLOW}Enter custom role: {Style.RESET_ALL}"
                ).strip()
                if custom:
                    return custom

        print(f"{Fore.RED}Invalid selection. Try again.{Style.RESET_ALL}")


# =====================================================
# MAIN
# =====================================================

def main() -> None:

    print_divider(78, "═", Fore.CYAN)
    print(f"{Fore.CYAN}  Offline Personal Identity Agent  {Style.RESET_ALL}")
    print_divider(78, "═", Fore.CYAN)

    print(help_text())
    print_divider(78, "─", Fore.YELLOW)

    # -------------------------------------------------
    # HARDCODED USER
    # -------------------------------------------------

    user_id = "PawanKrGunjan"

    print(f"{Fore.GREEN}User: {user_id}{Style.RESET_ALL}")

    # -------------------------------------------------
    # ROLE SELECTION
    # -------------------------------------------------

    role = select_role()
    print(f"{Fore.GREEN}Active Role: {role}{Style.RESET_ALL}")

    log = setup_logger(user_id)
    log.info("Starting Agent | model=%s | user=%s | role=%s",
             MODEL_NAME, user_id, role)

    pg_conn: Optional[psycopg.Connection] = None

    try:
        # -------------------------------------------------
        # DATABASE SETUP
        # -------------------------------------------------

        pg_conn = get_pg_connection(log)   # ✅ FIXED

        log.info("Connected to PostgreSQL")

        setup_chat_history_table(pg_conn, log)
        setup_documents_table(pg_conn, log)
        setup_document_chunks_table(pg_conn, log)

        log.info("Syncing documents from %s", DATA_DIR)
        sync_documents_to_db(pg_conn, log)
        log.info("Document indexing complete")

        # -------------------------------------------------
        # GRAPH BUILD
        # -------------------------------------------------

        checkpointer = MemorySaver()

        graph = build_graph(
            checkpointer=checkpointer,
            pg_conn=pg_conn,
            embeddings_model=embeddings_model,
            logger=log,
        )

        log.info("LangGraph compiled successfully")

        save_graph_visualization(
            graph,
            GRAPH_DIR,
            logger=log,
            save_png=True,
        )

        # -------------------------------------------------
        # UI INIT
        # -------------------------------------------------

        ui = TerminalChatUI(user_id=user_id, log=log)

        thread_id = make_thread_id(user_id)

        log.info("Session started | thread=%s | role=%s",
                 thread_id, role)

        print_divider()

        # -------------------------------------------------
        # CHAT LOOP
        # -------------------------------------------------

        while True:

            try:
                user_input = ui.prompt(thread_id)
            except (EOFError, KeyboardInterrupt):
                log.info("User exit")
                print(f"\n{Fore.MAGENTA}Goodbye 👋{Style.RESET_ALL}")
                break

            if not user_input:
                continue

            text = user_input.strip()
            cmd = text.lower()

            # ---------------- COMMANDS ----------------

            if cmd in {"/exit", "exit", "quit", "q"}:
                log.info("Exit command received")
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
                log.info("New thread: %s", thread_id)
                print(f"\n{Fore.CYAN}→ New session: {thread_id}{Style.RESET_ALL}\n")
                continue

            elif cmd.startswith("/role "):
                role = text.split("/role ", 1)[1].strip()
                log.info("Role switched to: %s", role)
                print(f"{Fore.CYAN}→ Role changed to: {role}{Style.RESET_ALL}")
                continue

            elif cmd in {"/sync", "sync"}:
                print(f"{Fore.YELLOW}Re-indexing documents...{Style.RESET_ALL}")
                sync_documents_to_db(pg_conn, log)
                print(f"{Fore.GREEN}Done.{Style.RESET_ALL}")
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

            elif cmd in {"/user","user"}:
                user_id = ui.switch_user()
                log = setup_logger(user_id)
                thread_id = make_thread_id(user_id)
                continue

            elif cmd in {"/role",'role'}:
                role = ui.switch_role(role)
                continue

            # ---------------- NORMAL CHAT ----------------

            print_user_message(text)
            log.info("User (%s): %s", role, text)

            print_thinking()

            reply = run_turn(
                graph=graph,
                user=user_id,
                thread_id=thread_id,
                role=role,
                text=text,
                logger=log,
            )

            if reply:
                print_ai_reply(reply)
                log.info("AI reply (%d chars)", len(reply))
            else:
                print_system_msg("No response", "✗", Fore.RED)
                log.warning("No reply generated")

            print_divider()

    except Exception:
        log.error("Main error:\n%s", traceback.format_exc())
        print_system_msg("Fatal error occurred", "✋", Fore.RED)

    finally:
        if pg_conn:
            try:
                pg_conn.close()
                log.info("PostgreSQL connection closed")
            except Exception:
                log.warning("Failed to close DB cleanly")


# =====================================================
# ENTRY
# =====================================================

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{Fore.MAGENTA}Interrupted. Bye!{Style.RESET_ALL}")
    except Exception:
        logging.error("Fatal:\n%s", traceback.format_exc())
        print(f"\n{Fore.RED}Fatal error. Check logs.{Style.RESET_ALL}")
        exit(1)