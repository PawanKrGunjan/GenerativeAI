#!/usr/bin/env python3
"""
Agent ─ Personal long-term memory + RAG agent
Run with: python run.py
"""

import logging
import sqlite3
import traceback
from pathlib import Path
from typing import Optional

from colorama import Fore, Style, init
from langgraph.checkpoint.sqlite import SqliteSaver

from Agent.config import DATA_DIR, MODEL_NAME
from Agent.database import (
    get_indexed_documents,
    setup_document_chunks_table,
    setup_documents_table,
    setup_facts_table,
    sync_documents_to_db,
)
from Agent.graph import build_graph, run_turn, save_graph_visualization
from Agent.logger import setup_logger
from Agent.ui import (
    TerminalChatUI,
    clear_screen,
    help_text,
    make_thread_id,
    normalize_user,
    print_ai_reply,
    print_divider,
    print_system_msg,
    print_thinking,
    print_user_message,
)

init(autoreset=True)


def main() -> None:
    print_divider(78, "═", Fore.CYAN)
    print(
        f"{Fore.CYAN}  Welcome to Agent • Personal Memory + Document RAG  {Style.RESET_ALL}"
    )
    print_divider(78, "═", Fore.CYAN)

    # Show full help on startup (once)
    print(f"{Fore.CYAN}Initial Help & Commands:{Style.RESET_ALL}")
    print(help_text())
    print_divider(78, "─", Fore.YELLOW)

    while True:
        raw = input(
            f"{Fore.YELLOW}Username (press Enter for anonymous): {Style.RESET_ALL}"
        ).strip()
        user_id = normalize_user(raw)
        if user_id:
            break
        print(
            f"{Fore.YELLOW}Please enter a username or just press Enter.{Style.RESET_ALL}\n"
        )

    log = setup_logger(user_id)
    log.info("Starting Agent | model=%s | user=%s", MODEL_NAME, user_id)

    conn: Optional[sqlite3.Connection] = None

    try:
        db_path = "agent_state.db"
        conn = sqlite3.connect(db_path, check_same_thread=False, timeout=20)
        log.info("Connected to SQLite: %s", db_path)

        setup_facts_table(conn, log)
        setup_documents_table(conn, log)
        setup_document_chunks_table(conn, log)

        log.info("Scanning and indexing documents in %s ...", DATA_DIR)
        sync_documents_to_db(conn, log)
        log.info("Document indexing complete.")

        checkpointer = SqliteSaver(conn)
        checkpointer.setup()

        graph = build_graph(checkpointer, conn, log)
        log.info("LangGraph compiled successfully")
        save_graph_visualization(graph, log)

        ui = TerminalChatUI(user_id=user_id, log=log)

        print_divider(78, "═", Fore.CYAN)
        log.info("Agent session ready • User: %s", user_id)
        print_divider(78, "═", Fore.CYAN)

        thread_id = make_thread_id(user_id)
        log.info("Session thread started: %s", thread_id)
        print_divider()

        while True:
            try:
                user_input = ui.prompt(thread_id)
            except (EOFError, KeyboardInterrupt):
                log.info("Exit requested by user")
                print(f"\n{Fore.MAGENTA}Goodbye 👋{Style.RESET_ALL}")
                break

            if user_input is None:
                continue

            text = user_input.strip()
            if not text:
                continue

            cmd = text.lower()

            if cmd in {"/exit", "exit", "quit", "q"}:
                log.info("Exit command received")
                print(f"{Fore.MAGENTA}Shutting down...{Style.RESET_ALL}")
                break

            elif cmd in {"/help", "help", "?"}:
                log.info("User requested help")
                print("\n" + help_text() + "\n")
                continue

            elif cmd in {"/clear", "clear"}:
                log.info("User requested screen clear")
                clear_screen()
                continue

            elif cmd in {"/new", "new"}:
                thread_id = make_thread_id(user_id)
                log.info("New conversation thread started: %s", thread_id)
                print(
                    f"\n{Fore.CYAN}→ New session started: {thread_id}{Style.RESET_ALL}\n"
                )
                continue

            elif cmd in {"/sync", "sync", "/index"}:
                log.info("User requested document re-index")
                print(f"{Fore.YELLOW}→ Re-indexing documents...{Style.RESET_ALL}")
                sync_documents_to_db(conn, log)
                print(f"{Fore.GREEN}→ Done.{Style.RESET_ALL}\n")
                continue

            elif cmd in {"/docs", "docs"}:
                log.info("User requested indexed documents list")
                paths = get_indexed_documents(conn)
                if not paths:
                    print_system_msg("No documents indexed yet.", "📂", Fore.YELLOW)
                else:
                    print(f"{Fore.CYAN}📄 Indexed documents:{Style.RESET_ALL}")
                    for p in paths:
                        print(f"  • {Path(p).name}")
                    print()
                continue

            # ── Normal conversation ─────────────────────────────────────────
            print_user_message(text)
            log.info("User message: %s", text)

            print_thinking()
            log.debug("Starting agent inference")

            reply = run_turn(
                graph=graph,
                log=log,
                user_id=user_id,
                thread_id=thread_id,
                text=text,
                conn=conn,
            )

            if reply:
                log.info("AI reply generated (length=%d chars)", len(reply))
                print_ai_reply(reply)
            else:
                log.warning("No reply generated from agent")
                print_system_msg("No response from agent", "✗", Fore.RED)

            print_divider()

    except sqlite3.Error as dberr:
        log.error("Database error: %s\n%s", dberr, traceback.format_exc())
        print_system_msg(f"Database error: {dberr}", "⚠️", Fore.RED)

    except Exception as exc:
        log.error("Unexpected error in main:\n%s", traceback.format_exc())
        print_system_msg(f"Unexpected error: {exc}", "✗", Fore.RED)

    finally:
        if conn is not None:
            try:
                conn.commit()
                conn.close()
                log.info("SQLite connection closed")
            except Exception as close_err:
                log.error("Error closing database:\n%s", traceback.format_exc())
                print_system_msg(f"Error closing DB: {close_err}", "⚠️", Fore.RED)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging.info("Interrupted by user. Exiting cleanly.")
        print(f"\n{Fore.MAGENTA}Interrupted. Exiting cleanly.{Style.RESET_ALL}")
    except Exception as exc:
        logging.error("Fatal top-level error:\n%s", traceback.format_exc())
        print(f"\n{Fore.RED}Fatal error: {exc}{Style.RESET_ALL}")
        print(f"{Fore.RED}{traceback.format_exc()}{Style.RESET_ALL}")
        exit(1)
