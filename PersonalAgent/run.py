#!/usr/bin/env python3
"""
Agent ─ Personal long-term memory + RAG agent
PostgreSQL + psycopg 3.x version
Run with: python run.py
"""

import logging
import traceback
from pathlib import Path
from typing import Optional

import psycopg
from colorama import Fore, Style, init
from langgraph.checkpoint.memory import MemorySaver

from Agent.config import DATA_DIR, MODEL_NAME
from Agent.db_postgresql import (
    get_pg_connection,
    get_indexed_documents,
    setup_chat_history_table,
    setup_facts_table,
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
    normalize_user,
    print_ai_reply,
    print_divider,
    print_system_msg,
    print_thinking,
    print_user_message,
)

init(autoreset=True)


# =====================================================
# MAIN
# =====================================================


def main() -> None:
    print_divider(78, "═", Fore.CYAN)
    print(
        f"{Fore.CYAN}  Welcome to Agent • PostgreSQL + pgvector RAG  {Style.RESET_ALL}"
    )
    print_divider(78, "═", Fore.CYAN)

    print(f"{Fore.CYAN}Initial Help & Commands:{Style.RESET_ALL}")
    print(help_text())
    print_divider(78, "─", Fore.YELLOW)

    # -------------------------------------------------
    # USER LOGIN
    # -------------------------------------------------

    while True:
        raw = input(
            f"{Fore.YELLOW}Username (Enter for anonymous): {Style.RESET_ALL}"
        ).strip()
        user_id = normalize_user(raw)
        if user_id:
            break
        print(
            f"{Fore.YELLOW}Please enter a username or press Enter.{Style.RESET_ALL}\n"
        )

    log = setup_logger(user_id)
    log.info("Starting Agent | model=%s | user=%s | DB=PostgreSQL", MODEL_NAME, user_id)

    pg_conn: Optional[psycopg.Connection] = None

    try:
        # -------------------------------------------------
        # DB SETUP
        # -------------------------------------------------

        pg_conn = get_pg_connection()
        log.info("✅ PostgreSQL + pgvector connected")

        setup_facts_table(pg_conn, log)
        setup_chat_history_table(pg_conn, log)
        setup_documents_table(pg_conn, log)
        setup_document_chunks_table(pg_conn, log)

        log.info("Scanning/indexing documents in %s...", DATA_DIR)
        sync_documents_to_db(pg_conn, log)
        log.info("✅ Document indexing complete")

        # -------------------------------------------------
        # GRAPH BUILD
        # -------------------------------------------------

        checkpointer = MemorySaver()

        graph = build_graph(
            checkpointer=checkpointer,
            pg_conn=pg_conn,
            logger=log,
            embeddings_model=embeddings_model,
        )

        log.info("✅ LangGraph compiled")
        save_graph_visualization(graph, log)

        # -------------------------------------------------
        # UI INIT
        # -------------------------------------------------

        ui = TerminalChatUI(user_id=user_id, log=log)

        print_divider(78, "═", Fore.CYAN)
        log.info("Agent ready • User: %s", user_id)
        print_divider(78, "═", Fore.CYAN)

        thread_id = make_thread_id(user_id)
        log.info("Session thread: %s", thread_id)
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
            if not text:
                continue

            cmd = text.lower()

            # ---------------- Commands ----------------

            if cmd in {"/exit", "exit", "quit", "q"}:
                log.info("Exit command received")
                print(f"{Fore.MAGENTA}Shutting down...{Style.RESET_ALL}")
                break

            elif cmd in {"/help", "help", "?"}:
                log.info("Help requested")
                print("\n" + help_text() + "\n")
                continue

            elif cmd in {"/clear", "clear"}:
                log.info("Screen clear")
                clear_screen()
                continue

            elif cmd in {"/new", "new"}:
                thread_id = make_thread_id(user_id)
                log.info("New thread: %s", thread_id)
                print(f"\n{Fore.CYAN}→ New session: {thread_id}{Style.RESET_ALL}\n")
                continue

            elif cmd in {"/sync", "sync", "/index"}:
                log.info("Re-indexing")
                print(f"{Fore.YELLOW}→ Re-indexing...{Style.RESET_ALL}")
                sync_documents_to_db(pg_conn, log)
                print(f"{Fore.GREEN}→ Done.{Style.RESET_ALL}\n")
                continue

            elif cmd in {"/docs", "docs"}:
                log.info("List docs")
                paths = get_indexed_documents(pg_conn, log)

                if not paths:
                    print_system_msg("No documents indexed.", "📂", Fore.YELLOW)
                else:
                    print(f"{Fore.CYAN}📄 Indexed docs:{Style.RESET_ALL}")
                    for p in paths:
                        print(f"  • {Path(p).name}")
                    print()
                continue

            # ---------------- Normal Chat ----------------

            print_user_message(text)
            log.info("User: %s", text)

            print_thinking()
            log.debug("Agent inference")

            reply = run_turn(
                graph=graph,
                user_id=user_id,
                thread_id=thread_id,
                text=text,
                logger=log,
            )

            if reply:
                log.info("AI reply (%d chars)", len(reply))
                print_ai_reply(reply)
            else:
                log.warning("No reply")
                print_system_msg("No response", "✗", Fore.RED)

            print_divider()

    except Exception as exc:
        log.error("Main error:\n%s", traceback.format_exc())
        print_system_msg(f"Error: {exc}", "✋", Fore.RED)

    finally:
        if pg_conn:
            pg_conn.close()
            log.info("PostgreSQL closed")


# =====================================================
# ENTRY
# =====================================================

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{Fore.MAGENTA}Interrupted. Bye!{Style.RESET_ALL}")
    except Exception as exc:
        logging.error("Fatal:\n%s", traceback.format_exc())
        print(f"\n{Fore.RED}Fatal: {exc}\n{traceback.format_exc()}{Style.RESET_ALL}")
        exit(1)
