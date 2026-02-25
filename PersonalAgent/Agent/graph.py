"""
LangGraph definition, nodes, compilation and execution logic
"""

import logging
import sqlite3
from datetime import datetime, timezone
from typing import Annotated, Dict, Optional, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from .config import GRAPH_DIR
from .database import semantic_search
from .llm import extraction_chain, llm
from .schemas import ExtractedFacts


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    user_id: str
    ltm_context: str


def chatbot(state: AgentState) -> Dict[str, list]:
    """Generate response using retrieved context + conversation history"""
    context = state.get("ltm_context", "(no context retrieved)")

    system_prompt = f"""You are a precise, memory-aware personal assistant.

USER PROFILE & LONG-TERM MEMORY:
{context}

Important rules:
- These are the USER'S OWN private documents — never redact, censor, or mask any information (salary, dates, names, numbers, etc.)
- ALWAYS ground your answers in the REFERENCE MEMORY when possible
- Quote exact values when available and mention the Source filename
- Be concise, accurate and factual
- If the information is not present in the context → clearly say: "I don't have that information in your documents."

REFERENCE MEMORY START
{context}
REFERENCE MEMORY END"""

    messages = [SystemMessage(content=system_prompt)] + state["messages"]

    response = llm.invoke(messages)
    return {"messages": [response]}


def make_retrieve_memories(
    conn: sqlite3.Connection, logger: logging.Logger | None = None
):
    def retrieve(state: AgentState) -> Dict[str, str]:
        user_id = state["user_id"]

        # Load long-term user facts
        fact_rows = conn.execute(
            "SELECT key, value, confidence FROM user_facts WHERE user_id = ? ORDER BY updated_at DESC",
            (user_id,),
        ).fetchall()

        facts_str = (
            "\n".join(
                f"{key}: {value} (conf: {float(conf):.2f})"
                for key, value, conf in fact_rows
            )
            or "(no long-term profile yet)"
        )

        if logger:
            logger.debug("Retrieved %d facts for user %s", len(fact_rows), user_id)

        # Find most recent human message for RAG
        last_human = next(
            (
                m.content
                for m in reversed(state["messages"])
                if isinstance(m, HumanMessage)
            ),
            "",
        )

        chunks = semantic_search(conn, last_human, top_k=7)
        docs_str = "\n\n".join(chunks) or "(no relevant documents found)"

        if logger:
            logger.debug(
                "Retrieved %d document chunks for query: %s",
                len(chunks),
                last_human[:60],
            )

        context = f"""USER PROFILE (long-term facts):
{facts_str}

RELEVANT DOCUMENTS (RAG):
{docs_str}"""

        return {"ltm_context": context}

    return retrieve


def make_save_memories(conn: sqlite3.Connection, logger: logging.Logger | None = None):
    """Extract and store new facts from the last user message"""

    def save(state: AgentState) -> Dict:
        last_human = next(
            (
                m.content
                for m in reversed(state["messages"])
                if isinstance(m, HumanMessage)
            ),
            None,
        )

        if not last_human or not last_human.strip():
            return {}

        try:
            extracted: ExtractedFacts = extraction_chain.invoke({"message": last_human})
        except Exception:
            if logger:
                logger.exception("Fact extraction chain failed")
            return {}

        if not extracted.facts:
            return {}

        user_id = state["user_id"]
        updates = 0

        try:
            for fact in extracted.facts:
                key = (fact.key or "").strip()
                value = (fact.value or "").strip()
                if not key or not value:
                    continue

                confidence = max(0.0, min(float(fact.confidence), 1.0))
                source = (fact.source_text or "")[:250]
                now = datetime.now(timezone.utc).isoformat()

                row = conn.execute(
                    "SELECT confidence FROM user_facts WHERE user_id = ? AND key = ?",
                    (user_id, key),
                ).fetchone()

                old_conf = float(row[0]) if row else -1.0

                if confidence > old_conf:
                    conn.execute(
                        """
                        INSERT INTO user_facts 
                            (user_id, key, value, confidence, source, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?)
                        ON CONFLICT(user_id, key) DO UPDATE SET
                            value       = excluded.value,
                            confidence  = excluded.confidence,
                            source      = excluded.source,
                            updated_at  = excluded.updated_at
                    """,
                        (user_id, key, value, confidence, source, now),
                    )
                    updates += 1

            if updates > 0:
                conn.commit()
                if logger:
                    logger.info(
                        "Saved/updated %d new or improved facts for user %s",
                        updates,
                        user_id,
                    )

        except Exception:
            if logger:
                logger.exception("Failed to save user facts")
            conn.rollback()

        return {}

    return save


def build_graph(
    checkpointer: SqliteSaver,
    conn: sqlite3.Connection,
    logger: logging.Logger | None = None,
):
    """Build and compile the LangGraph agent workflow"""
    workflow = StateGraph(state_schema=AgentState)

    workflow.add_node("retrieve_memories", make_retrieve_memories(conn, logger))
    workflow.add_node("chatbot", chatbot)
    workflow.add_node("save_memories", make_save_memories(conn, logger))

    workflow.add_edge(START, "retrieve_memories")
    workflow.add_edge("retrieve_memories", "chatbot")
    workflow.add_edge("chatbot", "save_memories")
    workflow.add_edge("save_memories", END)

    return workflow.compile(checkpointer=checkpointer)


def save_graph_visualization(graph, logger: logging.Logger | None = None) -> None:
    """Save Mermaid diagram of the graph (MD + PNG)"""
    try:
        md_path = GRAPH_DIR / "agent_graph_latest.md"
        png_path = GRAPH_DIR / "agent_graph_latest.png"

        if md_path.exists() and png_path.exists():
            return

        mermaid_code = graph.get_graph().draw_mermaid()
        md_path.write_text(f"```mermaid\n{mermaid_code}\n```", encoding="utf-8")

        png_bytes = graph.get_graph().draw_mermaid_png()
        png_path.write_bytes(png_bytes)

        if logger:
            logger.info("Graph visualization saved: %s & %s", md_path, png_path)

    except Exception as e:
        if logger:
            logger.warning("Could not save graph visualization: %s", e)


def run_turn(
    *,
    graph,
    log: logging.Logger | None = None,
    user_id: str,
    thread_id: str,
    text: str,
    conn: sqlite3.Connection,
) -> Optional[str]:
    """Execute one full agent turn and return the final AI response"""
    if log:
        log.info(
            "Starting agent turn | user=%s | thread=%s | input=%s",
            user_id,
            thread_id,
            text[:80],
        )

    config = {"configurable": {"thread_id": thread_id}}

    inputs = {
        "messages": [HumanMessage(content=text)],
        "user_id": user_id,
        "ltm_context": "",
    }

    try:
        last_ai_response = None

        for event in graph.stream(inputs, config, stream_mode="values"):
            messages = event.get("messages", [])
            if not messages:
                continue

            last_msg = messages[-1]
            if isinstance(last_msg, AIMessage):
                last_ai_response = (last_msg.content or "").strip()
                if log:
                    log.debug("Generated AI response: %s", last_ai_response[:100])

        if log:
            log.info(
                "Agent turn complete | reply length=%d", len(last_ai_response or "")
            )

        return last_ai_response

    except Exception:
        if log:
            log.exception(
                "Graph execution failed for user=%s thread=%s", user_id, thread_id
            )
        return None
