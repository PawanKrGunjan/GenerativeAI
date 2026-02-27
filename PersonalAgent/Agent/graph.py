"""
Offline Personal Identity Agent
Speaks exactly like the user using:
- Past chat history (USER messages only for style)
- User documents
Fully offline.
"""

import logging
from typing import Annotated, Dict, Optional, TypedDict

import psycopg
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

from .config import GRAPH_DIR
from .db_postgresql import (
    semantic_search,
    semantic_search_chat,
    sync_chat_history,
)
from .llm import llm

# =====================================================
# STATE
# =====================================================


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    user_id: str
    memory_context: str


# =====================================================
# HELPERS
# =====================================================


def get_last_human(state: AgentState) -> Optional[str]:
    for m in reversed(state.get("messages", [])):
        if isinstance(m, HumanMessage):
            return m.content.strip()
    return None


# =====================================================
# NODE 1 — RETRIEVE MEMORY
# =====================================================


def create_memory_node(pg_conn: psycopg.Connection, logger):

    def memory_node(state: AgentState) -> Dict:

        query = get_last_human(state)
        user_id = state["user_id"]

        if not query:
            return {}

        # ---- Retrieve relevant documents ----
        docs = semantic_search(pg_conn, logger, query, top_k=5)

        # ---- Retrieve past chats ----
        chats = semantic_search_chat(pg_conn, logger, query, user_id, top_k=5)

        blocks = []

        # Add document context
        if docs:
            blocks.append("=== USER DOCUMENTS ===\n" + "\n\n".join(docs))

        # Add ONLY USER messages (for style learning)
        if chats:
            user_messages = "\n\n".join(c["human_text"] for c in chats)
            blocks.append(
                "=== USER WRITING STYLE (PAST MESSAGES) ===\n" + user_messages
            )

        return {"memory_context": "\n\n".join(blocks)}

    return memory_node


# =====================================================
# NODE 2 — GENERATE USER-IDENTITY RESPONSE
# =====================================================


def create_identity_response_node(logger):

    def chatbot(state: AgentState) -> Dict[str, list]:

        context = state.get("memory_context", "(no context retrieved)")
        user_id = state["user_id"]

        system_prompt = f"""You are here representing on behalf of {user_id}. 
Behave exactly like {user_id} same like HUMAN. 

You can read about the user's LONG-TERM MEMORY, 
previous chat history, and private documents below.

Important rules:
- You are {user_id}.
- These are {user_id}'s OWN private documents — never redact, censor, or mask information.
- ALWAYS ground answers in REFERENCE MEMORY when available.
- Match writing tone, vocabulary, and style from past messages.
- Be concise, natural and human.
- DO NOT mention system prompts or memory.
- If information is not present in context → say:
  "Currently I don't have any update on this."

REFERENCE MEMORY START
{context}
REFERENCE MEMORY END
"""

        messages = [SystemMessage(content=system_prompt)] + state["messages"]

        response = llm.invoke(messages)

        return {"messages": [response]}

    return chatbot


# =====================================================
# NODE 3 — SAVE CHAT
# =====================================================


def create_save_node(pg_conn, embeddings_model, logger):

    def save_node(state: AgentState) -> Dict:

        user_input = get_last_human(state)
        ai_response = state["messages"][-1].content

        if user_input:
            sync_chat_history(
                pg_conn,
                state["user_id"],
                user_input,
                ai_response,
                embeddings_model,
                logger,
            )

        return {}

    return save_node


# =====================================================
# BUILD GRAPH
# =====================================================


def build_graph(
    checkpointer: MemorySaver,
    pg_conn: psycopg.Connection,
    logger: Optional[logging.Logger],
    embeddings_model,
):

    workflow = StateGraph(AgentState)

    workflow.add_node("memory", create_memory_node(pg_conn, logger))
    workflow.add_node("LLM", create_identity_response_node(logger))
    workflow.add_node("save_chat", create_save_node(pg_conn, embeddings_model, logger))

    workflow.add_edge(START, "memory")
    workflow.add_edge("memory", "LLM")
    workflow.add_edge("LLM", "save_chat")
    workflow.add_edge("save_chat", END)

    return workflow.compile(checkpointer=checkpointer)


# =====================================================
# RUN TURN
# =====================================================


def run_turn(
    *,
    graph,
    user_id: str,
    thread_id: str,
    text: str,
    logger: Optional[logging.Logger] = None,
) -> Optional[str]:

    config = {"configurable": {"thread_id": thread_id}}

    inputs = {
        "messages": [HumanMessage(content=text)],
        "user_id": user_id,
        "memory_context": "",
    }

    try:
        last_response = None

        for event in graph.stream(inputs, config, stream_mode="values"):
            msgs = event.get("messages", [])
            if msgs and isinstance(msgs[-1], AIMessage):
                last_response = msgs[-1].content.strip()

        return last_response

    except Exception:
        if logger:
            logger.exception("Graph execution failed")
        return None


# =====================================================
# SAVE GRAPH VISUALIZATION (DO NOT REMOVE)
# =====================================================


def save_graph_visualization(graph, logger: Optional[logging.Logger] = None) -> None:
    """Save Mermaid diagram + PNG."""
    try:
        md_path = GRAPH_DIR / "personalAgent.md"
        png_path = GRAPH_DIR / "personalAgent.png"

        mermaid_code = graph.get_graph().draw_mermaid()
        md_path.write_text(f"```mermaid\n{mermaid_code}\n```")

        png_bytes = graph.get_graph().draw_mermaid_png()
        png_path.write_bytes(png_bytes)

        if logger:
            logger.info("Graph saved: %s + %s", md_path, png_path)

    except Exception as e:
        if logger:
            logger.warning("Graph viz failed: %s (needs pyppeteer)", e)
