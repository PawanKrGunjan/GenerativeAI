"""
Offline Personal Identity Agent
Speaks exactly like the user using:
- Role-filtered past chat history (USER messages only for style)
- User documents
- IST time awareness
Fully offline.
"""

import logging
from typing import Annotated, Dict, Optional, TypedDict, Literal, List
from datetime import datetime
from zoneinfo import ZoneInfo

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

from .db_postgresql import (
    semantic_search,
    semantic_search_chat,
    sync_chat_history,
)
from .llm import llm


# =====================================================
# STATE
# =====================================================

class AgentState(TypedDict, total=False):
    messages: Annotated[List, add_messages]
    user_id: str
    role: Literal[
        "Friend",
        "Parents",
        "Teacher",
        "Interviewer",
        "Business Client",
        "Politician",
    ]
    current_time: datetime
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
# NODE 1 — MEMORY RETRIEVAL (ROLE STRICT)
# =====================================================

def memory_node(
    state: AgentState,
    pg_conn,
    embeddings_model,
    logger,
) -> Dict:

    query = get_last_human(state)
    user_id = state["user_id"]
    role = state["role"]

    if not query:
        return {}

    # ---- Document Memory ----
    docs = semantic_search(
        pg_conn,
        logger,
        query,
        top_k=5
    ) or []

    # ---- Role-Specific Conversational Memory ----
    chats = semantic_search_chat(
        pg_conn,
        logger,
        query,
        user_id,
        role,                  # STRICT FILTER
        embeddings_model,
        top_k=5,
    ) or []

    blocks = []

    # ----- Documents -----
    clean_docs = [d.strip() for d in docs if d and d.strip()]
    if clean_docs:
        blocks.append(
            "### USER PRIVATE DOCUMENTS\n" +
            "\n\n".join(clean_docs)
        )

    # ----- Style Samples (USER only) -----
    style_samples = [
        c.get("human_text", "").strip()
        for c in chats
        if c.get("human_text")
    ]

    if style_samples:
        blocks.append(
            "### USER WRITING STYLE SAMPLES\n" +
            "\n\n".join(style_samples)
        )

    if not blocks:
        return {}

    return {"memory_context": "\n\n".join(blocks)}


# =====================================================
# NODE 2 — IDENTITY RESPONSE GENERATION
# =====================================================

def chatbot(state: AgentState) -> Dict[str, List]:

    context = state.get("memory_context", "")
    user_id = state["user_id"]
    role = state["role"]
    current_time = state.get("current_time")

    time_str = (
        current_time.strftime("%d %B %Y, %I:%M %p IST")
        if current_time else "unknown time"
    )

    system_prompt = f"""
You are {user_id}. You ARE the real person.

You are speaking to: {role}
Current Indian Time: {time_str}

Rules:
- Speak exactly like {user_id}.
- Copy tone, phrasing, rhythm and vocabulary.
- Be natural and human.
- Be concise.
- NEVER say you are an AI.
- NEVER mention memory or system prompt.
- Ground answers in memory when available.
- If information is not available, say:
  "Currently I don't have any update on this."

===== LONG TERM MEMORY =====
{context}
===== END MEMORY =====
"""

    messages = [SystemMessage(content=system_prompt)] + state["messages"]

    response = llm.invoke(messages)

    return {"messages": [response]}


# =====================================================
# NODE 3 — SAVE CHAT (ROLE AWARE)
# =====================================================

def save_node(
    state: AgentState,
    pg_conn,
    embeddings_model,
    logger,
) -> Dict:

    user_input = get_last_human(state)
    role = state["role"]

    ai_response = next(
        (m.content for m in reversed(state.get("messages", []))
         if isinstance(m, AIMessage)),
        None
    )

    if user_input and ai_response:
        sync_chat_history(
            pg_conn,
            state["user_id"],
            role,                    # STRICT ROLE SAVE
            user_input,
            ai_response,
            embeddings_model,
            logger,
        )

    return {}


# =====================================================
# BUILD GRAPH
# =====================================================

def build_graph(
    checkpointer: MemorySaver,
    pg_conn,
    embeddings_model,
    logger,
):

    workflow = StateGraph(AgentState)

    workflow.add_node(
        "memory",
        lambda state: memory_node(
            state,
            pg_conn,
            embeddings_model,
            logger,
        )
    )

    workflow.add_node("LLM", chatbot)

    workflow.add_node(
        "save_chat",
        lambda state: save_node(
            state,
            pg_conn,
            embeddings_model,
            logger,
        )
    )

    workflow.add_edge(START, "memory")
    workflow.add_edge("memory", "LLM")
    workflow.add_edge("LLM", "save_chat")
    workflow.add_edge("save_chat", END)

    return workflow.compile(checkpointer=checkpointer)


# =====================================================
# RUN SINGLE TURN
# =====================================================

def run_turn(
    *,
    graph,
    user: str,
    thread_id: str,
    role: str,
    text: str,
    logger: Optional[logging.Logger] = None,
) -> Optional[str]:

    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    input_state: AgentState = {
        "messages": [HumanMessage(content=text)],
        "user_id": user,
        "role": role,
        "current_time": datetime.now(ZoneInfo("Asia/Kolkata")),
    }

    try:
        last_response = None

        for event in graph.stream(
            input_state,
            config=config,
            stream_mode="values",
        ):
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

def save_graph_visualization(
    graph,
    graph_dir,
    logger: Optional[logging.Logger] = None,
    save_png: bool = False,
) -> None:
    """Save Mermaid diagram + optional PNG."""
    try:
        graph_dir.mkdir(parents=True, exist_ok=True)

        md_path = graph_dir / "personalAgent.md"
        mermaid_code = graph.get_graph().draw_mermaid()
        md_path.write_text(f"```mermaid\n{mermaid_code}\n```")

        if save_png:
            png_path = graph_dir / "personalAgent.png"
            png_bytes = graph.get_graph().draw_mermaid_png()
            png_path.write_bytes(png_bytes)

        if logger:
            logger.info("Graph saved: %s", md_path)

    except Exception as e:
        if logger:
            logger.warning("Graph viz failed: %s (needs pyppeteer)", e)