"""
chat_run.py

Async runtime for the Investment Agent.

Responsibilities
----------------
1. Maintain conversation state per thread
2. Pre-fetch news context before running the graph
3. Execute the LangGraph agent
4. Format the final response for terminal or web clients
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict

from langchain_core.messages import HumanMessage
from agents.investment_agent import InvestmentAgentState, gr, IST
from chat.response_formatter import format_final_response
from tools.tool_registry import search_recent_news


LOGGER = logging.getLogger("chat_run")

# Conversation state per thread/session
state_by_thread: dict[str, InvestmentAgentState] = {}


# ---------------------------------------------------------
# NEWS PREFETCH
# ---------------------------------------------------------

async def fetch_company_news_if_needed(
    prev_state: InvestmentAgentState,
    user_query: str,
) -> InvestmentAgentState:
    """
    Fetch recent news before the graph executes.

    Why this exists
    ---------------
    - Gives the LLM fresh news context
    - Reduces unnecessary tool calls
    - Improves reasoning quality

    News is cached for several turns to reduce API usage.
    """

    updated_state = prev_state.model_copy(deep=True)

    should_fetch = (
        len(updated_state.news) == 0
        or len(updated_state.messages) < 5
    )

    if should_fetch:
        try:
            news_result = search_recent_news.invoke({"query": user_query})

            updated_state.news[user_query] = news_result

            LOGGER.info(
                "Pre-fetched %s news items for query: %s",
                len(news_result) if isinstance(news_result, list) else 1,
                user_query,
            )

        except Exception as e:
            LOGGER.warning("News fetch failed: %s", e)
            updated_state.news[user_query] = f"News unavailable: {str(e)}"

    return updated_state


# ---------------------------------------------------------
# GRAPH EXECUTION STEP
# ---------------------------------------------------------

async def run_agent_step(
    prev_state: InvestmentAgentState,
    user_query: str,
) -> InvestmentAgentState:
    """
    Execute one full agent reasoning cycle.

    Flow
    ----
    1. Pre-fetch news
    2. Add user message
    3. Run LangGraph agent
    4. Return updated state
    """

    # 1️⃣ Pre-fetch news
    state_with_news = await fetch_company_news_if_needed(prev_state, user_query)

    # 2️⃣ Append user message
    new_messages = list(state_with_news.messages) + [
        HumanMessage(content=user_query)
    ]

    # 3️⃣ Build input state
    input_state = InvestmentAgentState(
        messages=new_messages,
        symbols=state_with_news.symbols,
        memory=state_with_news.memory,
        news=state_with_news.news,
        attempt_count=0,
        tool_calls=[],
        result=None,
        current_datetime=datetime.now(IST),
    )

    # 4️⃣ Run graph
    final_state_dict = await gr.ainvoke(input_state)

    return InvestmentAgentState(**final_state_dict)


# ---------------------------------------------------------
# MAIN CHAT HANDLER
# ---------------------------------------------------------

async def handle_user_message(
    thread_id: str,
    user_query: str,
) -> Dict[str, str]:
    """
    Main entrypoint for chat interaction.

    Works for:
    - terminal chatbot
    - REST API
    - web chatbot
    """

    global state_by_thread

    # Create thread state if new conversation
    if thread_id not in state_by_thread:
        state_by_thread[thread_id] = InvestmentAgentState(
            messages=[],
            symbols={},
            memory=[],
            news={},
            tool_history=[],
            result=None,
            current_datetime=datetime.now(IST),
        )

    prev_state = state_by_thread[thread_id]

    # Run agent
    final_state = await run_agent_step(prev_state, user_query)

    # Format response using your formatter
    formatted_response = format_final_response(final_state)

    # Memory debug summary
    memory_summary = (
        f"🧠 {len(final_state.symbols)} symbols | "
        f"{len(final_state.news)} news | "
        f"{len(final_state.memory)} memories"
    )

    # Save updated state
    state_by_thread[thread_id] = final_state

    return {
        "answer": formatted_response,
        "memory_summary": memory_summary,
        "thread_id": thread_id,
    }