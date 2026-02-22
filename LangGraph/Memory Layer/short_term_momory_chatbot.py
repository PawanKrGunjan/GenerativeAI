import logging
from pathlib import Path
from typing import Annotated

from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.graph.message import add_messages

# ────────────────────────────────────────────────────────────────
#  Configuration & Logging
# ────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Folder where we save graph visualizations
GRAPH_DIR = Path("graphs")
GRAPH_DIR.mkdir(exist_ok=True)


# ────────────────────────────────────────────────────────────────
#  LLM Setup
# ────────────────────────────────────────────────────────────────

llm = ChatOllama(
    model="llama3.2:1b",
    temperature=0.7,
)

# ────────────────────────────────────────────────────────────────
#  Chatbot Node
# ────────────────────────────────────────────────────────────────

def chatbot(state: MessagesState) -> dict:
    """
    Core node: sends the full accumulated message history to the LLM
    and appends the new response.

    Short-term memory works because:
    - MessagesState uses Annotated[list, add_messages]
    - add_messages appends instead of replacing
    - MemorySaver checkpoints the state per thread_id
    """
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


# ────────────────────────────────────────────────────────────────
#  Build the graph (using built-in MessagesState)
# ────────────────────────────────────────────────────────────────

workflow = StateGraph(state_schema=MessagesState)

workflow.add_node("chatbot", chatbot)
workflow.add_edge(START, "chatbot")

# Short-term memory: checkpoints saved in RAM per thread_id
memory = MemorySaver()
graph = workflow.compile(checkpointer=memory)


# ────────────────────────────────────────────────────────────────
#  Helper function to run one conversation turn
# ────────────────────────────────────────────────────────────────

def run_conversation_turn(
    thread_id: str,
    user_input: str,
    label: str = ""
) -> None:
    """
    Runs one user message → AI response cycle
    Shows that history is preserved only within the same thread_id
    """
    config = {"configurable": {"thread_id": thread_id}}

    inputs = {"messages": [("human", user_input)]}

    print(f"\n{label or 'User'}  ({thread_id}):")
    print("  " + user_input)
    print("─" * 70)

    for event in graph.stream(inputs, config, stream_mode="values"):
        messages = event.get("messages", [])
        if messages:
            last_msg = messages[-1]
            if last_msg.type == "ai":
                print("AI:")
                print("  " + last_msg.content.strip())
                print("─" * 70)

# ────────────────────────────────────────────────────────────────
#  Save graph visualization (Mermaid → PNG)
# ────────────────────────────────────────────────────────────────

try:
    png_bytes = graph.get_graph().draw_mermaid_png()
    output_file = GRAPH_DIR / "short_term_memory_chatbot.png"
    output_file.write_bytes(png_bytes)
    logger.info("Graph PNG saved: %s", output_file)
    print(f"\nGraph diagram saved → {output_file}")
except Exception as e:
    logger.warning("Failed to render Mermaid PNG: %s", e)
    print("\nCould not save graph PNG (graphviz / pygraphviz not installed?)")

# ────────────────────────────────────────────────────────────────
#  Demonstration – showing short-term memory behavior
# ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Short-term memory demo (using MemorySaver + MessagesState)\n")
    print("Key points:")
    print("• Memory is preserved only within the same thread_id")
    print("• Different thread_id → starts with empty history")
    print("• Memory lives in RAM → lost on script restart\n")

    # ── Same conversation thread ────────────────────────────────
    thread = "session_pawan_001"

    run_conversation_turn(
        thread,
        "Hi, I'm PawanGunjan and I'm from Bihar.",
        "First message"
    )

    run_conversation_turn(
        thread,
        "What is my name and where am I from?",
        "Second message – should remember"
    )

    run_conversation_turn(
        thread,
        "Tell me one interesting fact about Patna.",
        "Third message – still same thread"
    )

    # ── New thread → memory reset ───────────────────────────────
    print("\n" + "="*70)
    print("Starting NEW thread → memory should be empty again\n")

    new_thread = "session_random_002"

    run_conversation_turn(
        new_thread,
        "What do you remember about me?",
        "First message in new thread"
    )

    run_conversation_turn(
        new_thread,
        "Do we have any previous conversation history?",
        "Second message in new thread"
    )

    print("\nEnd of demo.")
    print("Short-term memory survives only within one thread_id and one Python process.")
