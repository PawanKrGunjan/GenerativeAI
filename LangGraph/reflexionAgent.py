# ref.py
import json
from pprint import pprint
from typing import Annotated, Dict, Any, List, Optional, Literal
from typing_extensions import TypedDict, NotRequired

from pydantic import BaseModel, Field

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from langchain_community.tools import DuckDuckGoSearchResults


# =============================
# CONFIG
# =============================
DEBUG = True
PRINT_GRAPH_DEBUG_EVENTS = True
MAX_ITERATIONS = 4

MODEL_NAME = "granite4:350m" #"qwen2.5:7b"  # change if needed (e.g., "llama3.1:8b")
llm = ChatOllama(model=MODEL_NAME, temperature=0.0)

ddg_tool = DuckDuckGoSearchResults(num_results=3)


def log(title: str, obj: Any = None) -> None:
    if not DEBUG:
        return
    print(f"\n========== {title} ==========")
    if obj is not None:
        if isinstance(obj, (dict, list)):
            pprint(obj, width=120)
        else:
            print(obj)


def dump_model(m: Any) -> dict:
    # Works for pydantic v1 and v2
    if hasattr(m, "model_dump"):
        return m.model_dump()
    if hasattr(m, "dict"):
        return m.dict()
    return {"value": str(m)}


# =============================
# PROMPTS
# =============================
# Put your long persona/system prompt here if you want.
prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are Dr. Paul Saladino, "Carnivore MD," advocating for animal-based nutrition and challenging plant-centric dietary dogma.
Your response must follow these steps:
1. {first_instruction}
2. Present the evolutionary and biochemical rationale for animal-based nutrition.
3. Challenge conventional "plants are healthy" narratives with mechanistic evidence.
4. Reflect and critique your answer.
5. After the reflection, list 1-3 search queries separately.

Be concise and clear.""",
        ),
        MessagesPlaceholder(variable_name="messages"),
        ("system", "Answer the user's question above using the required format."),
    ]
)

first_responder_prompt = prompt_template.partial(
    first_instruction="Provide a detailed ~250 word answer."
)

revise_instructions = """Revise your previous answer using the new information.
- Keep under 500 words.
- Acknowledge individual variability and limits of evidence.
- Add a References section at the bottom (URLs).
- Use the critique to remove speculation.
"""
revisor_prompt = prompt_template.partial(first_instruction=revise_instructions)


# =============================
# SCHEMAS
# =============================
class Reflection(BaseModel):
    missing: str = Field(description="What information is missing")
    superfluous: str = Field(description="What information is unnecessary")


class AnswerQuestion(BaseModel):
    answer: str = Field(description="Main response to the question")
    reflection: Reflection = Field(description="Self-critique of the answer")
    search_queries: List[str] = Field(description="Queries for additional research")


class ReviseAnswer(AnswerQuestion):
    references: List[str] = Field(description="Citations motivating your updated answer.")


# =============================
# STATE
# =============================
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    iterations: NotRequired[int]
    draft: NotRequired[AnswerQuestion]
    search_results: NotRequired[Dict[str, Any]]
    revision: NotRequired[ReviseAnswer]


def _latest_structured(state: AgentState) -> Optional[BaseModel]:
    return state.get("revision") or state.get("draft")


def _render_references(refs: List[str]) -> str:
    if not refs:
        return ""
    lines = ["\nReferences:"]
    for i, url in enumerate(refs, start=1):
        lines.append(f"[{i}] {url}")
    return "\n".join(lines)


# =============================
# NODES
# =============================
def respond_node(state: AgentState) -> dict:
    log("NODE respond_node (input state keys)", list(state.keys()))
    log("NODE respond_node (messages count)", len(state["messages"]))
    log("NODE respond_node (iterations)", state.get("iterations", 0))

    responder = llm.with_structured_output(AnswerQuestion)
    structured: AnswerQuestion = (first_responder_prompt | responder).invoke(
        {"messages": state["messages"]}
    )

    log("Responder structured output", dump_model(structured))

    # IMPORTANT: only put Message objects in messages
    return {
        "draft": structured,
        "messages": [AIMessage(content=structured.answer)],
        "iterations": state.get("iterations", 0),
    }


def execute_tools_node(state: AgentState) -> dict:
    log("NODE execute_tools_node (iterations before)", state.get("iterations", 0))

    latest = _latest_structured(state)
    queries = getattr(latest, "search_queries", []) if latest else []

    log("Tool queries", queries)

    results: Dict[str, Any] = {}
    for q in queries:
        log("DDG invoking query", q)
        results[q] = ddg_tool.invoke(q)

    log("Tool results (raw)", results)

    return {
        "search_results": results,
        "iterations": state.get("iterations", 0) + 1,
    }


def revisor_node(state: AgentState) -> dict:
    log("NODE revisor_node (iterations)", state.get("iterations", 0))

    tool_blob = json.dumps(state.get("search_results", {}), ensure_ascii=False)
    log("Search results JSON length", len(tool_blob))

    msgs = list(state["messages"]) + [
        HumanMessage(content=f"External search results (JSON):\n{tool_blob}")
    ]

    reviser = llm.with_structured_output(ReviseAnswer)
    revised: ReviseAnswer = (revisor_prompt | reviser).invoke({"messages": msgs})

    log("Revisor structured output", dump_model(revised))

    final_text = revised.answer + _render_references(revised.references)
    return {
        "revision": revised,
        "messages": [AIMessage(content=final_text)],
    }


# =============================
# ROUTING
# =============================
def route_after_respond(state: AgentState) -> Literal["execute_tools", "revisor"]:
    iters = state.get("iterations", 0)
    draft = state.get("draft")
    has_queries = bool(draft and getattr(draft, "search_queries", []))
    log("ROUTE after respond", {"iterations": iters, "has_queries": has_queries})

    if has_queries and iters < MAX_ITERATIONS:
        return "execute_tools"
    return "revisor"


def route_after_revisor(state: AgentState) -> Literal["execute_tools", END]:
    iters = state.get("iterations", 0)
    rev = state.get("revision")
    has_queries = bool(rev and getattr(rev, "search_queries", []))
    log("ROUTE after revisor", {"iterations": iters, "has_queries": has_queries})

    if has_queries and iters < MAX_ITERATIONS:
        return "execute_tools"
    return END


# =============================
# BUILD GRAPH
# =============================
builder = StateGraph(AgentState)

builder.add_node("respond", respond_node)
builder.add_node("execute_tools", execute_tools_node)
builder.add_node("revisor", revisor_node)

builder.add_edge(START, "respond")
builder.add_conditional_edges("respond", route_after_respond, {
    "execute_tools": "execute_tools",
    "revisor": "revisor",
})
builder.add_edge("execute_tools", "revisor")
builder.add_conditional_edges("revisor", route_after_revisor, {
    "execute_tools": "execute_tools",
    END: END,
})

app = builder.compile()


# =============================
# MAIN
# =============================
if __name__ == "__main__":
    question = (
        "I'm pre-diabetic and need to lower my blood sugar, and I have heart issues. "
        "What breakfast foods should I eat and avoid?"
    )
    input_state: AgentState = {"messages": [HumanMessage(content=question)], "iterations": 0}

    # ---- Mermaid graph output
    mermaid = app.get_graph().draw_mermaid()
    print("\n================= MERMAID GRAPH (copy into Mermaid Live Editor) =================")
    print(mermaid)

    with open("reflexionAgent.mmd", "w", encoding="utf-8") as f:
        f.write(mermaid)

    # Optional PNG render (uses Mermaid rendering; may require internet depending on method)
    try:
        png_bytes = app.get_graph().draw_mermaid_png()
        with open("reflexionAgent.png", "wb") as f:
            f.write(png_bytes)
        print("\nSaved reflexionAgent.png")
    except Exception as e:
        print("\nCould not render reflexionAgent.png:", repr(e))

    # ---- LangGraph debug streaming
    if PRINT_GRAPH_DEBUG_EVENTS:
        print("\n================= LANGGRAPH DEBUG STREAM =================")
        for event in app.stream(input_state, stream_mode="debug"):
            # This can be very verbose; keep it on while learning.
            pprint(event, width=120)

    # ---- Final result
    final_state = app.invoke(input_state)

    print("\n================= FINAL ANSWER =================")
    print(final_state["messages"][-1].content)

    print("\n================= STRUCTURED OBJECTS (LEARNING) =================")
    if "draft" in final_state:
        print("\n--- Draft (AnswerQuestion) ---")
        pprint(dump_model(final_state["draft"]), width=120)
    if "revision" in final_state:
        print("\n--- Revision (ReviseAnswer) ---")
        pprint(dump_model(final_state["revision"]), width=120)
