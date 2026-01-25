from __future__ import annotations

import json
import logging
from typing import Annotated, Dict, Any, List, Optional, Literal
from typing_extensions import TypedDict, NotRequired
from pathlib import Path

from pydantic import BaseModel, Field

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from langchain_community.tools import DuckDuckGoSearchResults

import mlflow
from logger_config import setup_logger


# =============================================================================
# MLFLOW
# =============================================================================
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("reflexion-agent")
mlflow.langchain.autolog()


# =============================
# CONFIG
# =============================
DEBUG = True
PRINT_GRAPH_DEBUG_EVENTS = True
MAX_ITERATIONS = 4

MODEL_NAME = "granite4:350m"

# Logger (rotating file + console)
logger = setup_logger(
    debug_mode=DEBUG,
    log_name="reflexion-agent",
    log_dir="logs"
)
logger.info("Logger configured", extra={"status": "INFO"})


# Use JSON mode for structured outputs to reduce parsing failures
llm_json = ChatOllama(model=MODEL_NAME, temperature=0.0, format="json")

ddg_tool = DuckDuckGoSearchResults(num_results=3)


def log(title: str, obj: Any = None) -> None:
    if not DEBUG:
        return
    if obj is None:
        logger.debug(title)
        return
    if isinstance(obj, (dict, list)):
        logger.debug("%s | %s", title, json.dumps(obj, ensure_ascii=False)[:4000])
    else:
        logger.debug("%s | %s", title, str(obj)[:4000])


def dump_model(m: Any) -> dict:
    if hasattr(m, "model_dump"):
        return m.model_dump()
    if hasattr(m, "dict"):
        return m.dict()
    return {"value": str(m)}


# =============================
# PROMPTS
# =============================
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
    draft: NotRequired[Dict[str, Any]]
    revision: NotRequired[Dict[str, Any]]
    search_results: NotRequired[Dict[str, Any]]


def _latest_structured(state: AgentState) -> Optional[Dict[str, Any]]:
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

    responder = llm_json.with_structured_output(AnswerQuestion)
    structured: AnswerQuestion = (first_responder_prompt | responder).invoke(
        {"messages": state["messages"]}
    )
    structured_dict = dump_model(structured)

    log("Responder structured output", structured_dict)

    reflection_text = structured_dict.get("reflection", {})
    reflection_msg = AIMessage(
        content=f"REFLECTION: missing={reflection_text.get('missing','')} | superfluous={reflection_text.get('superfluous','')}"
    )

    return {
        "draft": structured_dict,
        "messages": [AIMessage(content=structured.answer), reflection_msg],
        "iterations": state.get("iterations", 0),
    }


def execute_tools_node(state: AgentState) -> dict:
    log("NODE execute_tools_node (iterations before)", state.get("iterations", 0))

    latest = _latest_structured(state)
    queries = (latest or {}).get("search_queries", [])
    if not isinstance(queries, list):
        queries = []
    queries = [str(q).strip() for q in queries if str(q).strip()][:3]
    log("Tool queries", queries)

    results: Dict[str, Any] = {}
    for q in queries:
        log("DDG invoking query", q)
        results[q] = ddg_tool.invoke(q)

    log("Tool results (raw)", results)

    return {"search_results": results, "iterations": state.get("iterations", 0) + 1}


def revisor_node(state: AgentState) -> dict:
    log("NODE revisor_node (iterations)", state.get("iterations", 0))

    tool_blob = json.dumps(state.get("search_results", {}), ensure_ascii=False)
    log("Search results JSON length", len(tool_blob))

    msgs = list(state["messages"]) + [
        HumanMessage(content=f"External search results (JSON):\n{tool_blob}")
    ]

    reviser = llm_json.with_structured_output(ReviseAnswer)
    revised: ReviseAnswer = (revisor_prompt | reviser).invoke({"messages": msgs})
    revised_dict = dump_model(revised)

    log("Revisor structured output", revised_dict)

    refs = revised_dict.get("references", [])
    if not isinstance(refs, list):
        refs = []

    final_text = str(revised_dict.get("answer", "")).strip() + _render_references([str(x) for x in refs])
    return {"revision": revised_dict, "messages": [AIMessage(content=final_text)]}


# =============================
# ROUTING
# =============================
def route_after_respond(state: AgentState) -> Literal["execute_tools", "revisor"]:
    iters = state.get("iterations", 0)
    draft = state.get("draft") or {}
    has_queries = bool(draft.get("search_queries"))
    log("ROUTE after respond", {"iterations": iters, "has_queries": has_queries})

    if has_queries and iters < MAX_ITERATIONS:
        return "execute_tools"
    return "revisor"


def route_after_revisor(state: AgentState) -> Literal["execute_tools", END]:
    iters = state.get("iterations", 0)
    rev = state.get("revision") or {}
    has_queries = bool(rev.get("search_queries"))
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
builder.add_conditional_edges("respond", route_after_respond, {"execute_tools": "execute_tools", "revisor": "revisor"})
builder.add_edge("execute_tools", "revisor")
builder.add_conditional_edges("revisor", route_after_revisor, {"execute_tools": "execute_tools", END: END})

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

    out_dir = Path("Graphs")
    out_dir.mkdir(exist_ok=True)

    mermaid = app.get_graph().draw_mermaid()
    (out_dir / "reflexionAgent.mmd").write_text(mermaid, encoding="utf-8")

    try:
        png_bytes = app.get_graph().draw_mermaid_png()
        (out_dir / "reflexionAgent.png").write_bytes(png_bytes)
        logger.info("Saved graph PNG: %s", str(out_dir / "reflexionAgent.png"))
    except Exception as e:
        logger.warning("Could not render reflexionAgent.png: %r", e)
        logger.info("Mermaid graph:\n%s", mermaid)

    with mlflow.start_run(run_name="reflexion_agent_run"):
        mlflow.log_text(question, "input/question.txt")
        mlflow.log_artifact(str(out_dir / "reflexionAgent.mmd"), artifact_path="graph")
        if (out_dir / "reflexionAgent.png").exists():
            mlflow.log_artifact(str(out_dir / "reflexionAgent.png"), artifact_path="graph")

        if PRINT_GRAPH_DEBUG_EVENTS:
            logger.info("LANGGRAPH DEBUG STREAM enabled")
            for event in app.stream(input_state, stream_mode="debug"):
                # This can be verbose; keep for learning
                logger.debug("DEBUG_EVENT | %s", json.dumps(event, default=str)[:4000])

        final_state = app.invoke(input_state)

        if "draft" in final_state:
            mlflow.log_text(json.dumps(final_state["draft"], indent=2, ensure_ascii=False), "outputs/draft.json")
        if "revision" in final_state:
            mlflow.log_text(json.dumps(final_state["revision"], indent=2, ensure_ascii=False), "outputs/revision.json")

    print("\n================= FINAL ANSWER =================")
    print(final_state["messages"][-1].content)
