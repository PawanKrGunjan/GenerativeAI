from typing import Annotated, TypedDict, List, Literal
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import re
import datetime
import logging


# =============================================================================
# PYTHON STANDARD LOGGER - CLEAN FORMAT
# =============================================================================
logger = logging.getLogger('REFLEX AGENT')
logger.setLevel(logging.INFO)

# Simple formatter: [HH:MM:SS] message
class CleanFormatter(logging.Formatter):
    def format(self, record):
        # Extract status from extra dict for context (optional)
        status = getattr(record, 'status', '')
        if status:
            record.msg = f"[{status.upper()}] {record.msg}"
        return super().format(record)

# Console handler - timestamp only
handler = logging.StreamHandler()
handler.setFormatter(CleanFormatter(
    '[%(asctime)s] %(message)s',
    datefmt='%H:%M:%S'
))
logger.addHandler(handler)



# =============================================================================
# OLLAMA SETUP
# =============================================================================
model_name = "llama3.2:latest"  #"qwen2.5:3b"
llm = ChatOllama(model=model_name, temperature=0.1)
logger.info(f"Using model: {model_name}", extra={'status': 'INFO'})


# =============================================================================
# STATE
# =============================================================================
class ReflectionState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    reflections: int
    draft_count: int
    quality_scores: List[float]
    is_finalized: bool
    last_node: str


def get_best_quality(state: ReflectionState) -> float:
    scores = state.get("quality_scores", [])
    return max(scores) if scores else 0.0


# =============================================================================
# PROMPTS (Fixed minor syntax)
# =============================================================================
draft_prompt = ChatPromptTemplate.from_messages([
    ("system", """Expert teacher. Answer in max 10 sentences or less.
Clear, accurate, complete. Use feedback to improve.
NO headings, NO fluff."""),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{query}")
])


critique_prompt = ChatPromptTemplate.from_messages([
    ("system", """Strict grader. Score 0-10 based on:
✓ Clear (easy to understand)
✓ Accurate (factually correct)  
✓ Short (≤5 lines)
✓ Complete (covers essentials)

EXACT FORMAT REQUIRED:

SCORE: X/10
GOOD: YES or NO
SUGGESTION: 1 short fix (max 10 words)"

Replace X with number 0-10."""),
    ("human", "GRADE:\n\n{draft}")
])

final_prompt = ChatPromptTemplate.from_template(
    """From these drafts and critiques, write FINAL answer (≤5 lines):
{history}

Answer only - no headings."""
)


# =============================================================================
# NODES WITH LOGGING
# =============================================================================
def generate_draft(state: ReflectionState) -> dict:
    logger.info(f"Generating DRAFT {state.get('draft_count', 0) + 1}", extra={'status': 'DRAFT'})
    
    query = state["messages"][0].content
    history = state["messages"][1:]
    
    chain = draft_prompt | llm
    response = chain.invoke({"query": query, "history": history})
    
    draft_count = state.get("draft_count", 0) + 1
    draft_content = response.content.strip()
    logger.info(f"Draft {draft_count} generated ({len(draft_content)} chars)", extra={'status': 'DRAFT'})
    
    return {
        "messages": [AIMessage(content=f"DRAFT {draft_count}:\n{draft_content}")],
        "draft_count": draft_count,
        "last_node": "generate_draft"
    }


def reflect_critique(state: ReflectionState) -> dict:
    logger.info(f"Starting CRITIQUE {state.get('reflections', 0) + 1}", extra={'status': 'CRITIQUE'})
    
    drafts = [m for m in state["messages"] if m.content.startswith("DRAFT")]
    if not drafts:
        logger.info("No draft found for critique", extra={'status': 'CRITIQUE'})
        return {
            "messages": [AIMessage(content="CRITIQUE 1:\nNo draft found")],
            "reflections": 1,
            "quality_scores": [0.0],
            "last_node": "reflect_critique"
        }
    
    draft_text = drafts[-1].content.split("DRAFT", 1)[1].strip()
    logger.info(f"Critiquing draft ({len(draft_text)} chars)", extra={'status': 'CRITIQUE'})
    
    chain = critique_prompt | llm
    response = chain.invoke({"draft": draft_text})
    content = response.content.strip()

    # Robust score extraction
    score = 5.0
    score_match = re.search(r"SCORE[:\s]*(\d+(?:\.\d+)?)", content, re.IGNORECASE)
    if score_match:
        try:
            score = max(0.0, min(10.0, float(score_match.group(1))))
        except:
            score = 5.0
    
    logger.info(f"Score extracted: {score:.1f}/10", extra={'status': 'CRITIQUE'})
    
    reflections = state.get("reflections", 0) + 1
    
    return {
        "messages": [AIMessage(content=f"CRITIQUE {reflections}:\n{content}")],
        "reflections": reflections,
        "quality_scores": state["quality_scores"] + [score],
        "last_node": "reflect_critique"
    }


def finalize_answer(state: ReflectionState) -> dict:
    logger.info("GENERATING FINAL ANSWER", extra={'status': 'FINAL'})
    
    recent = "\n---\n".join([m.content for m in state["messages"][-6:]])
    logger.info(f"Using {len(recent)} chars of history", extra={'status': 'FINAL'})
    
    chain = final_prompt | llm
    response = chain.invoke({"history": recent})
    final_content = response.content.strip()
    
    logger.info(f"Final answer generated ({len(final_content)} chars)", extra={'status': 'FINAL'})
    
    return {
        "messages": [AIMessage(content=f"**FINAL ANSWER** (≤5 lines):\n{final_content}")],
        "is_finalized": True,
        "last_node": "finalize_answer"
    }


# =============================================================================
# ROUTING WITH LOGGING
# =============================================================================
def route_reflection(state: ReflectionState) -> Literal["reflect_critique", "generate_draft", "finalize_answer", str]:
    reflections = state.get("reflections", 0)
    best_quality = get_best_quality(state)
    last_node = state.get("last_node", "")
    
    MAX_REFLECTIONS = 3
    MIN_QUALITY = 8.0
    
    logger.info(f"🔄 Reflections: {reflections}/{MAX_REFLECTIONS} | Best: {best_quality:.1f}/10 | Last: {last_node}", extra={'status': 'ITER'})
    
    if state.get("is_finalized", False):
        logger.info("Already finalized → END", extra={'status': 'END'})
        return END
    
    if reflections >= MAX_REFLECTIONS:
        logger.info("❌ Max reflections reached → FINALIZE", extra={'status': 'END'})
        return "finalize_answer"
    
    if best_quality >= MIN_QUALITY:
        logger.info(f"⭐ Quality {best_quality:.1f} ≥ {MIN_QUALITY} → FINALIZE", extra={'status': 'END'})
        return "finalize_answer"
    
    if last_node == "generate_draft":
        logger.info("Draft done → CRITIQUE", extra={'status': 'ITER'})
        return "reflect_critique"
    elif last_node == "reflect_critique":
        logger.info("Critique done → DRAFT", extra={'status': 'ITER'})
        return "generate_draft"
    
    logger.info("Default → CRITIQUE", extra={'status': 'ITER'})
    return "reflect_critique"


# =============================================================================
# GRAPH
# =============================================================================
def create_reflection_agent():
    logger.info("Building reflection graph...", extra={'status': 'INFO'})
    
    builder = StateGraph(ReflectionState)
    
    builder.add_node("generate_draft", generate_draft)
    builder.add_node("reflect_critique", reflect_critique)
    builder.add_node("finalize_answer", finalize_answer)
    
    builder.set_entry_point("generate_draft")
    
    builder.add_conditional_edges(
        "generate_draft", 
        route_reflection,
        {
            "reflect_critique": "reflect_critique", 
            "generate_draft": "generate_draft", 
            "finalize_answer": "finalize_answer",
            END: END
        }
    )
    builder.add_conditional_edges(
        "reflect_critique", 
        route_reflection,
        {
            "reflect_critique": "reflect_critique", 
            "generate_draft": "generate_draft", 
            "finalize_answer": "finalize_answer",
            END: END
        }
    )
    builder.add_edge("finalize_answer", END)
    
    app = builder.compile()
    logger.info("Graph compiled successfully", extra={'status': 'INFO'})
    return app


# =============================================================================
# RUN
# =============================================================================
if __name__ == "__main__":
    logger.info("REFLECTION AGENT STARTED", extra={'status': 'START'})
    print("="*80)
    
    app = create_reflection_agent()
    
    try:
        app.get_graph().draw_png("reflect_agent_logged.png")
        logger.info("Graph saved: reflect_agent_logged.png", extra={'status': 'INFO'})
    except Exception as e:
        logger.info(f"Graph PNG failed: {e}", extra={'status': 'INFO'})
        print(app.get_graph().draw_mermaid())
    
    query = "Explain Reflection Agent"
    logger.info(f"Query: {query}", extra={'status': 'INFO'})
    
    initial_state = {
        "messages": [HumanMessage(content=query)],
        "reflections": 0,
        "draft_count": 0,
        "quality_scores": [],
        "is_finalized": False,
        "last_node": ""
    }
    
    result = app.invoke(initial_state, {"recursion_limit": 10})
    
    print("\n" + "="*80)
    logger.info("AGENT COMPLETE", extra={'status': 'END'})
    print("="*80)
    
    print(f"SUMMARY: Drafts: {result['draft_count']} | Critiques: {result['reflections']} | Best: {get_best_quality(result):.1f}/10")
    
    print("\n FINAL ANSWER:")
    print("-" * 50)
    
    final_found = False
    for msg in reversed(result["messages"]):
        if isinstance(msg, AIMessage) and "FINAL ANSWER" in msg.content:
            print(msg.content.split("FINAL ANSWER", 1)[1].strip())
            final_found = True
            break
    
    if not final_found:
        logger.info("No final answer found in messages", extra={'status': 'INFO'})
    
    logger.info("PROCESS COMPLETE", extra={'status': 'END'})
    print("\n✅ DONE!")
