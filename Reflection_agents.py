from typing import Annotated, TypedDict, List, Literal
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

# =============================================================================
# STATE - NO BUILTIN FUNCTIONS, ONLY "add"
# =============================================================================
class ReflectionState(TypedDict):
    messages: Annotated[List[BaseMessage], "add"]
    reflections: Annotated[int, "add"]
    draft_count: Annotated[int, "add"]
    quality_scores: Annotated[List[float], "add"]  # Store all scores
    is_finalized: bool

# =============================================================================
# CUSTOM REDUCER FUNCTIONS
# =============================================================================
def max_reducer(left: List[float], right: List[float]) -> List[float]:
    """Keep highest quality score."""
    return [max(left + right)]

def get_best_quality(state: ReflectionState) -> float:
    """Helper to get current best quality."""
    scores = state.get("quality_scores", [0.0])
    return max(scores) if scores else 0.0

# =============================================================================
# NODES
# =============================================================================
def generate_draft(state: ReflectionState) -> dict:
    draft_count = state.get("draft_count", 0)
    
    drafts = [
        "Photosynthesis: Plants use sunlight, water + CO2 → sugar + oxygen.",
        "Photosynthesis occurs in chloroplasts. Chlorophyll captures light, producing ATP/NADPH.",
        "Complete: Light reactions (thylakoids) split H2O → O2 + ATP + NADPH. Calvin cycle fixes CO2 → glucose.",
        "Perfect: 6CO2 + 6H2O + light → C6H12O6 + 6O2. Factors: light, CO2, temperature."
    ]
    
    draft = drafts[min(draft_count, len(drafts)-1)]
    return {
        "messages": [AIMessage(content=f".. DRAFT #{draft_count+1}: {draft}")],
        "draft_count": draft_count + 1
    }

def reflect_critique(state: ReflectionState) -> dict:
    reflections = state.get("reflections", 0)
    
    critiques = [
        {"text": " ! Too basic. Needs: chlorophyll, reactions, equation. SCORE: 3/10", "score": 3.0},
        {"text": " **Better structure. Missing: chemical equation, factors. SCORE: 6/10**", "score": 6.0},
        {"text": " ! Good detail on reactions. Add equation for perfection. SCORE: 8/10 !", "score": 8.0},
        {"text": " !!Excellent! Complete, accurate, well-structured. SCORE: 10/10 !!", "score": 10.0}
    ]
    
    critique = critiques[min(reflections, len(critiques)-1)]
    return {
        "messages": [AIMessage(content=f"🔍 CRITIQUE #{reflections+1}: {critique['text']}")],
        "reflections": reflections + 1,
        "quality_scores": [critique["score"]]
    }

def finalize_answer(state: ReflectionState) -> dict:
    final = """ **FINAL ANSWER: Photosynthesis Explained**

**Process**: Plants convert light energy → chemical energy in chloroplasts.

**1. Light Reactions** (thylakoids):
- Chlorophyll absorbs sunlight  
- H₂O splits → O₂ + ATP + NADPH

**2. Calvin Cycle** (stroma):
- CO₂ + ATP + NADPH → glucose (C₆H₁₂O₆)

**Equation**: `6CO₂ + 6H₂O + light → C₆H₁₂O₆ + 6O₂`

**Factors**: Light intensity, CO₂, temperature, chlorophyll."""
    
    return {
        "messages": [AIMessage(content=final)],
        "is_finalized": True
    }

# =============================================================================
# ROUTING - Bulletproof
# =============================================================================
def route_reflection(state: ReflectionState) -> Literal["reflect_critique", "generate_draft", "finalize_answer", "__end__"]:
    reflections = state.get("reflections", 0)
    quality = get_best_quality(state)
    draft_count = state.get("draft_count", 0)
    
    MAX_REFLECTIONS = 3
    MIN_QUALITY = 8.0
    
    print(f"... Reflections: {reflections}, Best Quality: {quality:.1f}")
    
    # TERMINATION CONDITIONS
    if reflections >= MAX_REFLECTIONS or quality >= MIN_QUALITY:
        print(" ! TERMINATING: Max reflections or quality met")
        return "finalize_answer"
    
    # CYCLE LOGIC
    last_msg = state["messages"][-1].content
    if ".. DRAFT" in last_msg:
        print("-->  Next: Critique")
        return "reflect_critique"
    else:
        print("-->  Next: New draft")
        return "generate_draft"

# =============================================================================
# BUILD GRAPH
# =============================================================================
def create_reflection_agent():
    builder = StateGraph(ReflectionState)
    
    builder.add_node("generate_draft", generate_draft)
    builder.add_node("reflect_critique", reflect_critique)
    builder.add_node("finalize_answer", finalize_answer)
    
    builder.set_entry_point("generate_draft")
    
    # Perfect routing
    builder.add_conditional_edges(
        "generate_draft",
        route_reflection,
        {
            "reflect_critique": "reflect_critique",
            "generate_draft": "generate_draft",
            "finalize_answer": "finalize_answer"
        }
    )
    builder.add_conditional_edges(
        "reflect_critique",
        route_reflection,
        {
            "reflect_critique": "reflect_critique",
            "generate_draft": "generate_draft",
            "finalize_answer": "finalize_answer"
        }
    )
    builder.add_edge("finalize_answer", END)
    
    return builder.compile()

# =============================================================================
# RUN
# =============================================================================
if __name__ == "__main__":
    app = create_reflection_agent()
    
    query = HumanMessage(content="Explain photosynthesis.")
    result = app.invoke(
        {"messages": [query], "reflections": 0, "draft_count": 0, "quality_scores": [], "is_finalized": False},
        {"recursion_limit": 50}
    )
    
    print("\n" + "="*70)
    print(" REFLECTION AGENT - FULL EXECUTION TRACE")
    print("="*70)
    for i, msg in enumerate(result["messages"][1:], 1):
        print(f"{i:2d}. {msg.content}")
    
    print(f"\n **FINAL STATS:**")
    print(f"   Reflections: {result['reflections']}")
    print(f"   Drafts: {result['draft_count']}")
    print(f"   Best Quality: {max(result['quality_scores']):.1f}/10")
