from typing import Annotated, TypedDict, List, Literal
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import re

# =============================================================================
# OLLAMA SETUP
# =============================================================================
model_name = "qwen2.5:3b" #"llama3.2:latest"
llm = ChatOllama(model=model_name, temperature=0.1)

# =============================================================================
# STATE
# =============================================================================
class ReflectionState(TypedDict):
    messages: Annotated[List[BaseMessage], "add"]
    reflections: Annotated[int, "add"]
    draft_count: Annotated[int, "add"]
    quality_scores: Annotated[List[float], "add"]
    is_finalized: bool
    last_node: str

def get_best_quality(state: ReflectionState) -> float:
    scores = state.get("quality_scores", [])
    return max(scores) if scores else 0.0

# =============================================================================
# PROMPTS
# =============================================================================
draft_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert teacher. Write CLEAR, ACCURATE explanations.
REQUIRED FORMAT:
**INTRODUCTION**
**STEP-BY-STEP PROCESS**
**KEY EQUATION**
**IMPORTANT FACTORS**

Improve using all previous feedback. Aim for 200-300 words."""),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{query}")
])

critique_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a strict professor grading an explanation on a 0-10 scale.
MANDATORY FORMAT - COPY EXACTLY:

SCORE: X/10

STRENGTHS:
* Point 1
* Point 2

WEAKNESSES:
* Issue 1
* Issue 2

IMPROVEMENTS:
* Suggestion 1
* Suggestion 2

Score FIRST. 10 = perfect, 0 = terrible."""),
    ("human", "GRADE THIS EXPLANATION:\n\n{draft}")
])

# =============================================================================
# NODES
# =============================================================================
def generate_draft(state: ReflectionState) -> dict:
    query = state["messages"][0].content
    history = state["messages"][1:]
    chain = draft_prompt | llm
    response = chain.invoke({"query": query, "history": history})
    
    draft_count = state.get("draft_count", 0) + 1
    
    return {
        "messages": [AIMessage(content=f".. DRAFT #{draft_count}\n\n{response.content}")],
        "draft_count": draft_count,
        "last_node": "generate_draft"
    }

def reflect_critique(state: ReflectionState) -> dict:
    # Extract the most recent draft
    drafts = [m for m in state["messages"] if m.content.startswith(".. DRAFT")]
    if not drafts:
        draft_text = ""
    else:
        draft_text = drafts[-1].content.split("\n\n", 1)[1] if "\n\n" in drafts[-1].content else drafts[-1].content

    chain = critique_prompt | llm
    response = chain.invoke({"draft": draft_text})
    
    # Robust score extraction
    score = 5.0
    content = response.content
    score_match = re.search(r"SCORE:\s*(\d+(?:\.\d+)?)\s*/\s*10", content, re.IGNORECASE)
    if score_match:
        try:
            score = float(score_match.group(1))
            score = max(0.0, min(10.0, score))
        except:
            pass

    reflections = state.get("reflections", 0) + 1
    
    return {
        "messages": [AIMessage(content=f"🔍 CRITIQUE #{reflections}\n\n{content}")],
        "reflections": reflections,
        "quality_scores": state["quality_scores"] + [score],  # append to list
        "last_node": "reflect_critique"
    }

def finalize_answer(state: ReflectionState) -> dict:
    # Use recent history for context (last 6 messages)
    recent_history = "\n\n".join([m.content for m in state["messages"][-6:]])
    
    final_prompt = ChatPromptTemplate.from_template(
        """Using the entire refinement process below, produce the FINAL PERFECT ANSWER.

Process history:
{history}

Required format:
  **FINAL ANSWER**

**Introduction**
**Step-by-Step Process**
**Key Equation**
**Important Factors**

Make it comprehensive, clear, accurate, and beautifully polished."""
    )
    
    chain = final_prompt | llm
    response = chain.invoke({"history": recent_history})
    
    return {
        "messages": [AIMessage(content=f" **FINAL ANSWER**\n\n{response.content}")],
        "is_finalized": True,
        "last_node": "finalize_answer"
    }

# =============================================================================
# ROUTING
# =============================================================================
def route_reflection(state: ReflectionState) -> Literal["reflect_critique", "generate_draft", "finalize_answer"]:
    reflections = state.get("reflections", 0)
    best_quality = get_best_quality(state)
    last_node = state.get("last_node", "")
    
    MAX_REFLECTIONS = 3
    MIN_QUALITY = 8.0  # Slightly higher threshold for better results
    
    print(f"... Reflections: {reflections}/{MAX_REFLECTIONS} | Best score: {best_quality:.1f}/10 | Last: {last_node}")
    
    # Termination conditions
    if reflections >= MAX_REFLECTIONS or best_quality >= MIN_QUALITY:
        print(" ! Terminating → FINALIZE")
        return "finalize_answer"
    
    # Normal alternation
    if last_node == "generate_draft":
        print("--> Next → CRITIQUE")
        return "reflect_critique"
    elif last_node == "reflect_critique":
        print("-->Next → DRAFT")
        return "generate_draft"
    
    # Fallback
    print("Fallback routing")
    return "reflect_critique"

# =============================================================================
# BUILD GRAPH
# =============================================================================
def create_reflection_agent():
    builder = StateGraph(ReflectionState)
    
    builder.add_node("generate_draft", generate_draft)
    builder.add_node("reflect_critique", reflect_critique)
    builder.add_node("finalize_answer", finalize_answer)
    
    builder.set_entry_point("generate_draft")
    
    # Conditional edges from both draft and critique
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
    print(f"Starting Reflection Agent with {model_name}")
    
    app = create_reflection_agent()
    
    query = HumanMessage(content="Explain photosynthesis step-by-step")
    
    initial_state = {
        "messages": [query],
        "reflections": 0,
        "draft_count": 0,
        "quality_scores": [],
        "is_finalized": False,
        "last_node": ""
    }
    
    print("\n" + "="*80)
    print("STARTING REFLECTION CYCLE\n")
    
    result = app.invoke(initial_state, {"recursion_limit": 20})
    
    print("\n" + "="*80)
    print("REFLECTION CYCLE COMPLETE")
    print(f"Drafts created: {result['draft_count']}")
    print(f"Critiques performed: {result['reflections']}")
    print(f"Best quality score: {get_best_quality(result):.1f}/10")
    print("\nFULL OUTPUT:\n")
    
    for i, msg in enumerate(result["messages"], 1):
        print(f"{i}. {msg.content}\n{'-'*60}")