#!/usr/bin/env python3
"""
ReflectAgent: Self-Improving LinkedIn Post Generator (Production Clean)
Separate Human/AI/Critique storage, no emojis, strict 160 char limit
"""

import os
import json
import re
from datetime import datetime
from typing import List, Annotated, Dict, TypedDict, Literal, Any
from pathlib import Path
from dataclasses import dataclass
from typing import List as ListType

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from logger_config import setup_logger

load_dotenv()

@dataclass
class IterationData:
    iteration: int
    human_prompt: str
    ai_post: str
    critique: str
    post_chars: int
    hashtags: ListType[str]
    timestamp: str

def save_state(state: Dict, path: Path):
    with open(path, 'w') as f:
        json.dump({
            "messages": [{"role": getattr(m, 'type', 'unknown'), "content": m.content} 
                        for m in state["messages"]],
            "metadata": state.get("metadata", {}),
            "iteration": state.get("iteration", 0)
        }, f, indent=2, default=str)

def load_state(path: Path) -> Dict:
    with open(path, 'r') as f:
        data = json.load(f)
    return data

# ========================================
# CONFIGURATION
# ========================================
class Config:
    MAX_ITERATIONS: int = 5
    MAX_CHARS: int = 300
    MODEL: str = "sonar-pro"
    PERPLEXITY_API_KEY: str = os.getenv("PERPLEXITY_API_KEY")
    LOG_DIR: str = "logs"
    STATE_DIR: str = "states"
    CRITIQUE_DIR: str = "critiques"
    
    @classmethod
    def validate(cls):
        if not cls.PERPLEXITY_API_KEY:
            raise ValueError("PERPLEXITY_API_KEY missing from .env")

Config.validate()

# ========================================
# LOGGER
# ========================================
logger = setup_logger(debug_mode=True, log_name='ReflectAgent', log_dir=Config.LOG_DIR)


# Graph folder
GRAPH_DIR = Path("Graphs")
GRAPH_DIR.mkdir(exist_ok=True)
# ========================================
# LLM + STRICT PROMPTS (Fixed char limit)
# ========================================
perplexity_llm = ChatOpenAI(
    base_url="https://api.perplexity.ai",
    api_key=Config.PERPLEXITY_API_KEY,
    model=Config.MODEL,
    temperature=0.0  # Deterministic
)

MODEL_NAME = "granite4:350m" #"qwen2.5:7b"  # change if needed (e.g., "llama3.1:8b")
llm = ChatOllama(model=MODEL_NAME, temperature=0.0)

generation_prompt = ChatPromptTemplate.from_messages([
    ("system", """LinkedIn content specialist. CRITICAL: Exactly {max_chars} chars MAX.
Structure: Hook (20c) + Value (90c) + CTA+hashtags (50c).
Refine using critique feedback. Count characters before posting."""),
    ("human", "{user_request}")
])

reflection_prompt = ChatPromptTemplate.from_messages([
    ("system", """Professional critique. Format exactly:
STRENGTHS: [2 bullets]
WEAKNESSES: [2 bullets] 
IMPROVEMENTS: [3 numbered actionable steps]
Max 300 chars total. Specific to LinkedIn engagement."""), 
    ("human", "{post_to_critique}")
])

generate_chain = generation_prompt | llm
reflect_chain = reflection_prompt | llm

# ========================================
# STATE
# ========================================
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    metadata: Dict[str, Any]
    iteration: int
    iterations_data: List[IterationData]

# ========================================
# NODES (Strict 160 char enforcement)
# ========================================
def generation_node(state: AgentState) -> AgentState:
    iteration = state.get("iteration", 0) + 1
    
    context = next((m.content for m in state["messages"][::-1] 
                   if isinstance(m, (HumanMessage, AIMessage))), "")
    
    logger.info(f"Generation {iteration}: Processing '{context[:50]}...'")
    
    result = generate_chain.invoke({
        "user_request": context,
        "max_chars": Config.MAX_CHARS
    })
    
    # STRICT TRUNCATION + PRESERVE HASHTAGS
    post = result.content.strip()
    hashtags = re.findall(r'#\w+', post)
    
    if len(post) > Config.MAX_CHARS:
        # Truncate body, keep hashtags
        body = post[:Config.MAX_CHARS - 20 - len(' '.join(hashtags))]
        post = f"{body.strip()} {' '.join(hashtags)}".strip()[:Config.MAX_CHARS]
        logger.info(f"Truncated from {len(result.content)} to {len(post)} chars")
    
    ai_msg = AIMessage(content=post)
    
    iter_data = IterationData(
        iteration=iteration,
        human_prompt=context,
        ai_post=post,
        critique="",
        post_chars=len(post),
        hashtags=hashtags,
        timestamp=datetime.now().isoformat()
    )
    
    logger.info(f"Generated {iteration}: {len(post)} chars, {len(hashtags)} hashtags")
    
    return {
        "messages": [ai_msg],
        "iteration": iteration,
        "iterations_data": state.get("iterations_data", []) + [iter_data]
    }

def reflection_node(state: AgentState) -> AgentState:
    iteration = state.get("iteration", 0)
    last_post = next((m.content for m in reversed(state["messages"]) 
                     if isinstance(m, AIMessage)), "")
    
    logger.info(f"Reflection {iteration}: Analyzing post ({len(last_post)} chars)")
    
    critique_result = reflect_chain.invoke({"post_to_critique": last_post})
    critique_text = critique_result.content.strip()
    
    # Save individual critique
    Path(Config.CRITIQUE_DIR).mkdir(exist_ok=True)
    critique_path = Path(Config.CRITIQUE_DIR) / f"critique_{iteration:02d}.md"
    
    with open(critique_path, 'w') as f:
        f.write(f"# Critique Iteration {iteration}\n\n")
        f.write(f"**Post ({len(last_post)} chars):**\n{last_post}\n\n")
        f.write(f"**Analysis:**\n{ critique_text }\n")
    
    logger.info(f"Critique saved: {critique_path}")
    
    critique_msg = AIMessage(content=f"CRITIQUE {iteration}: {critique_text}")
    
    # Backfill iteration data
    iterations_data = list(state["iterations_data"])
    iterations_data[-1].critique = critique_text
    
    return {
        "messages": [critique_msg],
        "iterations_data": iterations_data
    }

def should_continue(state: AgentState) -> Literal["reflect", END]:
    iteration = state.get("iteration", 0)
    logger.info(f"Decision point: iteration {iteration}/{Config.MAX_ITERATIONS}")
    
    return END if iteration >= Config.MAX_ITERATIONS else "reflect"

# ========================================
# WORKFLOW FACTORY
# ========================================
def create_workflow(session_id: str = None):
    session_id = session_id or f"reflect_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    graph = StateGraph(AgentState)
    graph.add_node("generate", generation_node)
    graph.add_node("reflect", reflection_node)
    
    graph.set_entry_point("generate")
    graph.add_conditional_edges("generate", should_continue, {
        "reflect": "reflect", END: END
    })
    graph.add_edge("reflect", "generate")
    
    workflow = graph.compile(checkpointer=MemorySaver())
    logger.info(f"Workflow initialized: session {session_id}")
    # Unique graph PNG
    png_path = GRAPH_DIR / f"reflection_agent_{session_id}.png"
    try:      
        # Save PNG file
        png_data = workflow.get_graph().draw_mermaid_png()
        png_path.write_bytes(png_data)
        logger.info(f"PNG saved: {png_path}")
        
    except Exception as e:
        logger.warning(f"Graph render failed: {e}")

    return workflow, session_id

# ========================================
# FILE SAVING (Separate Human/AI/Critique)
# ========================================
def save_separate_files(response: AgentState, session_id: str):
    Path(Config.STATE_DIR).mkdir(exist_ok=True)
    
    # 1. Full state JSON
    state_path = Path(Config.STATE_DIR) / f"{session_id}_full.json"
    save_state(response, state_path)
    
    # 2. Structured iterations (Human/AI/Critique)
    iterations = response.get("iterations_data", [])
    iters_path = Path(Config.STATE_DIR) / f"{session_id}_iterations.json"
    
    iters_export = [{
        "iteration": it.iteration,
        "human_prompt": it.human_prompt[:100] + "..." if len(it.human_prompt) > 100 else it.human_prompt,
        "ai_post": it.ai_post,
        "critique": it.critique[:200] + "..." if len(it.critique) > 200 else it.critique,
        "chars": it.post_chars,
        "hashtags": it.hashtags,
        "valid_length": it.post_chars <= Config.MAX_CHARS
    } for it in iterations]
    
    with open(iters_path, 'w') as f:
        json.dump(iters_export, f, indent=2)
    
    # 3. Best post (shortest valid)
    valid_posts = [it for it in iterations if it.post_chars <= Config.MAX_CHARS]
    best_post = valid_posts[-1].ai_post if valid_posts else iterations[-1].ai_post
    
    best_path = Path(Config.STATE_DIR) / f"{session_id}_best_post.txt"
    with open(best_path, 'w') as f:
        f.write(best_post)
    
    logger.info(f"Files saved: full={state_path}, iterations={iters_path}, best={best_path}")
    return state_path, iters_path, best_path

# ========================================
# ANALYSIS PRINTER
# ========================================
def print_conversation(response: AgentState):
    print("\nCONVERSATION LOG")
    print("=" * 60)
    
    messages = response["messages"]
    for i, msg in enumerate(messages):
        role = "Human" if isinstance(msg, HumanMessage) else "AI"
        msg_type = "POST" if i % 4 == 1 else "CRITIQUE" if i % 4 == 2 else ""
        print(f"\n[{role}] {msg_type}:")
        print("-" * 40)
        print(msg.content.strip()[:300] + ("..." if len(msg.content) > 300 else ""))
        print()

# ========================================
# MAIN EXECUTION
# ========================================
def run_reflect_agent(user_prompt: str, session_id: str = None) -> Dict[str, Any]:
    workflow, session_id = create_workflow(session_id)
    
    inputs = {
        "messages": [HumanMessage(content=user_prompt)],
        "metadata": {"original_prompt": user_prompt, "max_chars": Config.MAX_CHARS},
        "iterations_data": []
    }
    
    config = {"configurable": {"thread_id": session_id}}
    
    logger.info(f"Starting workflow: '{user_prompt[:60]}...'")
    response = workflow.invoke(inputs, config)
    
    state_path, iters_path, best_path = save_separate_files(response, session_id)
    print_conversation(response)
    
    return {
        "session_id": session_id,
        "files": {
            "full_state": str(state_path),
            "iterations": str(iters_path), 
            "best_post": str(best_path)
        },
        "response": response,
        "iterations": response.get("iterations_data", [])
    }

if __name__ == "__main__":
    result = run_reflect_agent("Write LinkedIn post: IBM software developer job under 160 characters")
    
    print(f"\nFiles generated:")
    for name, path in result["files"].items():
        print(f"  {name}: {path}")
    
    print(f"\nBest post saved to: {result['files']['best_post']}")
