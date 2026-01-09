#!/usr/bin/env python3
"""
Tool Agent: Single Logger + Planner → Tool → Format Pipeline
Fixed 'logger not defined' + Your exact reflect_prompt + calculator
"""

from __future__ import annotations
from typing import Annotated, Sequence, TypedDict, Literal, List, Dict, Any
import uuid
import operator
from pathlib import Path
from datetime import datetime
import os, json
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, ConfigDict
from dotenv import load_dotenv
from logger_config import setup_logger  # Single logger source

# ==============================
# Single Logger (Fixed)
# ==============================
load_dotenv()
logger = setup_logger(debug_mode=True, log_name='ToolAgent', log_dir='logs')  # ONE logger

GRAPH_DIR = Path("Graphs")
GRAPH_DIR.mkdir(exist_ok=True)

# ==============================
# Your Exact Calculator
# ==============================
class CalcInput(BaseModel):
    model_config = ConfigDict(extra='forbid')
    operation: Literal["add", "subtract", "multiply", "divide", "power"] = Field(...)
    a: float = Field(..., ge=-1e12, le=1e12)
    b: float = Field(0.0, ge=-1e12, le=1e12)

@tool(args_schema=CalcInput)
def calculator(operation: str, a: float, b: float = 0.0) -> float:
    """Financial calculator with per-call logging (your exact code)."""
    call_id = str(uuid.uuid4())[:8]
    
    logger.info(f"CALL {call_id}: {operation}(a={a}, b={b})")
    
    operations = {
        "add": operator.add, "subtract": operator.sub,
        "multiply": operator.mul, "divide": operator.truediv, "power": operator.pow
    }
    
    try:
        if operation not in operations:
            raise ValueError(f"Invalid operation '{operation}'")
        if operation == "divide" and abs(b) < 1e-12:
            result = float('inf')
        else:
            result = operations[operation](a, b)
            result = round(float(result), 8)
        
        logger.info(f"RESULT {call_id}: {result}")
        return result
        
    except Exception as e:
        error_msg = f"ERROR {call_id}: {str(e)}"
        logger.error(error_msg)
        return 0.0

# ==============================
# LLM (Planner Only)
# ==============================
perplexity_llm = ChatOpenAI(
    base_url="https://api.perplexity.ai",
    api_key=os.getenv("PERPLEXITY_API_KEY"),
    model="sonar-pro",
    temperature=0.0
)

# ==============================
# State
# ==============================
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    iteration: int
    operations: List[Dict[str, Any]]  # Planned ops queue
    results: Dict[str, float]         # Step results
    tool_trace: List[Dict]            # Call history

# ==============================
# 1. PLANNER (LLM → JSON Operations)
# ==============================
def escape_braces(s: str) -> str:
    return s.replace("{", "{{").replace("}", "}}")

system_text = escape_braces("""
Parse math -> JSON operations ONLY.

"SI P=1000 R=3% T=2" ->
[
  {"step":"rate","op":"multiply","a":1000,"b":0.03},
  {"step":"final","op":"multiply","a":"{{rate}}","b":2}
]

JSON response ONLY:
{"operations":[...]}
""".strip())

planner_prompt = ChatPromptTemplate.from_messages([
    ("system", system_text),
    ("human", "{query}")
])



planner_chain = planner_prompt | perplexity_llm

def planner_node(state: AgentState) -> dict:
    iteration = state.get("iteration", 1)
    query = state["messages"][-1].content
    
    logger.info(f"Planner iter{iteration}: '{query[:40]}...'")
    
    plan = planner_chain.invoke({"query": query})
    
    try:
        # Strict JSON extraction
        start = plan.content.find('{')
        end = plan.content.rfind('}') + 1
        ops_data = json.loads(plan.content[start:end])
        operations = ops_data["operations"]
        
        logger.info(f"Planner: {len(operations)} operations queued")
        return {
            "messages": [AIMessage(content=f"Planned {len(operations)} steps")],
            "iteration": iteration,
            "operations": operations,
            "results": {},
            "tool_trace": []
        }
    except Exception as e:
        logger.error(f"Planner failed: {e}")
        return {"messages": [AIMessage(content="PLAN_ERROR")], "iteration": iteration}

# ==============================
# 2. TOOL EXECUTOR (Runs Queue)
# ==============================
def tool_node(state: AgentState) -> dict:
    operations = state["operations"]
    results = state.get("results", {})
    trace = state.get("tool_trace", [])
    
    logger.info(f"Executor: {len(operations)} operations")
    
    tool_msgs = []
    for i, op in enumerate(operations):
        try:
            # Resolve {{stepN}} references
            a = op["a"]
            b = op.get("b", 0.0)
            
            if isinstance(a, str) and a.startswith("{{") and a.endswith("}}"):
                a = results.get(a[2:-2], 0.0)
            if isinstance(b, str) and b.startswith("{{") and b.endswith("}}"):
                b = results.get(b[2:-2], 0.0)
            
            # Execute (triggers your logger!)
            args = CalcInput(operation=op["op"], a=float(a), b=float(b))
            result = calculator.invoke(args.model_dump())
            
            step_id = op.get("step", f"step{i+1}")
            results[step_id] = result
            
            trace.append({
                "step": step_id,
                "op": op["op"],
                "input": {"a": a, "b": b},
                "output": result
            })
            
            # tool_msgs.append(ToolMessage(
            #     content=f"Step{step_id}: {op['op']}({a},{b})={result}",
            #     name="calculator"
            # ))
            tool_msgs.append(ToolMessage(
                content=str(result),  # Only the result as content
                tool_call_id=str(uuid.uuid4()),  # Required!
                name="calculator"
            ))
            logger.info(f"Step {i+1}: {step_id} = {result}")
            
        except Exception as e:
            logger.error(f"Step {i+1} error: {e}")
            results[f"step{i+1}"] = 0.0
    
    logger.info(f"Executor done: {len(results)} results")
    return {
        "messages": tool_msgs,
        "results": results,
        "tool_trace": trace
    }

# ==============================
# 3. FORMATTER (Perfect Output)
# ==============================
def formatter_node(state: AgentState) -> dict:
    operations = state["operations"]
    results = state["results"]
    
    steps = []
    for op in operations:
        step_id = op.get("step", "unknown")
        r = results.get(step_id, 0.0)
        steps.append(f"{op['op']}({op['a']},{op.get('b', '?')})={r}")
    
    final = f"""1. FORMULA: SI = P * R * T / 100
2. VALUES: Parsed from query
3. STEPS: {' → '.join(steps)}
4. RESULT: {results.get('final', list(results.values())[-1] or 0):.2f}"""
    
    logger.info("Format complete")
    return {"messages": [AIMessage(content=final.strip())]}

# ==============================
# Your Reflection Prompt
# ==============================
reflect_prompt = ChatPromptTemplate.from_messages([
    ("system", """YES if format perfect:
- 1.FORMULA, 2.VALUES P/R/T, 3.STEPS w/ calculator calls, 4.RESULT
NO otherwise. ONLY "YES"/"NO"."""),
    MessagesPlaceholder("messages")
])

reflect_chain = reflect_prompt | perplexity_llm
def reflect_node(state: AgentState) -> dict:
    messages = state["messages"]
    
    # Find the original user query (first HumanMessage)
    user_query = None
    for msg in messages:
        if isinstance(msg, HumanMessage):
            user_query = msg
            break
    
    if not user_query:
        user_query = HumanMessage(content="Unknown query")
    
    # Get the final formatted answer (the last AIMessage with the format)
    final_answer = None
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and "1. FORMULA:" in msg.content:
            final_answer = msg
            break
    
    if not final_answer:
        # Fallback: take the very last AIMessage
        for msg in reversed(messages):
            if isinstance(msg, AIMessage):
                final_answer = msg
                break
    
    if not final_answer:
        final_answer = AIMessage(content="No output generated.")
    
    # Construct STRICTLY alternating messages: Human → Assistant
    reflection_messages = [
        user_query,           # Human asks question
        final_answer          # Agent gives formatted answer
    ]
    
    logger.info(f"Reflect input: HumanMessage → AIMessage (content has '1. FORMULA:')")

    try:
        verdict = reflect_chain.invoke({"messages": reflection_messages})
        verdict_text = verdict.content.strip().upper()
        
        # Force strict YES/NO
        if verdict_text not in ["YES", "NO"]:
            logger.warning(f"Invalid verdict: '{verdict.content}'. Forcing NO.")
            verdict_text = "NO"
            
        logger.info(f"Reflect verdict: {verdict_text}")
        
        return {
            "messages": [AIMessage(content=f"VERDICT: {verdict_text}")],
            "iteration": state.get("iteration", 0) + 1
        }
        
    except Exception as e:
        logger.error(f"Reflection failed: {e}")
        return {
            "messages": [AIMessage(content="VERDICT: NO")],
            "iteration": state.get("iteration", 0) + 1
        }
    
# def reflect_node(state: AgentState) -> dict:
#     # Get the full conversation history
#     messages = state["messages"]
    
#     # Find the last HumanMessage to anchor the reflection context
#     # Everything after it is agent/tool activity we want to judge
#     reflection_messages = []
#     for msg in reversed(messages):
#         if isinstance(msg, HumanMessage):
#             reflection_messages.append(msg)
#             break
#         else:
#             # Collapse all trailing Assistant/Tool messages into one AI message
#             if not reflection_messages:
#                 # First non-human message (going backwards) becomes the content to judge
#                 if hasattr(msg, "content"):
#                     reflection_messages.append(AIMessage(content=msg.content))
    
#     # Now go forward from that HumanMessage and include final AI response
#     found_human = False
#     for msg in messages:
#         if isinstance(msg, HumanMessage):
#             found_human = True
#             reflection_messages.append(msg)
#         elif found_human and isinstance(msg, (AIMessage, ToolMessage)):
#             # Only add the very last AI message (the formatted output)
#             if isinstance(msg, AIMessage) and msg.content.strip().startswith("1. FORMULA:"):
#                 reflection_messages.append(msg)
#                 break

#     # Fallback: if something went wrong, just use last human + last AI
#     if len(reflection_messages) < 2:
#         last_human = next((m for m in reversed(messages) if isinstance(m, HumanMessage)), None)
#         last_ai = messages[-1] if isinstance(messages[-1], AIMessage) else None
#         reflection_messages = []
#         if last_human:
#             reflection_messages.append(last_human)
#         if last_ai:
#             reflection_messages.append(last_ai)

#     logger.info(f"Reflect input messages: {[type(m).__name__ for m in reflection_messages]}")

#     verdict = reflect_chain.invoke({"messages": reflection_messages})
#     verdict_text = verdict.content.strip().upper()

#     # Only accept strict YES/NO
#     if verdict_text not in ["YES", "NO"]:
#         verdict_text = "NO"
#         logger.warning(f"Reflect returned invalid verdict: '{verdict.content}'. Forcing NO.")

#     logger.info(f"Reflect verdict: {verdict_text}")

#     return {
#         "messages": [AIMessage(content=f"VERDICT: {verdict_text}")],
#         "iteration": state.get("iteration", 0) + 1
#     }

# ==============================
# Smart Router
# ==============================
def route_node(state: AgentState) -> Literal["planner", "tools", "format", "reflect", END]:
    ops = state.get("operations", [])
    results = state.get("results", {})
    iteration = state.get("iteration", 0)

    if iteration >= 3:
        return END

    # Check for reflection verdict
    if state["messages"]:
        last_msg = state["messages"][-1]
        if isinstance(last_msg, AIMessage):
            content = last_msg.content.upper()
            if "VERDICT: YES" in content:
                return END
            if "VERDICT: NO" in content:
                return "planner"

    # Main flow logic
    if not ops or len(ops) == 0:
        return "planner"
    if ops and not results:
        return "tools"
    if results:
        # After tools → format, or after format → reflect if needed
        # But if we just formatted and no verdict yet → go to reflect
        if len(state["messages"]) >= 3 and "1. FORMULA:" in state["messages"][-1].content:
            return "reflect"
        return "format"

    return "reflect"  # fallback

# ==============================
# Pipeline Graph
# ==============================
def create_workflow():
    graph = StateGraph(AgentState)
    
    graph.add_node("planner", planner_node)
    graph.add_node("tools", tool_node)
    graph.add_node("format", formatter_node)
    graph.add_node("reflect", reflect_node)
    
    graph.set_entry_point("planner")

    # === CRITICAL: All possible returns from route_node MUST be mapped ===
    path_map = {
        "planner": "planner",
        "tools": "tools",
        "format": "format",
        "reflect": "reflect",
        END: END
    }

    # From planner: can go anywhere
    graph.add_conditional_edges("planner", route_node, path_map)

    # From tools: usually to format or reflect
    graph.add_conditional_edges("tools", route_node, {
        "format": "format",
        "reflect": "reflect",
        "planner": "planner",  # in case of verdict NO
        END: END
    })

    # From format: go to reflect or end
    graph.add_conditional_edges("format", route_node, {
        "reflect": "reflect",
        END: END
    })

    # From reflect: always back to planner (to retry or end via route)
    graph.add_edge("reflect", "planner")
    
    app = graph.compile()

    # Graph visualization
    png_path = GRAPH_DIR / f"pipeline_{datetime.now().strftime('%H%M%S')}.png"
    try:
        png_data = app.get_graph().draw_mermaid_png()
        png_path.write_bytes(png_data)
        logger.info(f"PNG saved: {png_path}")
    except Exception as e:
        logger.warning(f"Graph render failed: {e}")
    
    return app

# ==============================
# Execute
# ==============================
if __name__ == "__main__":
    query = "Simple Interest: P=1000, R=3%, T=2 years"
    
    logger.info("Pipeline started")
    print("=" * 60)
    
    app = create_workflow()
    result = app.invoke({
        "messages": [HumanMessage(content=query)],
        "iteration": 0
    })
    
    print("=" * 60)
    
    # Trace
    trace = result.get("tool_trace", [])
    if trace:
        print("\nTOOL EXECUTION:")
        for t in trace:
            print(f"Step {t['step']}: {t['op']}({t['args']['a']},{t['args']['b']}) = {t['output']}")
    
    print("\nRESULT:")
    print("-" * 40)
    print(result["messages"][-1].content.strip())
