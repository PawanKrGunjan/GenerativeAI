from __future__ import annotations
from typing import Annotated, Sequence, TypedDict
import json
import datetime
from typing import Literal, Union
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, BaseMessage
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import tool
from pydantic import BaseModel, Field, ConfigDict


# ==============================
# Logger
# ==============================
def log_status(message: str, status: str = "INFO"):
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    icons = {"START": "Start", "DRAFT": "Draft", "TOOL": "Tool", "REFLECT": "Reflect", "END": "End", "ITER": "Iter"}
    print(f"[{timestamp}] {icons.get(status, 'Info')} {message}")


# ==============================
# ROBUST Calculator Tool
# ==============================
class CalculatorInput(BaseModel):
    model_config = ConfigDict(extra='allow')
    
    operation: Literal["add", "subtract", "multiply", "divide", "power"] = Field(...)
    a: float = Field(...)
    b: float = Field(0.0)


@tool(args_schema=CalculatorInput)
def calculator(operation: str, a: float, b: float = 0.0, **kwargs) -> float:    
    """Basic calculator: add, subtract, multiply, divide, power ONLY."""
    ops = {
        "add": a + b,
        "subtract": a - b,
        "multiply": a * b,
        "divide": a / b if b != 0 else 0.0,
        "power": a ** b,
    }
    return round(ops.get(operation, 0.0), 6)


# ==============================
# LLM - More Precise
# ==============================
llm = ChatOllama(model="llama3.2:latest", temperature=0.0)
llm_with_tools = llm.bind_tools([calculator])


# ==============================
# ULTRA-SPECIFIC PROMPTS
# ==============================
generation_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a math agent. Solve EXACTLY like this for Simple Interest (SI = P*R*T/100):

1. Formula: SI = P * R * T / 100
2. Values: P=1000, R=0.03, T=2  
3. Calculation: First multiply(1000, 0.03) then multiply(result, 2)
4. Final answer: 60

CALCULATOR RULES:
- Use EXACTLY: operation="multiply", a=first_number, b=second_number
- NO other field names (P, R, T)
- R must be decimal (3% = 0.03)

If you cannot use calculator correctly, just compute mentally and write the 4 lines."""),
    MessagesPlaceholder("messages")
])

chain = generation_prompt | llm_with_tools

# FIXED Reflection - Much stricter
reflection_prompt = ChatPromptTemplate.from_messages([
    ("system", """Check LAST AI MESSAGE (ignore tool calls):
YES if it contains ALL:
✓ "Formula:" 
✓ "Values:" with P=, R=, T=  
✓ "Final answer:" with number
✓ Exactly 4 lines

NO otherwise.

Reply ONLY: YES or NO (no explanation)"""),
    MessagesPlaceholder("messages")
])

reflect_chain = reflection_prompt | llm


# ==============================
# State
# ==============================
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    iteration: int


# ==============================
# BULLETPROOF Agent Node
# ==============================
def agent_node(state: AgentState) -> dict:
    iter_num = state.get("iteration", 0) + 1
    log_status(f"Starting iteration {iter_num}", "ITER")
    
    response = chain.invoke({"messages": state["messages"]})
    current = list(state["messages"]) + [response]

    # Smart tool handling with arg fixing
    max_tools = 3
    tool_count = 0
    
    while (hasattr(response, 'tool_calls') and 
           response.tool_calls and 
           tool_count < max_tools):
        tool_count += 1
        log_status(f"Tool attempt {tool_count}", "TOOL")
        tool_msgs = []
        
        for tc in response.tool_calls:
            try:
                args = tc["args"].copy()
                # AUTO-FIX common LLM mistakes
                args_fix = {
                    'P': 'a', 'principal': 'a',
                    'R': 'b', 'rate': 'b', 
                    'T': 'b', 'time': 'b'
                }
                for wrong, correct in args_fix.items():
                    if wrong in args:
                        args[correct] = float(args.pop(wrong))
                
                # Convert strings to float
                for k, v in args.items():
                    if isinstance(v, str):
                        args[k] = float(v) if v.replace('.','').replace('-','').isdigit() else 0.0
                
                result = calculator.invoke(args)
                log_status(f"✅ Tool OK: {result}", "TOOL")
                tool_msgs.append(ToolMessage(
                    content=f"Result = {result}",
                    tool_call_id=tc["id"]
                ))
            except Exception as e:
                log_status(f"❌ Tool failed, skipping: {str(e)[:50]}", "TOOL")
                tool_msgs.append(ToolMessage(
                    content="Use manual calculation instead",
                    tool_call_id=tc["id"]
                ))
        
        current.extend(tool_msgs)
        response = chain.invoke({"messages": current})

    # FORCE good final response
    final_content = response.content.strip()
    if not final_content or len(final_content.split('\n')) < 3:
        final_content = """None"""
    current.append(AIMessage(content=final_content))
    log_status("Answer generated", "DRAFT")
    return {"messages": current, "iteration": iter_num}


def reflection_node(state: AgentState) -> dict:
    log_status("Checking quality...", "REFLECT")
    response = reflect_chain.invoke({"messages": state["messages"]})
    critique = response.content.strip().upper()
    log_status(f"Quality: {critique}", "REFLECT")
    return {"messages": state["messages"] + [AIMessage(content=critique)]}


# ==============================
# Smarter Routing
# ==============================
def should_continue(state: AgentState):
    iteration = state.get("iteration", 0)
    if iteration >= 4:  # Reduced max iterations
        log_status("Max iterations → END", "END")
        return END

    # Check for good answer directly
    for msg in reversed(state["messages"]):
        if isinstance(msg, AIMessage) and msg.content:
            content = msg.content.lower()
            if all(word in content for word in ["formula", "values", "final answer"]):
                log_status("✅ Perfect format found!", "END")
                return END
    
    log_status(f"Iter {iteration} → improve", "ITER")
    return "reflect"


# ==============================
# Graph & Run
# ==============================
workflow = StateGraph(AgentState)
workflow.add_node("agent", agent_node)
workflow.add_node("reflect", reflection_node)
workflow.set_entry_point("agent")
workflow.add_conditional_edges("agent", should_continue, {"reflect": "reflect", END: END})
workflow.add_edge("reflect", "agent")
app = workflow.compile()

try:
    app.get_graph().draw_png("graph.png")
    log_status("Graph saved", "INFO")
except:
    pass


if __name__ == "__main__":
    query = "Compute Compound Interest for 2 years if Principal is 1000 rupees at the Rate of 3% half yearly"
    
    log_status("AGENT STARTED", "START")
    print("="*80)
    
    result = app.invoke({"messages": [HumanMessage(content=query)], "iteration": 0})
    
    print("="*80)
    log_status("AGENT COMPLETE", "END")
    print("\nFINAL SOLUTION:\n")
    
    # Better answer extraction
    for msg in reversed(result["messages"]):
        if isinstance(msg, AIMessage) and msg.content:
            content = msg.content.strip()
            if "formula" in content.lower() and "final answer" in content.lower():
                print(content)
                break
