import json
from typing import List, Dict, Any, Optional
from datetime import datetime
from zoneinfo import ZoneInfo

from langgraph.graph import StateGraph, END
from langgraph.store.memory import InMemoryStore
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    ToolMessage,
)

from pydantic import BaseModel, Field
from typing import Annotated
import operator
from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    MessagesPlaceholder
)

# Local imports
from agents.llm import get_llm, embed_text, EMBEDDING_DIM
from tools.tool_registry import TOOLS
from utils.logger import LOGGER
from utils.config import GRAPH_DIR
from utils.db_connect import get_connection
from memory.store_memory import update_reflection


IST = ZoneInfo("Asia/Kolkata")
max_attempts = 3

TOOL_MAP = {}
for tool in TOOLS:
    TOOL_MAP[tool.name]=tool

LLM = get_llm(temperature=0.0)
LLM_WITH_TOOLS = LLM.bind_tools(TOOLS)

conn = get_connection(LOGGER)

system_template = """
You are a professional Investment Advisor specializing in the Indian Stock Market.
Your goal is to provide well-researched, data-driven investment advice that helps users maximize returns while balancing risk.

Current IST time: {current_datetime}
Symbols in use: {symbols}

=====================
STRICT TOOL RULES
=====================

You MUST use tools to gather data.

- Never write Python code
- Never generate code blocks
- Never simulate tool outputs
- Never guess stock symbols
- Always call the correct tool
- Always wait for tool results before continuing

If a tool is required, CALL THE TOOL instead of answering.

=====================
CRITICAL TOOL PIPELINE
=====================

Follow EXACTLY in this order:

1️⃣ SYMBOL FIRST  
Always call **lookup_stock_symbol** using the company name from the user query.  
Do NOT guess stock symbols.

2️⃣ PRICE  
Call **get_stock_info** with the symbol returned by lookup.

3️⃣ NEWS  
Call **get_stock_news** with the confirmed symbol.

4️⃣ ANALYSIS  
Only after ALL tools succeed provide Stock-Specific Advice.

=====================
RETRY LOGIC
=====================

If get_stock_info fails:
→ Call lookup_stock_symbol again
→ Retry get_stock_info

If lookup_stock_symbol fails:
→ Tell the user the company is not listed on NSE.

=====================
CONTEXT
=====================

News: {news}

Memory: {memory}

=====================
FINAL OUTPUT RULE
=====================

Provide the final answer ONLY after the price tool succeeds.

Output Format:

Stock-Specific Advice

Company: <Company Name>  
Symbol: <Stock Symbol>  
Current Price: ₹<price>  
Price Time: <datetime IST>

Signal: BUY / HOLD / SELL / NEUTRAL  
Confidence: <0-100>/100

If any step fails → mention it in Notes.
"""

PROMPTS = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(system_template),
        MessagesPlaceholder(variable_name="messages"),
    ]
)


LLMChain = PROMPTS | LLM_WITH_TOOLS

# Agent state
class InvestmentAgentState(BaseModel):
    # Conversation messages
    messages: List[BaseMessage] = Field(default_factory=list)
    # Retry / loop counter
    attempt_count: int = 0
    # Resolved stock symbols
    symbols: List[Dict[str, str]] = Field(default_factory=list)
    # Current timestamp
    current_datetime: datetime = Field(default_factory=datetime.now)
    # Tool results / intermediate memory
    memory: List[Any] = Field(default_factory=list)
    # News data cache
    news: Dict[str, Any] = Field(default_factory=dict)
    # Tool calls returned by LLM
    tool_calls: List[Dict[str, Any]] = Field(default_factory=list)
    #last_tool_result: Dict[str, Any] | None = None
    # Final LLM answer
    result: Optional[str] = None

def reasoning_node(state: InvestmentAgentState):

    LOGGER.info("NODE → REACT")

    try:
        response = LLMChain.invoke(
            {
                "messages": state.messages,
                "symbols": state.symbols,
                "news": state.news,
                "memory": state.memory,
                "current_datetime": state.current_datetime,
            }
        )

        updates = {}

        # Case 1: Proper structured tool call
        if response.tool_calls:
            updates["tool_calls"] = response.tool_calls
            return updates

        content = response.content or ""

        # Case 2: Tool call returned as JSON text
        if "<|python_tag|>" in content:
            try:
                json_str = content.replace("<|python_tag|>", "").strip()
                tool_json = json.loads(json_str)

                updates["tool_calls"] = [
                    {
                        "name": tool_json["name"],
                        "args": tool_json["parameters"],
                        "id": "manual_tool_call"
                    }
                ]

                return updates

            except Exception:
                LOGGER.warning("Failed to parse tool call from python_tag")

        # Case 3: Final answer
        if content:
            ai_msg = AIMessage(content=content)

            updates["messages"] = state.messages + [ai_msg]
            updates["result"] = content

        return updates

    except Exception:
        LOGGER.exception("Unexpected error in reasoning_node")
        return {}
    
def router(state: InvestmentAgentState):
    if state.tool_calls:
        return "Tool_Call"

    if state.result:
        return "REFLECT"

    if state.attempt_count >= 5:
        return END

    return END

# Define a function to execute tool calls based on the response
def execute_tool_calls(state: InvestmentAgentState):

    LOGGER.info(f"Executing tool calls | count={len(state.tool_calls)}")

    messages = list(state.messages)

    for tool in state.tool_calls:
        tool_name = tool["name"]
        tool_args = tool["args"]
        tool_call_id = tool["id"]

        result = TOOL_MAP[tool_name].invoke(tool_args)

        LOGGER.info(f"{tool_name} result received")

        messages.append(
            ToolMessage(
                content=json.dumps(result),
                tool_call_id=tool_call_id
            )
        )

        if tool_name == "lookup_stock_symbol":
            if isinstance(result, list):
                state.symbols.extend(result)

    return {
        "messages": messages,
        "tool_calls": [],
        "symbols": state.symbols
    }


def reflection_node(state: InvestmentAgentState) -> dict:

    LOGGER.info("NODE → reflection")

    if not state.symbols:
        LOGGER.info("No symbols found → skipping reflection")
        return {}

    if not state.messages:
        LOGGER.warning("No messages in state → skipping reflection")
        return {}

    # Extract symbol string
    symbol = next(iter(state.symbols[0].values()))

    last_msg = state.messages[-1]

    if not isinstance(last_msg, AIMessage):
        LOGGER.info("Last message not AIMessage → skipping reflection")
        return {}

    reflection = f"""
Agent Decision:

{last_msg.content}

Reflection Task:
1. Track price movement after this signal.
2. Evaluate if the recommendation was profitable.
3. Store lessons for future signals.

Timestamp:
{datetime.now(IST)}
"""

    try:
        update_reflection(symbol, reflection)
        LOGGER.info(f"Reflection stored → {symbol}")

    except Exception:
        LOGGER.exception("Reflection storage failed")

    return {}

store = InMemoryStore(index={"embed": embed_text,"dims": EMBEDDING_DIM})

def build_graph(state: InvestmentAgentState, store):
    memory= InMemorySaver()
    workflow = StateGraph(state)

    workflow.add_node("REACT", reasoning_node)
    workflow.add_node("Tool_Call", execute_tool_calls)
    workflow.add_node("REFLECT", reflection_node)

    workflow.set_entry_point("REACT")

    workflow.add_conditional_edges(
        "REACT",
        router,
        {
            "Tool_Call": "Tool_Call",
            "REFLECT": "REFLECT",
            END: END,
        }
    )

    workflow.add_edge("Tool_Call", "REACT")
    workflow.add_edge("REFLECT", END)

    graph = workflow.compile(store=store)
    save_graph_visualization(graph,
                             save_png=False)
    return graph

def save_graph_visualization(graph,save_png: bool = False):
    try:
        GRAPH_DIR.mkdir(parents=True, exist_ok=True)
        graph_name= f"investment_agent_base"

        path = GRAPH_DIR / f"{graph_name}.md"
        mermaid = graph.get_graph().draw_mermaid()
        path.write_text(f"```mermaid\n{mermaid}\n```")
        LOGGER.info(f"Graph saved → {path}")

        if save_png:
            png_path = GRAPH_DIR / f"{graph_name}.png"
            png_bytes = graph.get_graph().draw_mermaid_png()
            png_path.write_bytes(png_bytes)
            LOGGER.info(f"PNG saved: {png_path}")

    except Exception as e:
        LOGGER.warning(f"Graph visualization failed: {e}")

gr = build_graph(InvestmentAgentState, store)
#chatbot = chat_step(gr, prev_state: InvestmentAgentState, user_query: str) -> InvestmentAgentState:

# if __name__== "__main__":
#     gr = build_graph(InvestmentAgentState, store)
#     save_graph_visualization(graph=gr,save_png=True)

#     # Create an initial state
#     user_query = 'Analyze Larsen & Toubro stocks'
#     AgentState = InvestmentAgentState(
#         messages=[HumanMessage(user_query)],
#         attempt_count=0,
#         symbols=[],
#         current_datetime=datetime.now(IST),
#         memory=[],
#         news={'What is current price of Larsen & Toubro Ltd': 'somwhat nearby 3000'},
#     )


#     AgentState.model_dump()
#     result = gr.invoke(AgentState)
#     print(result)
