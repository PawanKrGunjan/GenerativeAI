from langgraph.graph import StateGraph, END
from agents.planner_agent import PlannerAgent
from agents.tool_executor import ToolExecutor
from agents.analyzer_agent import AnalyzerAgent
from agents.reflection_agent import ReflectionAgent
import traceback
from utils.config import GRAPH_DIR
from utils.logger import LOGGER
from typing import TypedDict, Optional, List, Dict, Any

class InvestmentState(TypedDict, total=False):
    tool_calls: Optional[List[Dict[str, Any]]]
    invalid_data: Optional[bool]
    analysis_tools_needed: Optional[bool]
    planner_output: Optional[str]
    data: Optional[Dict[str, Any]]
    analysis: Optional[Dict[str, Any]]
    reflection: Optional[str]

planner = PlannerAgent()
data_executor = ToolExecutor(tool_type="data")
analysis_executor = ToolExecutor(tool_type="analysis")
analyzer = AnalyzerAgent()
reflector = ReflectionAgent()


def planner_node(state):
    LOGGER.info("Planner node")
    return planner.run(state)


def data_tool_node(state):
    LOGGER.info("Data tool execution node")
    return data_executor.run(state)


def analysis_tool_node(state):
    LOGGER.info("Analysis tool execution node")
    return analysis_executor.run(state)


def analyzer_node(state):
    LOGGER.info("Analyzer node")
    return analyzer.run(state)


def reflection_node(state):
    LOGGER.info("Reflection node")
    return reflector.run(state)


# -------------------------------
# ROUTERS
# -------------------------------

def planner_router(state):

    tool_calls = state.get("tool_calls")

    if tool_calls:
        return "data_tools"

    return "analysis"


def analyzer_router(state):

    if state.get("invalid_data"):
        return "planner"

    if state.get("analysis_tools_needed"):
        return "analysis_tools"

    return "reflection"


# -------------------------------
# GRAPH
# -------------------------------

def build_graph():

    workflow = StateGraph(InvestmentState)

    workflow.add_node("planner", planner_node)
    workflow.add_node("data_tools", data_tool_node)
    workflow.add_node("analysis_tools", analysis_tool_node)
    workflow.add_node("analysis", analyzer_node)
    workflow.add_node("reflection", reflection_node)

    workflow.set_entry_point("planner")

    workflow.add_conditional_edges(
        "planner",
        planner_router,
        {
            "data_tools": "data_tools",
            "analysis": "analysis"
        }
    )

    workflow.add_edge("data_tools", "analysis")

    workflow.add_conditional_edges(
        "analysis",
        analyzer_router,
        {
            "planner": "planner",
            "analysis_tools": "analysis_tools",
            "reflection": "reflection"
        }
    )

    workflow.add_edge("analysis_tools", "analysis")

    workflow.add_edge("reflection", END)

    LOGGER.info("Investment graph compiled")

    return workflow.compile()

def save_workflow_graph(gr, save_png: bool = False):
    try:
        GRAPH_DIR.mkdir(parents=True, exist_ok=True)
        graph_name= "investment_agent_v1"

        path = GRAPH_DIR / f"{graph_name}.md"
        mermaid = gr.get_graph().draw_mermaid()
        path.write_text(f"```mermaid\n{mermaid}\n```")
        LOGGER.info(f"Graph saved → {path}")

        if save_png:
            png_path = GRAPH_DIR / f"{graph_name}.png"
            png_bytes = gr.get_graph().draw_mermaid_png()
            png_path.write_bytes(png_bytes)
            LOGGER.info(f"PNG saved: {png_path}")

    except Exception as e:
        LOGGER.warning(f"Graph visualization failed: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    gr = build_graph()
    save_workflow_graph(gr, save_png=False)


