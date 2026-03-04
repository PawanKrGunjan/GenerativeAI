"""
Offline Personal Identity Agent (ReAct - Stable Version)

Fixes:
- Proper termination flag
- Safe routing
- No silent loop
- Guaranteed final output
"""

import logging
import json
import re
from typing import Annotated, Dict, Optional, TypedDict, Literal, List, Any
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, BaseMessage
from langchain_core.tools import StructuredTool
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

from Agent.db_postgresql import (
    semantic_search,
    semantic_search_chat,
    sync_chat_history,
)
from Agent.llm import llm, embeddings_model
from Agent.config import DEFAULT_MAX_ITERATIONS


# =====================================================
# STATE
# =====================================================

class AgentState(TypedDict, total=False):
    messages: Annotated[List[BaseMessage], add_messages]
    observations: List[str]
    user_id: str
    role: Literal[
        "Friend", "Mentor", "Developer", "Researcher",
        "Strategist", "Business", "Therapist",
    ]
    current_time: datetime
    memory_context: str
    pending_action: Optional[Dict[str, Any]]
    iteration: int
    max_iterations: int
    done: bool


# =====================================================
# AGENT
# =====================================================

class PersonalAssistanceAgent:

    def __init__(
        self,
        pg_conn,
        embeddings_model,
        logger: logging.Logger,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
    ):
        self.pg_conn = pg_conn
        self.embeddings_model = embeddings_model
        self.logger = logger
        self.max_iterations = max_iterations

        self.checkpointer = MemorySaver()
        self.graph = self._build_graph()

    # =====================================================
    # UTILITIES
    # =====================================================

    def _last_user_message(self, state: AgentState) -> Optional[str]:
        for msg in reversed(state.get("messages", [])):
            if isinstance(msg, HumanMessage):
                return msg.content.strip()
        return None

    # =====================================================
    # MEMORY NODE
    # =====================================================

    def memory_node(self, state: AgentState) -> Dict:

        query = self._last_user_message(state)
        if not query:
            return {}

        docs = semantic_search(
            self.pg_conn, self.logger, query, top_k=5
        ) or []

        chats = semantic_search_chat(
            self.pg_conn,
            self.logger,
            query,
            state["user_id"],
            state["role"],
            self.embeddings_model,
            top_k=5
        ) or []

        blocks = []

        clean_docs = [d.strip() for d in docs if d and d.strip()]
        if clean_docs:
            blocks.append("### USER DOCUMENTS\n" + "\n\n".join(clean_docs))

        style_samples = [
            r.get("human_text", "").strip()
            for r in chats if r.get("human_text")
        ]
        if style_samples:
            blocks.append("### USER STYLE\n" + "\n\n".join(style_samples))

        if not blocks:
            return {}

        return {"memory_context": "\n\n".join(blocks)}

    # =====================================================
    # LLM NODE (ReAct)
    # =====================================================

    def react_llm_node(self, state: AgentState) -> Dict:

        role = state["role"]
        user_id = state["user_id"]
        context = state.get("memory_context", "")
        time_str = state.get("current_time", datetime.now(ZoneInfo("Asia/Kolkata"))) \
            .strftime("%d %B %Y, %I:%M %p IST")

        observations = "\n".join(state.get("observations", [])) or "(none)"

        system_prompt = f"""
You are {user_id}. You ARE the real person.

Speaking to: {role}
Time: {time_str}

Rules:
- Be natural and concise.
- Never mention AI or reasoning.
- Use memory + observations.
- Max steps: {state.get("max_iterations", self.max_iterations)}

Tools:
- search_documents(query)
- search_role_chats(query)

ReAct format:

Thought: ...
Action: tool_name{{"query": "..."}}
OR
Final Answer: ...

===== MEMORY =====
{context}
===== OBSERVATIONS =====
{observations}
"""

        messages = [SystemMessage(content=system_prompt)] + state["messages"]

        try:
            response = llm.invoke(messages)
            content = (response.content or "").strip()
        except Exception:
            self.logger.exception("LLM failed")
            return {
                "messages": [AIMessage(content="Something went wrong.")],
                "done": True
            }

        iteration = state.get("iteration", 0) + 1

        # 1️⃣ Final Answer detected
        if "Final Answer:" in content:
            final = content.split("Final Answer:", 1)[1].strip()
            return {
                "messages": [AIMessage(content=final)],
                "iteration": iteration,
                "done": True,
                "pending_action": None
            }

        # 2️⃣ Action detected
        action_match = re.search(
            r"Action\s*:\s*(\w+)\s*\{(.*?)\}",
            content,
            re.DOTALL
        )

        if action_match:
            tool_name = action_match.group(1).strip()
            args_raw = action_match.group(2).strip()

            try:
                args = json.loads("{" + args_raw + "}")
                return {
                    "pending_action": {"tool": tool_name, "args": args},
                    "iteration": iteration
                }
            except Exception:
                return {
                    "observations": ["Failed to parse tool arguments."],
                    "iteration": iteration
                }

        # 3️⃣ Fallback → treat entire output as final
        return {
            "messages": [AIMessage(content=content)],
            "iteration": iteration,
            "done": True
        }

    # =====================================================
    # TOOL NODE
    # =====================================================

    def tool_node(self, state: AgentState) -> Dict:

        action = state.get("pending_action")
        if not action:
            return {}

        tool = action["tool"]
        args = action.get("args", {})
        query = args.get("query", "")

        try:
            if tool == "search_documents":
                result = semantic_search(
                    self.pg_conn,
                    self.logger,
                    query,
                    top_k=4
                ) or []

                result_text = "\n\n".join(result) if result else "No documents found."

            elif tool == "search_role_chats":
                rows = semantic_search_chat(
                    self.pg_conn,
                    self.logger,
                    query,
                    state["user_id"],
                    state["role"],
                    self.embeddings_model,
                    top_k=4
                ) or []

                texts = [
                    r.get("human_text", "")
                    for r in rows if r.get("human_text")
                ]
                result_text = "\n\n".join(texts) if texts else "No chats found."

            else:
                result_text = f"Unknown tool: {tool}"

        except Exception:
            self.logger.exception("Tool failed")
            result_text = "Tool execution failed."

        return {
            "observations": [f"[{tool}] {result_text}"],
            "pending_action": None
        }

    # =====================================================
    # SAVE NODE
    # =====================================================

    def save_node(self, state: AgentState) -> Dict:

        user_input = self._last_user_message(state)

        ai_output = None
        for msg in reversed(state.get("messages", [])):
            if isinstance(msg, AIMessage):
                ai_output = msg.content.strip()
                break

        if user_input and ai_output:
            try:
                sync_chat_history(
                    self.pg_conn,
                    state["user_id"],
                    state["role"],
                    user_input,
                    ai_output,
                    self.embeddings_model,
                    self.logger,
                )
            except Exception:
                self.logger.exception("Save failed")

        return {}

    # =====================================================
    # ROUTER
    # =====================================================

    def router(self, state: AgentState) -> str:

        if state.get("pending_action"):
            return "tool"

        if state.get("done"):
            return "save"

        if state.get("iteration", 0) >= state.get("max_iterations", self.max_iterations):
            return "save"

        return "react"

    # =====================================================
    # GRAPH
    # =====================================================

    def _build_graph(self):

        workflow = StateGraph(AgentState)

        workflow.add_node("memory", self.memory_node)
        workflow.add_node("react", self.react_llm_node)
        workflow.add_node("tool", self.tool_node)
        workflow.add_node("save", self.save_node)

        workflow.add_edge(START, "memory")
        workflow.add_edge("memory", "react")

        workflow.add_conditional_edges(
            "react",
            self.router,
            {
                "tool": "tool",
                "save": "save",
                "react": "react"
            }
        )

        workflow.add_edge("tool", "react")
        workflow.add_edge("save", END)

        return workflow.compile(checkpointer=self.checkpointer)

    # =====================================================
    # RUN
    # =====================================================

    def run(
        self,
        user_id: str,
        role: str,
        text: str,
        thread_id: Optional[str] = None,
    ) -> Optional[str]:

        config = {
            "configurable": {
                "thread_id": thread_id or f"{user_id}_{role}"
            }
        }

        state = {
            "messages": [HumanMessage(content=text)],
            "user_id": user_id,
            "role": role,
            "current_time": datetime.now(ZoneInfo("Asia/Kolkata")),
            "iteration": 0,
            "max_iterations": self.max_iterations,
            "observations": []
        }

        final_answer = None

        try:
            for event in self.graph.stream(state, config=config, stream_mode="values"):
                if "messages" in event:
                    for msg in event["messages"]:
                        if isinstance(msg, AIMessage):
                            final_answer = msg.content.strip()

            return final_answer

        except Exception:
            self.logger.exception("Run failed")
            return None

    def save_graph_visualization(
        self,
        graph_dir: Path,
        save_png: bool = False,
    ) -> None:
        """Save Mermaid diagram + optional PNG."""
        try:
            graph_dir.mkdir(parents=True, exist_ok=True)

            md_path = graph_dir / "personalAgent_react.md"
            mermaid_code = self.graph.get_graph().draw_mermaid()
            md_path.write_text(f"```mermaid\n{mermaid_code}\n```")

            if save_png:
                try:
                    png_path = graph_dir / "personalAgent_react.png"
                    png_bytes = self.graph.get_graph().draw_mermaid_png()
                    png_path.write_bytes(png_bytes)
                    self.logger.info("Graph PNG saved: %s", png_path)
                except Exception as e:
                    self.logger.warning("PNG generation failed: %s", e)

            self.logger.info("Graph Markdown saved: %s", md_path)

        except Exception as e:
            self.logger.warning("Graph visualization failed: %s", e)


# =====================================================
# Example usage (commented out — uncomment for quick testing)
# =====================================================

if __name__ == "__main__":
    import logging
    from Agent.db_postgresql import get_pg_connection

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("personal-agent")

    conn = get_pg_connection(logger)

    agent = PersonalAssistanceAgent(
        pg_conn=conn,
        embeddings_model=embeddings_model,
        logger=logger,
        max_iterations=5,
    )

    # Optional: save graph viz
    agent.save_graph_visualization(Path("graphs"), save_png=False)

    response = agent.run(
        user_id="PawanGunjan",
        role="Mentor",
        text="How should I structure my preparation for FAANG interviews in 2026?",
    )

    print("Final response:")
    print(response or "[No response]")