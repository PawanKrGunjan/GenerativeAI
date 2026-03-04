# self_improving_agent_v2.py
# ============================================================
# Stable Self-Learning Personal Agent
# ReAct + Reflection + Style Evolution (Production Safe)
# ============================================================

import logging
import json
import re
import hashlib
from typing import Dict, Optional, TypedDict, List
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, BaseMessage
from langchain_core.tools import StructuredTool
from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import MemorySaver

from Agent.db_postgresql import semantic_search, semantic_search_chat, sync_chat_history
from Agent.llm import llm, embeddings_model


# ============================================================
# STATE (Fully Defined — No Silent Field Dropping)
# ============================================================

class AgentState(TypedDict, total=False):
    messages: List[BaseMessage]
    user_id: str
    role: str
    thread_id: str
    current_time: datetime

    # iteration
    iteration: int
    max_iterations: int

    # memory
    memory_context: str
    style_guide: str

    # tools
    pending_action: Optional[Dict]
    observations: List[str]

    # answer
    candidate_answer: Optional[str]

    # reflection
    reflection_result: Dict
    lessons: List[str]


# ============================================================
# AGENT
# ============================================================

class PersonalAssistanceAgent:

    # --------------------------------------------------------
    # INIT
    # --------------------------------------------------------

    def __init__(
        self,
        pg_conn,
        embeddings_model,
        logger: logging.Logger,
        max_iterations: int = 3,
    ):
        self.pg_conn = pg_conn
        self.embeddings_model = embeddings_model
        self.logger = logger
        self.max_iterations = max_iterations

        self.checkpointer = MemorySaver()
        self._cache = {}
        self._style_guides = {}

        # Tools
        self.search_docs_tool = StructuredTool.from_function(
            func=self._search_docs,
            name="search_documents",
            description="Search private documents."
        )

        self.search_chats_tool = StructuredTool.from_function(
            func=self._search_chats,
            name="search_role_chats",
            description="Search past messages in this role."
        )

        self.graph = self._build_graph()

    # ========================================================
    # UTILITIES
    # ========================================================

    def _get_last_human(self, state: AgentState) -> str:
        for m in reversed(state["messages"]):
            if isinstance(m, HumanMessage):
                return m.content.strip()
        return ""

    def _get_last_ai(self, state: AgentState) -> str:
        for m in reversed(state["messages"]):
            if isinstance(m, AIMessage):
                return m.content.strip()
        return ""

    # ---------------- Retrieval -----------------------------

    def _search_docs(self, query: str) -> str:
        try:
            docs = semantic_search(self.pg_conn, self.logger, query, top_k=2) or []
            return "\n\n".join(d.strip() for d in docs)
        except:
            return "No documents found."

    def _search_chats(self, query: str, user_id: str, role: str) -> str:
        try:
            rows = semantic_search_chat(
                self.pg_conn,
                self.logger,
                query,
                user_id,
                role,
                self.embeddings_model,
                top_k=2
            ) or []
            return "\n".join(r.get("human_text", "")[:120] for r in rows)
        except:
            return "No chat history."

    def _get_context(self, query, user_id, role, thread_id):
        if len(query) < 30:
            return ""

        key = f"{thread_id}:{hashlib.md5(query.encode()).hexdigest()[:8]}"
        if key in self._cache:
            return self._cache[key]

        docs = self._search_docs(query)
        chats = self._search_chats(query, user_id, role)

        ctx = f"Docs:\n{docs}\n\nChat Style Samples:\n{chats}"
        self._cache[key] = ctx
        return ctx

    # ---------------- Style Guide -----------------------------

    def _load_style_guide(self, user_id, role):
        key = f"{user_id}_{role}"
        if key in self._style_guides:
            return self._style_guides[key]

        path = Path(f"style_guides/{key}.txt")
        if path.exists():
            txt = path.read_text().strip()
            self._style_guides[key] = txt
            return txt
        return ""

    def _update_style_guide(self, user_id, role, lessons):
        key = f"{user_id}_{role}"
        current = self._load_style_guide(user_id, role)

        prompt = f"""
Current style guide:
{current}

New successful lessons:
{chr(10).join('- ' + l for l in lessons)}

Merge them into a concise guide (max 15 bullets).
Return full updated guide only.
"""

        try:
            updated = llm.invoke([SystemMessage(content=prompt)]).content.strip()
            if len(updated) > 100:
                path = Path(f"style_guides/{key}.txt")
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(updated)
                self._style_guides[key] = updated
                self.logger.info("Style guide updated.")
        except Exception as e:
            self.logger.warning(f"Style guide update failed: {e}")

    # ========================================================
    # GRAPH NODES
    # ========================================================

    def memory_node(self, state: AgentState):
        q = self._get_last_human(state)

        return {
            "memory_context": self._get_context(
                q,
                state["user_id"],
                state["role"],
                state["thread_id"]
            ),
            "style_guide": self._load_style_guide(
                state["user_id"],
                state["role"]
            ),
        }

    # --------------------------------------------------------

    def generator_node(self, state: AgentState):

        system_prompt = f"""
You are {state['user_id']}.
Speak naturally and concisely.

Role: {state['role']}
Time: {state['current_time'].strftime("%d %b %Y, %I:%M %p IST")}

Style guide:
{state.get("style_guide", "")}

Context:
{state.get("memory_context", "")}

Rules:
- Be human.
- Be concise.
- Do not mention AI or internal reasoning.
- Use tool at most once.

Format:
Action: tool_name{{"query":"..."}}
OR
Final Answer: ...
"""

        messages = [SystemMessage(content=system_prompt)] + state["messages"]

        try:
            output = llm.invoke(messages).content.strip()
        except Exception as e:
            self.logger.error(f"LLM failed: {e}")
            output = "Sorry, something went wrong."

        new_state = {"iteration": state["iteration"] + 1}

        # Explicit Final Answer
        if "Final Answer:" in output:
            final = output.split("Final Answer:", 1)[1].strip()
            new_state["candidate_answer"] = final
            new_state["messages"] = state["messages"] + [AIMessage(content=final)]
            return new_state

        # Tool call
        action_match = re.search(r"Action:\s*(\w+)\s*\{([^}]*)\}", output)
        if action_match:
            tool = action_match.group(1)
            args = json.loads("{" + action_match.group(2) + "}")
            new_state["pending_action"] = {"tool": tool, "args": args}
            return new_state

        # Fallback: accept output
        cleaned = output.replace("Thought:", "").strip()

        if len(cleaned) > 2:
            new_state["candidate_answer"] = cleaned
            new_state["messages"] = state["messages"] + [AIMessage(content=cleaned)]

        return new_state

    # --------------------------------------------------------

    def tool_node(self, state: AgentState):

        action = state.get("pending_action")
        if not action:
            return {}

        tool = action["tool"]
        args = action.get("args", {})

        try:
            if tool == "search_documents":
                result = self.search_docs_tool.invoke(args)
            elif tool == "search_role_chats":
                result = self.search_chats_tool.invoke({
                    **args,
                    "user_id": state["user_id"],
                    "role": state["role"],
                })
            else:
                result = "Unknown tool."

        except Exception as e:
            result = f"Tool failed: {e}"

        return {
            "observations": [result],
            "pending_action": None
        }

    # --------------------------------------------------------

    def reflect_node(self, state: AgentState):

        answer = state.get("candidate_answer")
        if not answer:
            return {}

        prompt = f"""
Rate this answer 1-5.

5 = excellent
4 = good
3 = okay
2 = weak
1 = bad

If 5, extract 1-2 lessons.

Answer:
{answer}
"""

        try:
            txt = llm.invoke([SystemMessage(content=prompt)]).content
            rating = int(re.search(r"\d", txt).group())
        except:
            rating = 4

        # Non-destructive reflection
        return {
            "reflection_result": {"rating": rating},
            "lessons": ["Natural concise tone"] if rating == 5 else []
        }

    # --------------------------------------------------------

    def save_node(self, state: AgentState):

        answer = state.get("candidate_answer")
        rating = state.get("reflection_result", {}).get("rating", 0)

        if not answer or rating < 4:
            return {}

        try:
            sync_chat_history(
                self.pg_conn,
                state["user_id"],
                state["role"],
                self._get_last_human(state),
                answer,
                self.embeddings_model,
                self.logger
            )

            if rating == 5 and state.get("lessons"):
                self._update_style_guide(
                    state["user_id"],
                    state["role"],
                    state["lessons"]
                )

        except Exception as e:
            self.logger.error("Save failed", exc_info=True)

        return {}

    # ========================================================
    # ROUTING
    # ========================================================

    def route_after_generate(self, state: AgentState):
        if state.get("pending_action"):
            return "tool"
        if state.get("candidate_answer"):
            return "reflect"
        if state["iteration"] >= state["max_iterations"]:
            return END
        return "generator"

    def route_after_reflect(self, state: AgentState):
        return "save"

    # ========================================================
    # BUILD GRAPH
    # ========================================================

    def _build_graph(self):

        g = StateGraph(AgentState)

        g.add_node("memory", self.memory_node)
        g.add_node("generator", self.generator_node)
        g.add_node("tool", self.tool_node)
        g.add_node("reflect", self.reflect_node)
        g.add_node("save", self.save_node)

        g.add_edge(START, "memory")
        g.add_edge("memory", "generator")

        g.add_conditional_edges(
            "generator",
            self.route_after_generate,
            {
                "tool": "tool",
                "reflect": "reflect",
                "generator": "generator",
                END: END
            }
        )

        g.add_edge("tool", "generator")
        g.add_conditional_edges("reflect", self.route_after_reflect, {"save": "save"})
        g.add_edge("save", END)

        return g.compile(checkpointer=self.checkpointer)

    # ========================================================
    # RUN
    # ========================================================

    def run(self, user_id, role, text, thread_id=None):

        tid = thread_id or f"{user_id}_{role}_{int(datetime.now().timestamp())}"

        state = {
            "messages": [HumanMessage(content=text)],
            "user_id": user_id,
            "role": role,
            "thread_id": tid,
            "current_time": datetime.now(ZoneInfo("Asia/Kolkata")),
            "iteration": 0,
            "max_iterations": self.max_iterations,
        }

        result = self.graph.invoke(
            state,
            config={"configurable": {"thread_id": tid}}
        )

        return result.get("candidate_answer")

    def save_graph_visualization(self, graph_dir: Path, save_png: bool = False):
        try:
            graph_dir.mkdir(parents=True, exist_ok=True)
            md_path = graph_dir / "personalAgent_reflection.md"
            mermaid = self.graph.get_graph().draw_mermaid()
            md_path.write_text(f"```mermaid\n{mermaid}\n```")

            if save_png:
                png_path = graph_dir / "personalAgent_reflection.png"
                png_bytes = self.graph.get_graph().draw_mermaid_png()
                png_path.write_bytes(png_bytes)
                self.logger.info(f"PNG saved: {png_path}")

            self.logger.info(f"Graph saved: {md_path}")
        except Exception as e:
            self.logger.warning(f"Graph viz failed: {e}")

# =====================================================
# Example usage (commented out)
# =====================================================

# if __name__ == "__main__":
#     import logging
#     from Agent.db_postgresql import get_pg_connection

#     logging.basicConfig(level=logging.INFO)
#     logger = logging.getLogger("personal-agent")

#     conn = get_pg_connection(logger)

#     agent = PersonalAssistanceAgent(
#         pg_conn=conn,
#         embeddings_model=embeddings_model,
#         logger=logger,
#         max_iterations=5,
#     )

#     agent.save_graph_visualization(Path("graphs"), save_png=False)

#     response = agent.run(
#         user_id="PawanGunjan",
#         role="Mentor",
#         text="How should I structure my preparation for FAANG interviews in 2026?",
#     )

#     print("Final response:")
#     print(response or "[No response]")