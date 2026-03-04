"""
Offline Personal Identity Agent (Self-Learning ReAct version)
Simple class structure — manual ReAct parsing + reflection + optional rating/correction
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
from Agent.config import DEFAULT_MAX_ITERATIONS, ENABLE_RESPONSE_RATING


# =====================================================
# STATE — with optional rating & correction
# =====================================================

class AgentState(TypedDict, total=False):
    messages: Annotated[List[BaseMessage], add_messages]
    observations: Annotated[List[str], add_messages]
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
    learned_examples: Annotated[List[str], add_messages]  # self-learned lessons
    rating_feedback: Optional[int]          # ← NEW: 1-10 rating (None if not rated)
    correction_text: Optional[str]          # ← NEW: user correction text


class PersonalAssistanceAgent:
    """
    Self-learning ReAct-based personal identity agent with optional user rating & correction.
    Feature can be disabled in production via config.
    """

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
        self.search_documents_tool = None
        self.search_role_chats_tool = None
        self.graph = self._build_graph()

    def get_last_human_message(self, state: AgentState) -> Optional[str]:
        for msg in reversed(state.get("messages", [])):
            if isinstance(msg, HumanMessage):
                return msg.content.strip()
        return None

    def _search_documents(self, query: str) -> str:
        try:
            docs = semantic_search(
                self.pg_conn, self.logger, query.strip(), top_k=4
            ) or []
            clean = [d.strip() for d in docs if d and d.strip()]
            return "\n\n".join(clean) if clean else "No relevant documents found."
        except Exception as exc:
            self.logger.error("search_documents failed", exc_info=True)
            return "Error retrieving documents."

    def _search_role_chats(self, query: str, user_id: str, role: str) -> str:
        try:
            rows = semantic_search_chat(
                self.pg_conn, self.logger, query.strip(),
                user_id, role, self.embeddings_model, top_k=4
            ) or []
            samples = [r.get("human_text", "").strip() for r in rows if r.get("human_text")]
            return "\n\n".join(samples) if samples else "No matching role-specific chat history."
        except Exception as exc:
            self.logger.error("search_role_chats failed", exc_info=True)
            return "Error retrieving chat history."

    def memory_node(self, state: AgentState) -> Dict[str, str]:
        query = self.get_last_human_message(state)
        if not query or not query.strip():
            return {}

        docs = semantic_search(
            self.pg_conn, self.logger, query, top_k=5
        ) or []
        chats = semantic_search_chat(
            self.pg_conn, self.logger, query,
            state["user_id"], state["role"], self.embeddings_model, top_k=5
        ) or []

        blocks = []

        if docs:
            clean_docs = [d.strip() for d in docs if d and d.strip()]
            if clean_docs:
                blocks.append("### USER PRIVATE DOCUMENTS\n" + "\n\n".join(clean_docs))

        style_samples = [r.get("human_text", "").strip() for r in chats if r.get("human_text")]
        if style_samples:
            blocks.append("### USER WRITING STYLE SAMPLES\n" + "\n\n".join(style_samples))

        # ── Fixed: extract .content from BaseMessage objects ─────────────────────
        learned_msgs = state.get("learned_examples", [])
        learned_texts = []
        for item in learned_msgs[-3:]:  # last 3
            if isinstance(item, BaseMessage) and item.content and item.content.strip():
                learned_texts.append(item.content.strip())
            elif isinstance(item, str) and item.strip():
                learned_texts.append(item.strip())

        if learned_texts:
            blocks.append("### LEARNED LESSONS FROM PAST RESPONSES\n" + "\n\n".join(learned_texts))

        # ── Feedback block (rating + correction) ─────────────────────────────────
        rating = state.get("rating_feedback")
        correction = state.get("correction_text", "")
        if rating is not None or correction:
            feedback_block = f"Previous response rating: {rating}/10"
            if correction:
                feedback_block += f"\nUser correction: {correction}"
            blocks.append("### USER FEEDBACK ON LAST RESPONSE\n" + feedback_block)

        if not blocks:
            return {}

        return {"memory_context": "\n\n".join(blocks)}

    def react_llm_node(self, state: AgentState) -> Dict[str, Any]:
        self.logger.info("Entering react_llm_node | iteration=%d", state.get("iteration", 0))

        context = state.get("memory_context", "")
        role = state["role"]
        user_id = state["user_id"]
        time_str = state.get("current_time", datetime.now(ZoneInfo("Asia/Kolkata"))) \
            .strftime("%d %B %Y, %I:%M %p IST")

        observations_block = "\n".join(state.get("observations", [])) or "(none yet)"

        # Feedback from rating/correction
        feedback_block = ""
        rating = state.get("rating_feedback")
        correction = state.get("correction_text", "")
        if rating is not None or correction:
            feedback_block = f"Previous response rating: {rating}/10\nUser correction: {correction or 'None'}\nUse this to improve."

        system_prompt = f"""\
    You are {user_id}. You ARE the real person — not an AI.

    Speaking to: {role}
    Time in India: {time_str}

    Rules:
    - Speak **exactly** like {user_id} — copy tone, phrasing, rhythm, vocabulary.
    - Be natural, concise, human.
    - NEVER mention AI, tools, memory, prompts, reasoning steps.
    - Use memory, learned lessons, and feedback to improve answers.
    - If missing info → use tools in the format shown.
    - Max reasoning steps: {state.get("max_iterations", self.max_iterations)}
    - Learn from past: avoid mistakes noted in learned examples and user feedback.

    Tools:
    - search_documents(query)          → facts from private documents
    - search_role_chats(query)         → your past messages to this role

    ReAct format:
    Thought: <reasoning>
    Action: tool_name{{"query": "..."}}   (only when needed)
    OR
    Final Answer: <your actual reply>

    ===== MEMORY & STYLE SAMPLES + LEARNED LESSONS + FEEDBACK =====
    {context}
    {feedback_block}
    ===== END MEMORY =====

    Previous observations:
    {observations_block}

    Respond to the last message.
    """

        messages = [SystemMessage(content=system_prompt)] + state["messages"]

        # Log prompt preview
        self.logger.debug("LLM prompt preview: %s...", system_prompt[:200])

        try:
            response = llm.invoke(messages)
            content = (response.content or "").strip()
            self.logger.info("LLM response received | length=%d | preview=%s...", 
                            len(content), content[:100])
        except Exception as exc:
            self.logger.error("LLM call failed", exc_info=True)
            content = "Sorry, something went wrong on my side."

        new_state: Dict[str, Any] = {"iteration": state.get("iteration", 0) + 1}

        # Parse output with logging
        if "Final Answer:" in content:
            final = content.split("Final Answer:", 1)[1].strip()
            new_state["messages"] = [AIMessage(content=final)]
            self.logger.info("Parsed Final Answer | content=%s...", final[:100])
            return new_state

        # Action parsing
        action_match = re.search(
            r"(?i)Action\s*:\s*(\w+)\s*\{\s*(.*?)\s*\}",
            content, re.DOTALL
        )

        if action_match:
            tool_name = action_match.group(1).strip()
            args_str = action_match.group(2).strip()

            try:
                if not args_str.startswith("{"):
                    args_str = "{" + args_str
                if not args_str.endswith("}"):
                    args_str += "}"
                args = json.loads(args_str)
                new_state["pending_action"] = {"tool": tool_name, "args": args}
                self.logger.info("Parsed Action | tool=%s | args=%s", tool_name, args)
            except json.JSONDecodeError as e:
                self.logger.warning("Failed to parse action args | raw=%s | error=%s", args_str, str(e))
                new_state["observations"] = ["Observation: Could not parse tool arguments."]
        else:
            # Thought only — log as fallback
            new_state["messages"] = [AIMessage(content=content)]
            self.logger.warning("No Final Answer or Action found | treating as thought | content=%s...", content[:100])

        self.logger.info("Exiting react_llm_node")
        return new_state


    def tool_caller_node(self, state: AgentState) -> Dict[str, Any]:
        self.logger.info("Entering tool_caller_node")

        action = state.get("pending_action")
        if not action:
            self.logger.warning("No pending action found")
            return {}

        tool_name = action["tool"]
        args = action.get("args", {})

        self.logger.info("Executing tool | name=%s | args=%s", tool_name, args)

        try:
            if tool_name == "search_documents":
                result = self.search_documents_tool.invoke({"query": args.get("query", "")})
            elif tool_name == "search_role_chats":
                result = self.search_role_chats_tool.invoke({
                    "query": args.get("query", ""),
                    "user_id": state["user_id"],
                    "role": state["role"]
                })
            else:
                result = f"Unknown tool: {tool_name}"

            # Simple self-correction check
            if "Error" in result or "No relevant" in result or len(result) < 20:
                result += " (This result was weak — consider refining query or skipping.)"

            self.logger.info("Tool result | length=%d | preview=%s...", len(result), result[:100])

        except Exception as exc:
            self.logger.error("Tool execution failed | tool=%s", tool_name, exc_info=True)
            result = "Tool failed to execute."

        self.logger.info("Exiting tool_caller_node")
        return {
            "observations": [f"[{tool_name}] {result}"],
            "pending_action": None
        }


    def reflection_node(self, state: AgentState) -> Dict[str, Any]:
        self.logger.info("Entering reflection_node")

        last_ai = ""
        for msg in reversed(state["messages"]):
            if isinstance(msg, AIMessage) and msg.content.strip():
                last_ai = msg.content.strip()
                break

        rating = state.get("rating_feedback")
        correction = state.get("correction_text", "")

        reflection_prompt = f"""\
    You just gave this response: {last_ai[:500]}

    User rating: {rating}/10 (if rated)
    User correction: {correction or 'None'}

    Critique briefly:
    - Was it accurate, concise, and in character?
    - What could be improved based on rating/correction?
    - Summarize as one concise learning example for future.

    Output ONLY: Learned: <one sentence lesson>
    """

        self.logger.debug("Reflection prompt preview: %s...", reflection_prompt[:200])

        try:
            response = llm.invoke([SystemMessage(content=reflection_prompt)])
            learned = response.content.strip()

            # Clean up common prefixes
            for prefix in ["Learned:", "Learned example:", "Lesson:"]:
                if prefix in learned:
                    learned = learned.split(prefix, 1)[1].strip()
                    break

            # Fallback
            if len(learned) < 10 or len(learned) > 300:
                learned = learned[:200] + "..." if len(learned) > 200 else "No useful lesson this time."

            self.logger.info("Reflection learned: %s (rating=%s, correction=%s)", 
                            learned, rating, bool(correction))

            return {"learned_examples": [learned]}

        except Exception as exc:
            self.logger.warning("Reflection failed", exc_info=True)
            return {}


    def save_chat_node(self, state: AgentState) -> Dict:
        user_input = self.get_last_human_message(state)

        ai_response = None
        for msg in reversed(state["messages"]):
            if isinstance(msg, AIMessage) and msg.content.strip():
                ai_response = msg.content.strip()
                break

        rating = state.get("rating_feedback")
        correction = state.get("correction_text", "")

        # NEW: Skip saving if rating is enabled and < 4
        if ENABLE_RESPONSE_RATING and rating is not None and rating < 4:
            self.logger.info("Response NOT saved | rating=%d < 4", rating)
            return {}

        if user_input and ai_response:
            full_ai = ai_response
            if correction:
                full_ai += f"\n\n[User correction]: {correction}"
            if rating is not None:
                full_ai += f"\n\n[User rating]: {rating}/10"

            try:
                sync_chat_history(
                    self.pg_conn,
                    state["user_id"],
                    state["role"],
                    user_input,
                    full_ai,
                    self.embeddings_model,
                    self.logger,
                )
                self.logger.info("Chat saved | rating=%s | correction=%s", rating, bool(correction))
            except Exception as exc:
                self.logger.error("Failed to save chat history", exc_info=True)

        return {}

    def route_after_llm(self, state: AgentState) -> str:
        if state.get("pending_action"):
            return "tool_caller"

        if state.get("iteration", 0) >= state.get("max_iterations", self.max_iterations):
            return "save_chat"

        last_msg = state.get("messages", [])[-1] if state.get("messages") else None
        if isinstance(last_msg, AIMessage) and "Final Answer:" in last_msg.content:
            return "reflection" if ENABLE_RESPONSE_RATING else "save_chat"  # only reflect if rating enabled

        return "react_llm"

    def _build_graph(self):
        search_documents_tool = StructuredTool.from_function(
            func=self._search_documents,
            name="search_documents",
            description="Search the user's private documents for relevant facts or details.",
        )

        search_role_chats_tool = StructuredTool.from_function(
            func=self._search_role_chats,
            name="search_role_chats",
            description="Search past conversations with the current role. Returns USER messages only.",
        )

        self.search_documents_tool = search_documents_tool
        self.search_role_chats_tool = search_role_chats_tool

        workflow = StateGraph(AgentState)

        workflow.add_node("memory", self.memory_node)
        workflow.add_node("react_llm", self.react_llm_node)
        workflow.add_node("tool_caller", self.tool_caller_node)
        workflow.add_node("save_chat", self.save_chat_node)
        workflow.add_node("reflection", self.reflection_node)

        workflow.add_edge(START, "memory")
        workflow.add_edge("memory", "react_llm")

        workflow.add_conditional_edges(
            "react_llm",
            self.route_after_llm,
            {
                "tool_caller": "tool_caller",
                "save_chat": "save_chat",
                "react_llm": "react_llm",
                "reflection": "reflection"
            }
        )

        workflow.add_edge("tool_caller", "react_llm")
        workflow.add_edge("save_chat", "reflection")
        workflow.add_edge("reflection", END)

        return workflow.compile(checkpointer=self.checkpointer)

    def save_graph_visualization(self, graph_dir: Path, save_png: bool = False) -> None:
        """Save Mermaid diagram + optional PNG."""
        try:
            graph_dir.mkdir(parents=True, exist_ok=True)

            md_path = graph_dir / "personalAgent_reflection.md"
            mermaid_code = self.graph.get_graph().draw_mermaid()
            md_path.write_text(f"```mermaid\n{mermaid_code}\n```")

            if save_png:
                try:
                    png_path = graph_dir / "personalAgent_reflection.png"
                    png_bytes = self.graph.get_graph().draw_mermaid_png()
                    png_path.write_bytes(png_bytes)
                    self.logger.info("Graph PNG saved: %s", png_path)
                except Exception as e:
                    self.logger.warning("PNG generation failed: %s", e)

            self.logger.info("Graph Markdown saved: %s", md_path)

        except Exception as e:
            self.logger.warning("Graph visualization failed: %s", e)

    def run(
        self,
        user_id: str,
        role: str,
        text: str,
        thread_id: Optional[str] = None,
        max_iterations: Optional[int] = None,
    ) -> Optional[str]:
        config = {"configurable": {"thread_id": thread_id or f"{user_id}_{role}"}}

        input_state = {
            "messages": [HumanMessage(content=text)],
            "user_id": user_id,
            "role": role,
            "current_time": datetime.now(ZoneInfo("Asia/Kolkata")),
            "max_iterations": max_iterations or self.max_iterations,
            "iteration": 0,
            "learned_examples": [],
            "rating_feedback": None,
            "correction_text": None,
        }

        try:
            final = None
            for event in self.graph.stream(input_state, config=config, stream_mode="values"):
                if "messages" in event and event["messages"]:
                    last = event["messages"][-1]
                    if isinstance(last, AIMessage):
                        final = last.content.strip()
            return final
        except Exception as e:
            self.logger.exception("Agent run failed")
            return None


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