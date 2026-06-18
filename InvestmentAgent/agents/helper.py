
import re
import json
import ast
from typing import Any, Dict, Optional
from agents.agent_state import InvestmentAgentState

# =============================
# Helper Functions
# =============================
def format_tool_history(results: list) -> str:
    if not results:
        return "None"
    lines = [
        f"Step {r['step']}:\n"
        f"Tool: {r['tool']}\n"
        f"Args: {r['args']}\n"
        f"Result: {r['result']}\n"
        for r in results
    ]
    return "\n".join(lines)

def compact_tool_result(result: Any) -> str:
    """Compact tool output to save tokens (critical for long conversations)."""
    if isinstance(result, dict):
        if result.get("status") == "success":
            data = result.get("data", {})
            short = {
                "symbol": result.get("symbol"),
                "last_price": data.get("last_price"),
                "change_percent": data.get("change_percent"),
                "volume": data.get("volume"),
                "timestamp": data.get("timestamp"),
            }
            return json.dumps({k: v for k, v in short.items() if v is not None}, default=str)
        else:
            return json.dumps(result, default=str)[:1500]
    return str(result)[:1500]

def format_state_for_prompt(state: InvestmentAgentState) -> Dict[str, Any]:
    """Clean, readable formatting for the system prompt."""
    return {
        "current_datetime": state.current_datetime.strftime("%Y-%m-%d %H:%M IST"),
        "attempt_count": state.attempt_count,
        "company_name": ", ".join(state.company_name) if state.company_name else "None",
        "symbols": json.dumps(state.symbols, indent=2) if state.symbols else "None",
        "prices": json.dumps(
            {k: {kk: vv for kk, vv in v.items() if kk in ["last_price", "change_percent", "volume"]}
             for k, v in state.prices.items()},
            default=str
        ) if state.prices else "None",
        "news": json.dumps(state.news, default=str) if state.news else "None",
        "memory": "\n".join(state.memory[-5:]) if state.memory else "None",  # last 5 only
        "tool_history": format_tool_history(state.tool_history),
    }

def clean_final_advice(content: str) -> Optional[str]:
    """Safety net: ensures the exact format and removes any junk."""
    content = content.strip()
    if not content.startswith("Stock-Specific Advice"):
        return None

    # Keep only the block (in case model added extra text)
    lines = content.splitlines()
    output = []
    in_block = False
    for line in lines:
        if line.strip().startswith("Stock-Specific Advice"):
            in_block = True
            output.append(line)
            continue
        if in_block:
            output.append(line)
    return "\n".join(output) if len(output) >= 6 else None


def extract_python_dict(text: str):
    if not text:
        return None

    # Remove special tokens
    text = text.replace("<|python_tag|>", "").strip()

    # Extract dict using regex
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None

    try:
        data = json.loads(match.group())
    except Exception:
        return None

    # Normalize format
    if "return" in data:
        return data["return"]

    return data