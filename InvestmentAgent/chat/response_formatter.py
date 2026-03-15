"""
response_formatter.py

Formats the final user-facing response using InvestmentAgentState.
The LLM only produces reasoning (Signal, Confidence, Points).
All stock metadata is injected here from the agent state.
"""

from agents.investment_agent import InvestmentAgentState


def _clean_reasoning(text: str) -> str:
    """
    Remove the header and separators the LLM may produce.
    Keeps only Signal, Confidence, and Pointwise Reasoning.
    """

    if not text:
        return ""

    lines = text.strip().splitlines()

    cleaned = []
    for line in lines:
        if line.strip().startswith("Stock-Specific Advice"):
            continue
        if set(line.strip()) == {"─"}:
            continue
        cleaned.append(line)

    return "\n".join(cleaned).strip()


def format_final_response(state: InvestmentAgentState) -> str:
    """
    Construct the final response shown to the user.

    Data Sources
    ------------
    Company name   → state.company_name
    Symbol         → state.symbols
    Prices         → state.prices
    News           → state.news
    Time           → state.current_datetime
    Reasoning      → state.result
    """
    with open("agent_state.json", "w") as f:
        f.write(state.model_dump_json(indent=2))


    # --------------------------------------------------
    # Symbol + Company
    # --------------------------------------------------

    company = state.company_name or "Unknown Company"

    symbol = "N/A"
    if state.symbols:
        symbol = state.symbols[0].get("symbol", "N/A")

    # --------------------------------------------------
    # Price data
    # --------------------------------------------------

    price = "N/A"
    low_52 = "N/A"
    high_52 = "N/A"
    
    if symbol in state.prices:
        p = state.prices[symbol]

        price = p.get("price", "N/A")
        low_52 = p.get("52w_low", "N/A")
        high_52 = p.get("52w_high", "N/A")

    # --------------------------------------------------
    # Time
    # --------------------------------------------------

    time_str = state.current_datetime.strftime("%Y-%m-%d %H:%M IST")

    # --------------------------------------------------
    # LLM reasoning (cleaned)
    # --------------------------------------------------

    reasoning = _clean_reasoning(state.result or "Analysis complete.")

    # --------------------------------------------------
    # News extraction
    # --------------------------------------------------

    news_lines = []

    for items in state.news.values():

        if isinstance(items, list):
            for n in items[:3]:
                news_lines.append(f"• {n}")

        elif isinstance(items, str):
            news_lines.append(f"• {items}")

    if not news_lines:
        news_lines.append("No major recent news.")

    news_section = "\n".join(news_lines)

    # --------------------------------------------------
    # Final formatted response
    # --------------------------------------------------

    return f"""
Stock-Specific Advice
─────────────────────
{reasoning}

Recent News
───────────
{news_section}
""".strip()