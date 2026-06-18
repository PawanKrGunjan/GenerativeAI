"""
response_formatter.py

Formats the final user-facing response using InvestmentAgentState.
The agent returns structured output in state.result.
"""

from agents.investment_agent import InvestmentAgentState


def _clean_reasoning(text) -> str:
    """
    Normalize reasoning text.
    Supports both string and dict inputs.
    """

    if not text:
        return ""

    # Structured agent output
    if isinstance(text, dict):
        reasoning = text.get("reasoning", [])

        if isinstance(reasoning, list):
            text = "\n".join(f"• {item}" for item in reasoning)
        else:
            text = str(reasoning)

    text = str(text)

    lines = text.strip().splitlines()

    cleaned = []

    for line in lines:
        stripped = line.strip()

        if stripped.startswith("Stock-Specific Advice"):
            continue

        if stripped and set(stripped) == {"─"}:
            continue

        cleaned.append(line)

    return "\n".join(cleaned).strip()


def format_final_response(state: InvestmentAgentState) -> str:
    """
    Construct final user response.
    """

    # Debug dump
    try:
        with open("agent_state.json", "w") as f:
            f.write(state.model_dump_json(indent=2))
    except Exception:
        pass

    # --------------------------------------------------
    # Defaults
    # --------------------------------------------------

    company = "Unknown Company"
    symbol = "N/A"
    price = "N/A"
    low_52 = "N/A"
    high_52 = "N/A"
    signal = "UNKNOWN"
    confidence = 0

    # --------------------------------------------------
    # Structured result
    # --------------------------------------------------

    result = state.result or {}

    if isinstance(result, dict):

        company = result.get("company", company)

        symbol = result.get("ticker", symbol)

        price = result.get("price", price)

        signal = result.get("signal", signal)

        confidence = result.get("confidence", confidence)

        reasoning = _clean_reasoning(result)

    else:
        reasoning = _clean_reasoning(result)

    # --------------------------------------------------
    # Price enrichment from state.prices
    # --------------------------------------------------

    if (
        symbol != "N/A"
        and isinstance(state.prices, dict)
        and symbol in state.prices
    ):
        p = state.prices[symbol]

        if isinstance(p, dict):
            low_52 = p.get("52w_low", low_52)
            high_52 = p.get("52w_high", high_52)

            if price == "N/A":
                price = p.get("price", price)

    # --------------------------------------------------
    # Time
    # --------------------------------------------------

    try:
        time_str = state.current_datetime.strftime(
            "%Y-%m-%d %H:%M IST"
        )
    except Exception:
        time_str = "N/A"

    # --------------------------------------------------
    # News
    # --------------------------------------------------

    news_lines = []

    if isinstance(state.news, dict):

        for items in state.news.values():

            if isinstance(items, list):
                for item in items[:3]:
                    news_lines.append(f"• {item}")

            elif isinstance(items, str):
                news_lines.append(f"• {items}")

    if not news_lines:
        news_lines.append("No major recent news.")

    news_section = "\n".join(news_lines)

    # --------------------------------------------------
    # Final response
    # --------------------------------------------------

    return f"""
Stock-Specific Advice
─────────────────────

Company      : {company}
Symbol       : {symbol}
Price        : {price}
52W Low      : {low_52}
52W High     : {high_52}

Signal       : {signal}
Confidence   : {confidence}%

Analysis Time: {time_str}

Reasoning
─────────
{reasoning}

Recent News
───────────
{news_section}
""".strip()