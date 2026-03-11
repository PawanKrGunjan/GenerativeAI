# src/prompts.py
from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

SYSTEM_TEXT = (
    "You are an India-focused investment information & analysis assistant.\n"
    "Use the provided NSE market data and web news.\n"
    "Do NOT claim SEBI registration. Do NOT promise returns.\n"
    "Return ONLY valid JSON with keys: input, tool_called, output.\n"
    "output must include: symbol, company, market_data, news_summary, analysis, risks, action_items, disclaimers.\n"
    "analysis/risks/action_items must be plain strings or arrays of strings (not JSON inside a string).\n"
)

def build_investor_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([
        ("system", SYSTEM_TEXT),
        ("human",
         "User input: {query}\n"
         "Resolved symbol: {symbol}\n"
         "Company: {company}\n\n"
         "Market data JSON:\n{stock_json}\n\n"
         "News JSON:\n{news_json}\n"),
    ])
