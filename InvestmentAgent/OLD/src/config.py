# src/config.py
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

from langchain_core.globals import set_debug, set_verbose
from langchain_community.utilities.duckduckgo_search import DuckDuckGoSearchAPIWrapper
from langchain_ollama import ChatOllama


@dataclass(frozen=True)
class Config:
    ollama_model: str
    temperature: float
    ddg_results: int
    checkpoint_db: str
    master_stock_file: str
    debug: bool

    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            ollama_model=os.getenv("OLLAMA_MODEL", "granite4:350m"),
            temperature=float(os.getenv("TEMPERATURE", "0.0")),
            ddg_results=int(os.getenv("DDG_RESULTS", "5")),
            checkpoint_db=os.getenv("CHECKPOINT_DB", "data/invest_agent_checkpoints.sqlite"),
            master_stock_file=os.getenv("MASTER_STOCK_FILE", "data/all_nse_stocks.json"),
            debug=(os.getenv("AGENT_DEBUG", "0") == "1"),
        )


def apply_langchain_debug(debug: bool) -> None:
    # LangChain globals expect bool. [web:159]
    set_debug(debug)
    set_verbose(debug)


def build_llm(cfg: Config) -> ChatOllama:
    return ChatOllama(
        model=cfg.ollama_model,
        temperature=cfg.temperature,
        verbose=cfg.debug,
    )


def build_ddg_api() -> DuckDuckGoSearchAPIWrapper:
    return DuckDuckGoSearchAPIWrapper()


def load_nse_master(master_stock_file: str) -> tuple[list[dict[str, str]], dict[str, dict[str, str]], list[tuple[str, str]]]:
    with open(master_stock_file, "r", encoding="utf-8") as f:
        nse_stocks: list[dict[str, str]] = json.load(f)

    symbol_index: dict[str, dict[str, str]] = {row["symbol"].upper(): row for row in nse_stocks}
    name_index: list[tuple[str, str]] = [
        (row["symbol"].upper(), row["companyName"].lower()) for row in nse_stocks
    ]
    return nse_stocks, symbol_index, name_index
