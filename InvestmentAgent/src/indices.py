# src/indices.py
from __future__ import annotations

import json
from dataclasses import dataclass


@dataclass(frozen=True)
class NseMasterIndex:
    symbol_index: dict[str, dict[str, str]]   # SYMBOL -> row
    name_index: list[tuple[str, str]]         # (SYMBOL, companyNameLower)


def load_nse_master(master_stock_file: str) -> NseMasterIndex:
    with open(master_stock_file, "r", encoding="utf-8") as f:
        nse_stocks: list[dict[str, str]] = json.load(f)

    symbol_index: dict[str, dict[str, str]] = {row["symbol"].upper(): row for row in nse_stocks}
    name_index: list[tuple[str, str]] = [
        (row["symbol"].upper(), row["companyName"].lower()) for row in nse_stocks
    ]

    return NseMasterIndex(symbol_index=symbol_index, name_index=name_index)
