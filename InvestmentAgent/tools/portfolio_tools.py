"""
User portfolio tools.
Used by agents to analyze or load investor portfolio data.
Supports upload, save, and interactive analysis.
"""

import json
import numpy as np
from typing import Dict, Any
from pathlib import Path

import pandas as pd
from langchain_core.tools import tool
from fastapi import UploadFile, File  # For file upload handling

from utils.logger import LOGGER
from utils.config import DATA_DIR


def _load_portfolio_dataframe(file_path: Path) -> pd.DataFrame:
    """Load portfolio data from CSV, XLSX, or JSON into a DataFrame."""
    suffix = file_path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(file_path)
    
    if suffix in [".xlsx", ".xls"]:
        return pd.read_excel(file_path)
    
    if suffix == ".json":
        with open(file_path) as f:
            data = json.load(f)
        
        if isinstance(data, dict) and "holdings" in data:
            return pd.DataFrame(data["holdings"])
        
        return pd.DataFrame(data)
    
    raise ValueError(f"Unsupported file format: {suffix}")


def _clean_dataframe_for_json(df: pd.DataFrame) -> pd.DataFrame:
    """Replace NaN/inf with None for JSON serialization."""
    df = df.copy()
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.where(pd.notna(df), None)
    return df


def _save_portfolio_file(uploaded_file: UploadFile, filename: str = None) -> Path:
    """Save uploaded file to DATA_DIR with unique/safe name."""
    if not filename:
        filename = f"holdings-{Path(uploaded_file.filename).stem}-{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}{Path(uploaded_file.filename).suffix}"
    
    save_path = DATA_DIR / filename
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, "wb") as buffer:
        content = uploaded_file.file.read()
        buffer.write(content)
    
    LOGGER.info("Saved portfolio file: %s", save_path)
    return save_path


@tool
def upload_portfolio(file_content: str, filename: str = "portfolio.xlsx") -> Dict[str, Any]:
    """
    Upload and save portfolio file to DATA_DIR for analysis.
    
    Args:
        file_content: Base64 encoded file content or file path
        filename: Save as this name (auto-generates timestamp if exists)
    
    Returns saved file path for immediate analysis.
    """
    try:
        # For now, treat as path (extend for base64 later)
        path = Path(file_content) if Path(file_content).exists() else None
        
        if path and path.exists():
            # Copy existing file to DATA_DIR
            save_path = DATA_DIR / filename
            path.copy(save_path)
            LOGGER.info("Copied portfolio file: %s -> %s", path, save_path)
        else:
            return {"status": "error", "message": "File not found. Use absolute path."}
        
        df = _load_portfolio_dataframe(save_path)
        return {
            "status": "success",
            "saved_path": str(save_path),
            "rows": len(df),
            "columns": list(df.columns),
            "sample": _clean_dataframe_for_json(df.head(3)).to_dict(orient="records")
        }
        
    except Exception as e:
        LOGGER.exception("upload_portfolio failed")
        return {"status": "error", "message": str(e)}


@tool
def list_portfolio_files() -> Dict[str, Any]:
    """List all portfolio files in DATA_DIR."""
    portfolio_files = list(DATA_DIR.glob("holdings*.xlsx")) + list(DATA_DIR.glob("holdings*.csv"))
    
    return {
        "status": "success",
        "files": [f.name for f in portfolio_files],
        "count": len(portfolio_files)
    }


@tool
def analyze_portfolio(file_path: str) -> Dict[str, Any]:
    """
    Interactive portfolio analysis. Supports any holdings-*.xlsx/csv in DATA_DIR.
    
    Required columns: symbol, quantity, price, sector (optional)
    Auto-finds recent files if path ends with '*'.
    """
    LOGGER.info("analyze_portfolio | file=%s", file_path)

    try:
        path = Path(file_path)
        
        # Auto-find if wildcard or not exact match
        if not path.exists():
            candidates = list(DATA_DIR.glob("holdings*"))
            if candidates:
                path = candidates[-1]  # Most recent
                LOGGER.info("Auto-selected latest file: %s", path)
        
        if not path.exists():
            return {"status": "error", "message": f"File not found: {file_path}"}
        
        df = _load_portfolio_dataframe(path)
        required_cols = {"symbol", "quantity", "price"}
        
        if not required_cols.issubset(df.columns):
            return {
                "status": "error",
                "message": f"Missing columns: {required_cols - set(df.columns)}. Found: {list(df.columns)}"
            }

        df["position_value"] = pd.to_numeric(df["quantity"], errors="coerce") * pd.to_numeric(df["price"], errors="coerce")
        total_value = float(df["position_value"].sum() or 0)
        
        sector_exposure = {}
        if "sector" in df.columns:
            sector_exposure = df.groupby("sector")["position_value"].sum().to_dict()
            sector_exposure = {k: float(v) if pd.notna(v) else 0.0 for k, v in sector_exposure.items()}
        
        top_holdings = _clean_dataframe_for_json(
            df.nlargest(5, "position_value")[["symbol", "quantity", "price", "position_value"]]
        ).to_dict(orient="records")

        return {
            "status": "success",
            "file": str(path),
            "total_value": total_value,
            "holdings_count": len(df),
            "top_holdings": top_holdings,
            "sector_exposure": sector_exposure,
            "avg_price": float(df["price"].mean() or 0)
        }

    except Exception as e:
        LOGGER.exception("analyze_portfolio failed")
        return {"status": "error", "message": str(e)}


@tool
def get_user_portfolio(filename: str = None) -> Dict[str, Any]:
    """
    Load specific portfolio file. Defaults to most recent holdings-*.xlsx.
    """
    if filename:
        path = DATA_DIR / filename
    else:
        paths = list(DATA_DIR.glob("holdings-*.xlsx")) + list(DATA_DIR.glob("holdings-*.csv"))
        path = paths[-1] if paths else None
    
    if not path or not path.exists():
        return {"status": "no_data", "holdings": [], "available_files": [p.name for p in DATA_DIR.glob("holdings*")]}
    
    LOGGER.info("get_user_portfolio | file=%s", path)
    
    try:
        df = _load_portfolio_dataframe(path)
        df_clean = _clean_dataframe_for_json(df)
        
        return {
            "status": "success",
            "file": path.name,
            "holdings": df_clean.to_dict(orient="records"),
            "shape": [len(df), len(df.columns)]
        }
    except Exception as e:
        LOGGER.exception("get_user_portfolio failed")
        return {"status": "error", "message": str(e)}
