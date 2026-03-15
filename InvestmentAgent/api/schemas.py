"""
schemas.py
Pydantic schemas for FastAPI request/response models.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, ConfigDict


# ─────────────────────────────────────────────
# Base Schema
# ─────────────────────────────────────────────

class BaseSchema(BaseModel):
    """
    Base schema providing common configuration
    used by all API models.
    """

    model_config = ConfigDict(
        from_attributes=True,
        extra="ignore"
    )


# ─────────────────────────────────────────────
# Chat
# ─────────────────────────────────────────────

class ChatRequest(BaseSchema):
    """
    User chat request sent to the AI investment agent.
    """

    message: str = Field(
        ...,
        description="User message to the AI investment agent",
        examples=["Analyze Reliance stock"]
    )

    session_id: Optional[str] = Field(
        default=None,
        description="Session identifier for maintaining conversation memory"
    )


class ChatResponse(BaseSchema):
    """
    AI response returned to the client.
    """

    answer: str = Field(
        ...,
        description="AI generated response"
    )

    time: str = Field(
        ...,
        description="Response timestamp in IST"
    )


# ─────────────────────────────────────────────
# Tool Execution
# ─────────────────────────────────────────────

class ToolRequest(BaseSchema):
    """
    Request to execute a tool from the tool registry.
    """

    tool_name: str = Field(
        ...,
        description="Name of the tool to execute",
        examples=["get_stock_info"]
    )

    args: Dict[str, Any] = Field(
        default_factory=dict,
        description="Arguments passed to the tool"
    )


class ToolResponse(BaseSchema):
    """
    Response returned after tool execution.
    """

    status: str = Field(
        ...,
        description="Execution status",
        examples=["success", "error"]
    )

    result: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Result returned by the tool"
    )

    error: Optional[str] = Field(
        default=None,
        description="Error message if execution failed"
    )


# ─────────────────────────────────────────────
# Agent Response (Internal)
# ─────────────────────────────────────────────

class AgentResult(BaseSchema):
    """
    Internal agent response structure used between
    the AI agent and API layer.
    """

    answer: str = Field(
        ...,
        description="Agent generated answer"
    )

    current_time_ist: str = Field(
        ...,
        description="Timestamp when the answer was generated"
    )

    memory_summary: Optional[str] = Field(
        default=None,
        description="Optional memory summary returned by the agent"
    )


# ─────────────────────────────────────────────
# Portfolio
# ─────────────────────────────────────────────

class PortfolioStock(BaseSchema):
    """
    Single stock entry in a portfolio.
    """

    symbol: str = Field(
        ...,
        description="Stock symbol",
        examples=["BEL"]
    )

    qty: float = Field(
        ...,
        description="Quantity held",
        examples=[10]
    )


class PortfolioRequest(BaseSchema):
    """
    Request to save or update a user portfolio.
    """

    portfolio: List[PortfolioStock] = Field(
        ...,
        description="List of portfolio stocks"
    )


class PortfolioResponse(BaseSchema):
    """
    Portfolio response returned by the API.
    """

    portfolio: List[PortfolioStock] = Field(
        ...,
        description="Portfolio holdings"
    )