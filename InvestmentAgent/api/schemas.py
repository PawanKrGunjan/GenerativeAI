"""
schemas.py
Pydantic schemas for FastAPI request/response models
"""

from typing import Any, Dict,List, Optional
from pydantic import BaseModel, Field


# ─────────────────────────────────────────────
# Base Schema
# ─────────────────────────────────────────────

class BaseSchema(BaseModel):
    """Base model with common configuration."""

    class Config:
        from_attributes = True
        extra = "ignore"


# ─────────────────────────────────────────────
# Chat
# ─────────────────────────────────────────────

class ChatRequest(BaseSchema):
    """User chat message sent to AI agent."""

    message: str = Field(
        ...,
        description="User message to the AI investment agent",
        example="Analyze Reliance stock"
    )


class ChatResponse(BaseSchema):
    """AI response returned to the client."""

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
    """Request to run a tool from the tool registry."""

    tool_name: str = Field(
        ...,
        description="Name of the tool to execute",
        example="stock_price"
    )

    args: Dict[str, Any] = Field(
        default_factory=dict,
        description="Arguments passed to the tool"
    )


class ToolResponse(BaseSchema):
    """Tool execution response."""

    status: str = Field(
        ...,
        description="Execution status",
        example="success"
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
# Agent Response
# ─────────────────────────────────────────────

class AgentResult(BaseSchema):
    """
    Internal agent response structure.
    Used between the agent and API layer.
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
        description="Optional memory summary from the agent"
    )

# ─────────────────────────────────────────────
# Portfolio
# ─────────────────────────────────────────────


class PortfolioStock(BaseSchema):
    """Single portfolio stock"""

    symbol: str = Field(
        ...,
        description="Stock symbol",
        example="BEL"
    )

    qty: float = Field(
        ...,
        description="Quantity held",
        example=10
    )


class PortfolioRequest(BaseSchema):
    """Portfolio save request"""

    portfolio: List[PortfolioStock]


class PortfolioResponse(BaseSchema):
    """Portfolio response"""

    portfolio: List[PortfolioStock]