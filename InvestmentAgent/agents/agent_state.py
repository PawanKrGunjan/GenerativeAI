from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, TypedDict
from langchain_core.messages import BaseMessage
from datetime import datetime
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")

class InvestmentAgentState(BaseModel):
    messages: List[BaseMessage] = Field(default_factory=list)

    company_name: List[str] = Field(default_factory=list)
    symbols: Dict[str, str] = Field(default_factory=dict)

    prices: Dict[str, Any] = Field(default_factory=dict)
    news: Dict[str, Any] = Field(default_factory=dict)

    memory: List[str] = Field(default_factory=list)
    tool_history: List[Dict[str, Any]] = Field(default_factory=list)

    #result: Optional[str] = None
    result: Optional[Dict[str, Any]] = None

    attempt_count: int = 0
    current_datetime: datetime = Field(default_factory=lambda: datetime.now(IST))


# class InvestmentAgentState(TypedDict):
#     messages: List[Any]
#     company_name: str
#     symbols: Dict[str, str]
#     prices: Dict[str, float]
#     news: str
#     memory: List[str]
#     tool_history: List[str]
#     attempt_count: int
#     current_datetime: Any

#     # ✅ FIXED
#     result: Optional[Dict[str, Any]]