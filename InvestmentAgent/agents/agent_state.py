from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, TypedDict
from langchain_core.messages import BaseMessage


class InvestmentState(TypedDict, total=False):
    tool_calls: Optional[List[Dict[str, Any]]]
    invalid_data: Optional[bool]
    analysis_tools_needed: Optional[bool]
    planner_output: Optional[str]
    data: Optional[Dict[str, Any]]
    analysis: Optional[Dict[str, Any]]
    reflection: Optional[str]

from pydantic import BaseModel, Field
from typing import List, Dict, Any
from langchain_core.messages import BaseMessage
from datetime import datetime

class InvestmentAgentState(BaseModel):

    messages: List[BaseMessage]

    attempt_count: int = 0

    symbols: List[str] = Field(default_factory=list)

    #dt: datetime

    memory: Dict[str, Any] = Field(default_factory=dict)

    tool_results: Dict[str, Any] = Field(default_factory=dict)

    indicator_cache: Dict[str, Any] = Field(default_factory=dict)

    sentiment: Dict[str, Any] = Field(default_factory=dict)