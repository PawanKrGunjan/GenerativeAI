from typing import List

from pydantic import BaseModel, Field


class UserFact(BaseModel):
    key: str = Field(..., description="Fact key (name, location, salary, doj, etc.)")
    value: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    source_text: str


class ExtractedFacts(BaseModel):
    facts: List[UserFact] = Field(default_factory=list)
