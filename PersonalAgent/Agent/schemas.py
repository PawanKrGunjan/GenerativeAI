from typing import List
from pydantic import BaseModel, Field

class UserFact(BaseModel):
    key: str = Field(..., description="Fact key (name, location, salary, doj, etc.)")
    value: str = Field(..., description="Exact literal value from text")
    confidence: float = Field(1.0, ge=0.0, le=1.0, description="1.0 for obvious facts, lower if uncertain")
    source_text: str = Field(..., description="Snippet of original text proving the fact")

class ExtractedFacts(BaseModel):
    facts: List[UserFact] = Field(default_factory=list, description="All extracted facts; empty if none")