from enum import Enum
from typing import List
from pydantic import BaseModel, Field

class CleanlinessStatus(str, Enum):
    CLEAN = "Clean"
    DIRTY = "Dirty"

class InspectionResult(BaseModel):
    status: CleanlinessStatus = Field(description="Final evaluation: Clean or Dirty")
    confidence: float = Field(description="Confidence score between 0.0 and 1.0")
    reasoning: str = Field(description="Detailed explanation citing Spills, Dust, Trash, or Stains")
    issues_detected: List[str] = Field(description="List of specific issues found")
