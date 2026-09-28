"""schemas/intel.py"""

from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel


class IntelEntry(BaseModel):
    fixture_id: int
    minute: int
    narration_type: str  # tactical | event_reaction | xg_divergence
    narrative: str
    score: float  # narratability score
    rag_docs_used: int
    via: str  # mistral | groq | template
    updated_at: datetime

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class IntelFeed(BaseModel):
    fixture_id: int
    entries: List[IntelEntry]
    updated_at: Optional[datetime] = None

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}
