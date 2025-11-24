# src/models/search.py
"""Search result models."""

from dataclasses import dataclass
from typing import Optional, Dict, Any

from .processed import ProcessedMessage


@dataclass
class SearchResult:
    """Search result with relevance score."""

    message: ProcessedMessage
    score: float  # Similarity score (0.0 to 1.0, higher is better)
    rank: int  # Position in result list (1-indexed)
    matched_on: str = 'vector'  # 'vector', 'text', 'metadata'
    snippet: Optional[str] = None  # Highlighted snippet
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
