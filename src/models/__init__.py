# src/models/__init__.py
"""Data models for DRecall."""

from .normalized import NormalizedMessage, Attachment
from .processed import ProcessedMessage, Classification
from .search import SearchResult

__all__ = [
    "NormalizedMessage",
    "Attachment",
    "ProcessedMessage",
    "Classification",
    "SearchResult",
]
