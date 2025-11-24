# src/storage/__init__.py
"""Storage backends."""

from .sqlite_vector import SQLiteVectorStorage

__all__ = ["SQLiteVectorStorage"]
