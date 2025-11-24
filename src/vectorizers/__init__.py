# src/vectorizers/__init__.py
"""Text vectorizers (embeddings)."""

from .sentence_transformer import SentenceTransformerVectorizer

__all__ = ["SentenceTransformerVectorizer"]
