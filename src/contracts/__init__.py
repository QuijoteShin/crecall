# src/contracts/__init__.py
"""Abstract contracts for pluggable components."""

from .detector import IFormatDetector
from .importer import IImporter
from .normalizer import INormalizer
from .classifier import IClassifier
from .vectorizer import IVectorizer
from .storage import IStorage

__all__ = [
    "IFormatDetector",
    "IImporter",
    "INormalizer",
    "IClassifier",
    "IVectorizer",
    "IStorage",
]
