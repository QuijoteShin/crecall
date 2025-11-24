# src/contracts/vectorizer.py
"""Abstract interface for text vectorization (embeddings)."""

from abc import ABC, abstractmethod
from typing import List
import numpy as np


class IVectorizer(ABC):
    """
    Generates embedding vectors from text.

    Implementations can use:
    - Sentence Transformers (local)
    - OpenAI Embeddings API
    - Custom models
    """

    @abstractmethod
    def vectorize(self, text: str) -> np.ndarray:
        """
        Generate embedding vector for single text.

        Args:
            text: Normalized text content

        Returns:
            NumPy array of embedding vector
        """
        pass

    @abstractmethod
    def vectorize_batch(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for multiple texts efficiently.

        Args:
            texts: List of normalized text contents

        Returns:
            2D NumPy array (n_texts, embedding_dim)
        """
        pass

    @abstractmethod
    def get_embedding_dim(self) -> int:
        """
        Return the dimensionality of embeddings.

        Examples: 384 (MiniLM), 768 (BERT), 1536 (OpenAI)
        """
        pass

    @abstractmethod
    def load(self) -> None:
        """
        Load model/resources into memory.

        Called by pipeline before processing stage.
        """
        pass

    @abstractmethod
    def unload(self) -> None:
        """
        Free resources (GPU memory, etc.).

        Called by pipeline after processing stage.
        """
        pass

    @abstractmethod
    def get_memory_usage_mb(self) -> float:
        """
        Report current memory usage in MB.

        Used by pipeline for resource management.
        """
        pass
