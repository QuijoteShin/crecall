# src/vectorizers/sentence_transformer.py
"""Sentence Transformers vectorizer."""

import gc
from typing import List, Optional, Dict, Any
import numpy as np

try:
    import torch
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

from ..contracts.vectorizer import IVectorizer


class SentenceTransformerVectorizer(IVectorizer):
    """
    Vectorizer using Sentence Transformers.

    Default model: sentence-transformers/all-MiniLM-L6-v2
    - Embedding dim: 384
    - VRAM: ~600MB
    - Quality: Good balance of speed and quality
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "sentence-transformers is required. "
                "Install with: pip install sentence-transformers"
            )

        self.config = config or {}
        self.model_name = self.config.get("model", "sentence-transformers/all-MiniLM-L6-v2")
        self.batch_size = self.config.get("batch_size", 64)
        self.device = self.config.get("device", "cuda" if torch.cuda.is_available() else "cpu")

        self.model = None
        self._embedding_dim = None
        self._memory_mb = 0

    def vectorize(self, text: str) -> np.ndarray:
        """Generate embedding vector for single text."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        return self.vectorize_batch([text])[0]

    def vectorize_batch(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for multiple texts efficiently."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Handle empty texts
        processed_texts = [text if text.strip() else " " for text in texts]

        # Encode in batches
        embeddings = self.model.encode(
            processed_texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,  # L2 normalization for cosine similarity
        )

        return embeddings

    def get_embedding_dim(self) -> int:
        """Return the dimensionality of embeddings."""
        if self._embedding_dim is None:
            if self.model is not None:
                self._embedding_dim = self.model.get_sentence_embedding_dimension()
            else:
                # Return known dimension for common models
                if "MiniLM-L6" in self.model_name:
                    return 384
                elif "mpnet" in self.model_name:
                    return 768
                else:
                    return 768  # Default guess

        return self._embedding_dim

    def load(self) -> None:
        """Load model into memory."""
        if self.model is not None:
            return  # Already loaded

        print(f"Loading vectorizer model: {self.model_name} on {self.device}...")

        self.model = SentenceTransformer(self.model_name, device=self.device)
        self._embedding_dim = self.model.get_sentence_embedding_dimension()

        # Estimate memory usage
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
            self._memory_mb = torch.cuda.memory_allocated() / 1024 / 1024
        else:
            # Rough estimate for CPU
            model_size_map = {
                "MiniLM-L6": 600,
                "mpnet": 1100,
                "e5-large": 2500,
            }
            self._memory_mb = 600  # Default
            for key, size in model_size_map.items():
                if key in self.model_name:
                    self._memory_mb = size
                    break

        print(f"✓ Vectorizer loaded (dim={self._embedding_dim}, {self._memory_mb:.0f} MB)")

    def unload(self) -> None:
        """Free resources."""
        if self.model is None:
            return

        print("Unloading vectorizer...")

        # Delete model
        del self.model
        self.model = None

        # Force garbage collection
        gc.collect()

        # Clear CUDA cache if applicable
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        self._memory_mb = 0
        print("✓ Vectorizer unloaded")

    def get_memory_usage_mb(self) -> float:
        """Report current memory usage."""
        if self.model is None:
            return 0.0

        if self.device == "cuda" and torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024 / 1024
        else:
            return self._memory_mb
