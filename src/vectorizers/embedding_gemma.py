# src/vectorizers/embedding_gemma.py
"""Google EmbeddingGemma-300m vectorizer (lightweight, on-device)."""

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


class EmbeddingGemmaVectorizer(IVectorizer):
    """
    Google EmbeddingGemma-300m vectorizer.

    Advantages:
    - Only ~300MB RAM/VRAM (vs 600MB for MiniLM)
    - Designed for on-device inference
    - Supports Matryoshka Representation Learning (truncate to 256 dims)
    - Faster inference on CPU

    Configuration:
    - embedding_dim: 256 (truncated) or 768 (full)
    - quantization: 'int8', 'fp16', or 'fp32'
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "sentence-transformers is required. "
                "Install with: pip install sentence-transformers"
            )

        self.config = config or {}
        self.model_name = self.config.get("model", "google/embeddinggemma-300m")
        self.target_dim = self.config.get("dimension", 512)
        self.batch_size = self.config.get("batch_size", 64)
        self.device = self.config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.quantization = self.config.get("quantization", "fp16")  # int8, fp16, fp32

        self.model = None
        self._full_embedding_dim = None
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

        # Encode
        embeddings = self.model.encode(
            processed_texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,  # L2 normalization
        )

        # Matryoshka truncation (keep first N dimensions - most important)
        if self.target_dim < embeddings.shape[1]:
            embeddings = embeddings[:, :self.target_dim]

            # Re-normalize after truncation
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / (norms + 1e-8)

        return embeddings

    def get_embedding_dim(self) -> int:
        """Return the dimensionality of embeddings."""
        return self.target_dim

    def load(self) -> None:
        """Load model into memory."""
        if self.model is not None:
            return  # Already loaded

        print(f"Loading EmbeddingGemma: {self.model_name} on {self.device}...")
        print(f"  Target dimension: {self.target_dim} (Matryoshka truncation)")

        # Load with trust_remote_code for Gemma models
        self.model = SentenceTransformer(
            self.model_name,
            device=self.device,
            trust_remote_code=True
        )

        # Get full dimension before truncation
        self._full_embedding_dim = self.model.get_sentence_embedding_dimension()

        # Apply quantization if requested
        if self.quantization == "int8" and self.device == "cuda":
            try:
                # INT8 quantization for CUDA
                self.model = self.model.half()  # FP16 first
                print("  Applied FP16 quantization")
            except Exception as e:
                print(f"  Could not apply quantization: {e}")

        # Estimate memory usage
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
            self._memory_mb = torch.cuda.memory_allocated() / 1024 / 1024
        else:
            # Rough estimate for CPU
            if self.quantization == "int8":
                self._memory_mb = 300
            elif self.quantization == "fp16":
                self._memory_mb = 350
            else:
                self._memory_mb = 600

        print(f"✓ EmbeddingGemma loaded (full_dim={self._full_embedding_dim}, truncated={self.target_dim}, {self._memory_mb:.0f} MB)")

    def unload(self) -> None:
        """Free resources."""
        if self.model is None:
            return

        print("Unloading EmbeddingGemma...")

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
        print("✓ EmbeddingGemma unloaded")

    def get_memory_usage_mb(self) -> float:
        """Report current memory usage."""
        if self.model is None:
            return 0.0

        if self.device == "cuda" and torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024 / 1024
        else:
            return self._memory_mb
