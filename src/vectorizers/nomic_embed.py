# src/vectorizers/nomic_embed.py
"""Nomic Embed v1.5 vectorizer con soporte Matryoshka nativo."""

import gc
from typing import List, Optional, Dict, Any
import numpy as np

try:
    import torch
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    torch = None

from ..contracts.vectorizer import IVectorizer


class NomicEmbedVectorizer(IVectorizer):
    """
    Nomic Embed v1.5 vectorizer.

    Ventajas:
    - Soporte nativo Matryoshka (MRL) - truncado configurable sin degradación
    - 8192 tokens de contexto
    - Prefijos search_query/search_document para mejor precisión

    Dimensiones válidas (MRL): 64, 128, 256, 384, 512, 768
    Recomendado: 256 (103% precisión vs 768, 67% menos storage)
    """

    QUERY_PREFIX = "search_query: "
    DOCUMENT_PREFIX = "search_document: "
    VALID_DIMENSIONS = [64, 128, 256, 384, 512, 768]
    MODEL_FULL_DIM = 768

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "sentence-transformers requerido. "
                "Instalar con: pip install sentence-transformers einops"
            )

        self.config = config or {}
        self.model_name = self.config.get("model", "nomic-ai/nomic-embed-text-v1.5")
        self.device = self.config.get("device", "cuda" if torch and torch.cuda.is_available() else "cpu")
        self.output_dim = self.config.get("dimension", 256)
        self.batch_size = self.config.get("batch_size", 32)

        # Validar dimensión
        if self.output_dim not in self.VALID_DIMENSIONS:
            raise ValueError(
                f"Dimensión {self.output_dim} no válida. "
                f"Dimensiones MRL válidas: {self.VALID_DIMENSIONS}"
            )

        self.model = None
        self._full_embedding_dim = None
        self._memory_mb = 0.0

    def vectorize(self, text: str) -> np.ndarray:
        """Vectoriza texto individual (asume documento por defecto)."""
        if not text or not text.strip():
            return np.zeros(self.output_dim, dtype=np.float32)

        self.load()
        input_text = self.DOCUMENT_PREFIX + text

        embedding = self.model.encode(
            input_text,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        return self._truncate_and_normalize(embedding)

    def vectorize_query(self, text: str) -> np.ndarray:
        """Vectoriza query de búsqueda (usa prefijo search_query)."""
        if not text or not text.strip():
            return np.zeros(self.output_dim, dtype=np.float32)

        self.load()
        input_text = self.QUERY_PREFIX + text

        embedding = self.model.encode(
            input_text,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        return self._truncate_and_normalize(embedding)

    def vectorize_batch(self, texts: List[str]) -> np.ndarray:
        """Vectoriza batch de documentos."""
        if not texts:
            return np.array([])

        self.load()

        processed_texts = [
            self.DOCUMENT_PREFIX + (text if text and text.strip() else " ")
            for text in texts
        ]

        embeddings = self.model.encode(
            processed_texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        truncated = embeddings[:, :self.output_dim]
        norms = np.linalg.norm(truncated, axis=1, keepdims=True)
        return truncated / (norms + 1e-8)

    def _truncate_and_normalize(self, embedding: np.ndarray) -> np.ndarray:
        """Trunca a output_dim y re-normaliza (Matryoshka)."""
        truncated = embedding[:self.output_dim]
        norm = np.linalg.norm(truncated)
        if norm > 0:
            truncated = truncated / norm
        return truncated.astype(np.float32)

    def get_embedding_dim(self) -> int:
        """Dimensión del embedding (truncado MRL)."""
        return self.output_dim

    def load(self) -> None:
        """Carga modelo en memoria."""
        if self.model is not None:
            return

        print(f"Cargando {self.model_name} (MRL={self.output_dim})...")

        self.model = SentenceTransformer(
            self.model_name,
            device=self.device,
            trust_remote_code=True
        )

        self._full_embedding_dim = self.model.get_sentence_embedding_dimension()

        if self.device == "cuda" and torch and torch.cuda.is_available():
            torch.cuda.synchronize()
            self._memory_mb = torch.cuda.memory_allocated() / 1024 / 1024
        else:
            self._memory_mb = 550.0

        print(f"✓ Nomic Embed cargado (full={self._full_embedding_dim}, truncado={self.output_dim}, {self._memory_mb:.0f} MB)")

    def unload(self) -> None:
        """Libera recursos."""
        if self.model is None:
            return

        print("Descargando Nomic Embed...")

        del self.model
        self.model = None

        gc.collect()

        if self.device == "cuda" and torch and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        self._memory_mb = 0.0
        print("✓ Nomic Embed descargado")

    def get_memory_usage_mb(self) -> float:
        """Uso de memoria actual en MB."""
        if self.model is None:
            return 0.0

        if self.device == "cuda" and torch and torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024 / 1024

        return self._memory_mb
