# src/classifiers/transformer.py
"""Transformer-based intent classifier."""

import gc
import re
from typing import List, Optional, Dict, Any

try:
    import torch
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from ..contracts.classifier import IClassifier
from ..models.processed import Classification


class TransformerClassifier(IClassifier):
    """
    Zero-shot intent classifier using transformers.

    Default model: valhalla/distilbart-mnli-12-1
    Memory-efficient with explicit load/unload for pipeline processing.
    """

    # Intent labels for zero-shot classification
    INTENT_LABELS = [
        "asking a question",
        "following up on previous answer",
        "requesting an action or task",
        "sharing knowledge or information",
        "casual conversation",
    ]

    # Mapping to standardized intent names
    INTENT_MAPPING = {
        "asking a question": "question_explicit",
        "following up on previous answer": "question_followup",
        "requesting an action or task": "command",
        "sharing knowledge or information": "knowledge_share",
        "casual conversation": "social",
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers and torch are required for TransformerClassifier. "
                "Install with: pip install transformers torch"
            )

        self.config = config or {}
        self.model_name = self.config.get("model", "valhalla/distilbart-mnli-12-1")
        self.batch_size = self.config.get("batch_size", 32)
        self.device = self.config.get("device", "cuda" if torch.cuda.is_available() else "cpu")

        self.pipeline = None
        self._memory_mb = 0

    def classify(self, text: str) -> Classification:
        """Classify single text."""
        if self.pipeline is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        results = self.classify_batch([text])
        return results[0]

    def classify_batch(self, texts: List[str]) -> List[Classification]:
        """Batch classification for efficiency."""
        if self.pipeline is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        classifications = []

        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_results = self._classify_batch_internal(batch)
            classifications.extend(batch_results)

        return classifications

    def _classify_batch_internal(self, texts: List[str]) -> List[Classification]:
        """Internal batch processing."""
        results = []

        for text in texts:
            # Skip empty texts
            if not text.strip():
                results.append(Classification(
                    intent="social",
                    confidence=0.5,
                    topics=[],
                ))
                continue

            # Truncate very long texts (model limit is typically 512 tokens)
            truncated_text = text[:2000]

            # Zero-shot classification
            prediction = self.pipeline(
                truncated_text,
                candidate_labels=self.INTENT_LABELS,
                hypothesis_template="This text is {}."
            )

            # Get top prediction
            top_label = prediction['labels'][0]
            top_score = prediction['scores'][0]

            # Map to standardized intent
            intent = self.INTENT_MAPPING.get(top_label, "social")

            # Extract topics (simple keyword extraction)
            topics = self._extract_topics(text)

            classification = Classification(
                intent=intent,
                confidence=float(top_score),
                topics=topics,
                metadata={
                    'raw_prediction': {
                        'labels': prediction['labels'][:3],
                        'scores': [float(s) for s in prediction['scores'][:3]],
                    }
                }
            )

            results.append(classification)

        return results

    def _extract_topics(self, text: str, max_topics: int = 5) -> List[str]:
        """
        Simple topic extraction using frequency analysis.

        For production, consider using NER or keyphrase extraction.
        """
        # Remove punctuation and lowercase
        words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())

        # Filter common stopwords (minimal set - ADR compliant)
        # We only remove the most meaningless words
        ultra_stopwords = {'this', 'that', 'these', 'those', 'what', 'which', 'who'}
        words = [w for w in words if w not in ultra_stopwords]

        # Count frequency
        from collections import Counter
        word_counts = Counter(words)

        # Get top N
        topics = [word for word, count in word_counts.most_common(max_topics)]

        return topics

    def load(self) -> None:
        """Load model into memory."""
        if self.pipeline is not None:
            return  # Already loaded

        print(f"Loading classifier model: {self.model_name} on {self.device}...")

        self.pipeline = pipeline(
            "zero-shot-classification",
            model=self.model_name,
            device=0 if self.device == "cuda" else -1
        )

        # Estimate memory usage
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
            self._memory_mb = torch.cuda.memory_allocated() / 1024 / 1024
        else:
            # Rough estimate for CPU
            self._memory_mb = 1200  # distilbart is ~1.2GB

        print(f"✓ Classifier loaded ({self._memory_mb:.0f} MB)")

    def unload(self) -> None:
        """Free resources."""
        if self.pipeline is None:
            return

        print("Unloading classifier...")

        # Delete pipeline
        del self.pipeline
        self.pipeline = None

        # Force garbage collection
        gc.collect()

        # Clear CUDA cache if applicable
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        self._memory_mb = 0
        print("✓ Classifier unloaded")

    def get_memory_usage_mb(self) -> float:
        """Report current memory usage."""
        if self.pipeline is None:
            return 0.0

        if self.device == "cuda" and torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024 / 1024
        else:
            return self._memory_mb
