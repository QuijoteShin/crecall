# src/classifiers/rule_based.py
"""Rule-based intent classifier (no GPU required)."""

import re
from typing import List, Dict, Any, Optional
from collections import Counter

from ..contracts.classifier import IClassifier
from ..models.processed import Classification


class RuleBasedClassifier(IClassifier):
    """
    Simple rule-based classifier using regex patterns.

    No GPU/AI required - uses heuristics:
    - "?" at end = question
    - Imperative verbs = command
    - Declarative statements = knowledge_share
    - Short responses = social

    Fast, no memory overhead, good enough for most use cases.
    """

    # Question patterns
    QUESTION_PATTERNS = [
        r'\?$',  # Ends with ?
        r'^(what|where|when|why|who|how|which|can|could|would|should|is|are|do|does)',
        r'(please|could you|can you|would you) (help|tell|explain|show)',
    ]

    # Command patterns
    COMMAND_PATTERNS = [
        r'^(create|make|build|generate|write|implement|add|fix|update|change|modify)',
        r'^(show|display|list|find|search|get|fetch|retrieve)',
        r'^(delete|remove|clear|reset|clean)',
        r'(please|kindly) (create|make|build|generate|write)',
    ]

    # Social/casual patterns
    SOCIAL_PATTERNS = [
        r'^(hi|hello|hey|thanks|thank you|ok|okay|yes|no|sure|great|cool)',
        r'^(lol|haha|nice|awesome|perfect|excellent)',
    ]

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

    def classify(self, text: str) -> Classification:
        """Classify single text."""
        results = self.classify_batch([text])
        return results[0]

    def classify_batch(self, texts: List[str]) -> List[Classification]:
        """Batch classification."""
        return [self._classify_single(text) for text in texts]

    def _classify_single(self, text: str) -> Classification:
        """Internal classification logic."""
        if not text or not text.strip():
            return Classification(
                intent="social",
                confidence=0.5,
                topics=[],
            )

        text_lower = text.lower().strip()

        # Check question patterns
        for pattern in self.QUESTION_PATTERNS:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return Classification(
                    intent="question_explicit",
                    confidence=0.85,
                    topics=self._extract_topics(text),
                    metadata={'rule': 'question_pattern'}
                )

        # Check command patterns
        for pattern in self.COMMAND_PATTERNS:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return Classification(
                    intent="command",
                    confidence=0.80,
                    topics=self._extract_topics(text),
                    metadata={'rule': 'command_pattern'}
                )

        # Check social patterns
        for pattern in self.SOCIAL_PATTERNS:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return Classification(
                    intent="social",
                    confidence=0.75,
                    topics=[],
                    metadata={'rule': 'social_pattern'}
                )

        # Short text = likely social
        if len(text.split()) < 5:
            return Classification(
                intent="social",
                confidence=0.60,
                topics=[],
                metadata={'rule': 'short_text'}
            )

        # Default: knowledge_share (statements, explanations)
        return Classification(
            intent="knowledge_share",
            confidence=0.70,
            topics=self._extract_topics(text),
            metadata={'rule': 'default'}
        )

    def _extract_topics(self, text: str, max_topics: int = 5) -> List[str]:
        """
        Extrae tópicos preservando acrónimos (SQL, API, AWS).
        """
        # Extraer palabras preservando case original
        words_raw = re.findall(r'\b\w+\b', text)

        # Stopwords expandido
        stopwords = {
            'this', 'that', 'these', 'those', 'what', 'which', 'who',
            'have', 'been', 'were', 'said', 'from', 'they', 'with',
            'the', 'and', 'for', 'not', 'but', 'you', 'all', 'can',
            'are', 'was', 'will', 'would', 'could', 'should', 'may',
            'es', 'de', 'la', 'el', 'en', 'que', 'los', 'las', 'una', 'del'
        }

        # Filtrar: preserva acrónimos (mayúsculas completas) o palabras > 2 chars
        words = [
            w.lower() for w in words_raw
            if w.lower() not in stopwords and (len(w) > 2 or w.isupper())
        ]

        # Count frequency
        word_counts = Counter(words)

        # Get top N
        topics = [word for word, count in word_counts.most_common(max_topics)]

        return topics

    def load(self) -> None:
        """No model to load - rule-based."""
        pass

    def unload(self) -> None:
        """Nothing to unload."""
        pass

    def get_memory_usage_mb(self) -> float:
        """Rule-based has negligible memory usage."""
        return 0.0
