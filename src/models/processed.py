# src/models/processed.py
"""Processed message models after pipeline stages."""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import numpy as np

from .normalized import NormalizedMessage


@dataclass
class Classification:
    """Message intent classification result."""

    intent: str  # 'question_explicit', 'question_followup', 'command', 'knowledge_share', 'social'
    confidence: float  # 0.0 to 1.0
    topics: List[str] = field(default_factory=list)  # Extracted topics/entities
    is_question: bool = False
    is_command: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Set convenience flags."""
        self.is_question = self.intent.startswith('question_')
        self.is_command = self.intent == 'command'


@dataclass
class ProcessedMessage:
    """
    Fully processed message after all pipeline stages.

    This is what gets stored in the database.
    """

    normalized: NormalizedMessage
    clean_content: str  # After normalization (formatted Markdown for UI)
    vectorizable_content: str  # After soft sanitization (for embeddings)
    classification: Classification
    vector: Optional[np.ndarray] = None

    def to_dict(self, include_vector: bool = False) -> dict:
        """
        Convert to dictionary.

        Args:
            include_vector: If True, include vector as list
        """
        result = {
            'normalized': self.normalized.to_dict(),
            'clean_content': self.clean_content,
            'vectorizable_content': self.vectorizable_content,
            'classification': {
                'intent': self.classification.intent,
                'confidence': self.classification.confidence,
                'topics': self.classification.topics,
                'is_question': self.classification.is_question,
                'is_command': self.classification.is_command,
                'metadata': self.classification.metadata,
            },
        }

        if include_vector and self.vector is not None:
            result['vector'] = self.vector.tolist()

        return result

    @classmethod
    def from_dict(cls, data: dict) -> 'ProcessedMessage':
        """Create from dictionary."""
        normalized = NormalizedMessage.from_dict(data['normalized'])

        classification = Classification(
            intent=data['classification']['intent'],
            confidence=data['classification']['confidence'],
            topics=data['classification']['topics'],
            metadata=data['classification'].get('metadata', {}),
        )

        vector = None
        if 'vector' in data:
            vector = np.array(data['vector'])

        return cls(
            normalized=normalized,
            clean_content=data['clean_content'],
            vectorizable_content=data['vectorizable_content'],
            classification=classification,
            vector=vector,
        )
