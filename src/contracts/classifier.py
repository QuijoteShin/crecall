# src/contracts/classifier.py
"""Abstract interface for intent classification."""

from abc import ABC, abstractmethod
from typing import List

from ..models.processed import Classification


class IClassifier(ABC):
    """
    Classifies message intent and extracts topics.

    Intent types:
    - question_explicit: User asking for new information
    - question_followup: Asking details about previous answer
    - command: Request for action/task
    - knowledge_share: User providing information
    - social: Casual conversation
    """

    @abstractmethod
    def classify(self, text: str) -> Classification:
        """
        Analyze text and return classification.

        Args:
            text: Normalized message content

        Returns:
            Classification object with intent, topics, confidence
        """
        pass

    @abstractmethod
    def classify_batch(self, texts: List[str]) -> List[Classification]:
        """
        Batch classification for efficiency.

        Args:
            texts: List of normalized message contents

        Returns:
            List of Classification objects
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
