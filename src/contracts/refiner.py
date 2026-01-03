# src/contracts/refiner.py
"""Abstract interface for semantic refinement using SLM."""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field


@dataclass
class RefinedOutput:
    """Output from semantic refinement process."""
    refined_content: str  # Dense semantic text optimized for vectorization
    intent: Optional[str] = None  # Primary intent (e.g., "Soporte Técnico")
    entities: List[str] = field(default_factory=list)  # Key entities extracted
    summary: Optional[str] = None  # One-sentence summary
    confidence: float = 1.0  # Confidence score of extraction
    metadata: Dict[str, Any] = field(default_factory=dict)  # Raw response, etc.


class IRefiner(ABC):
    """
    Extracts semantic intent from raw message content using SLM.

    Purpose:
    - Transform "what user wrote" into "what user meant"
    - Produce dense, semantically-rich text for vectorization
    - Extract structured intent/entities for filtering

    Memory Management:
    - Implements load()/unload() pattern for GPU resource control
    - Reports memory usage for MemoryManager coordination
    """

    @abstractmethod
    def refine(self, text: str) -> RefinedOutput:
        """
        Refine single text to extract semantic intent.

        Args:
            text: Raw or normalized message content

        Returns:
            RefinedOutput with refined_content for vectorization
        """
        pass

    @abstractmethod
    def refine_batch(self, texts: List[str]) -> List[RefinedOutput]:
        """
        Batch refinement for efficiency.

        Args:
            texts: List of messages to refine

        Returns:
            List of RefinedOutput objects
        """
        pass

    @abstractmethod
    def load(self) -> None:
        """
        Load model into memory (GPU/CPU).

        Called by MemoryManager before processing stage.
        Should be idempotent (safe to call multiple times).
        """
        pass

    @abstractmethod
    def unload(self) -> None:
        """
        Free resources (GPU memory, model weights).

        Called by MemoryManager when memory is needed.
        Should be idempotent.
        """
        pass

    @abstractmethod
    def get_memory_usage_mb(self) -> float:
        """
        Report current memory usage in MB.

        Used by MemoryManager to make load/unload decisions.

        Returns:
            Current VRAM/RAM usage in megabytes
        """
        pass

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """
        Return model metadata for logging and versioning.

        Returns:
            Dict with keys like: model_name, quantization, context_length
        """
        pass
