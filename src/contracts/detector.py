# src/contracts/detector.py
"""Abstract interface for format detection."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional


class IFormatDetector(ABC):
    """
    Detects the format of an input source (ZIP, directory, file).

    Returns a format identifier string (e.g., 'chatgpt', 'claude', 'whatsapp')
    or None if format cannot be determined.
    """

    @abstractmethod
    def detect(self, source: Path) -> Optional[str]:
        """
        Analyze the source and return format identifier.

        Args:
            source: Path to ZIP file, directory, or file

        Returns:
            Format identifier string or None

        Examples:
            'chatgpt' - OpenAI ChatGPT export
            'claude' - Anthropic Claude export
            'whatsapp' - WhatsApp chat export
            'generic' - Unknown format, use generic importer
        """
        pass

    @abstractmethod
    def get_confidence(self, source: Path, format_id: str) -> float:
        """
        Return confidence score (0.0 to 1.0) for a format guess.

        Useful when multiple detectors claim the same source.
        """
        pass
