# src/contracts/importer.py
"""Abstract interface for data importers."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator

from ..models.normalized import NormalizedMessage


class IImporter(ABC):
    """
    Converts source-specific format to normalized message stream.

    Each importer handles one or more format types (e.g., ChatGPT, Claude).
    """

    @abstractmethod
    def supports_format(self, format_id: str) -> bool:
        """
        Check if this importer can handle the given format.

        Args:
            format_id: Format identifier from IFormatDetector

        Returns:
            True if this importer supports the format
        """
        pass

    @abstractmethod
    def import_data(self, source: Path) -> Iterator[NormalizedMessage]:
        """
        Import data from source and yield normalized messages.

        Args:
            source: Path to source file/directory/ZIP

        Yields:
            NormalizedMessage instances

        Raises:
            ValueError: If source format is incompatible
            FileNotFoundError: If source doesn't exist
        """
        pass

    @abstractmethod
    def get_metadata(self, source: Path) -> dict:
        """
        Extract metadata from source without full import.

        Returns:
            Dict with keys like 'total_conversations', 'date_range', 'user_info'
        """
        pass
