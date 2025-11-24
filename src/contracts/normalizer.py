# src/contracts/normalizer.py
"""Abstract interface for content normalization."""

from abc import ABC, abstractmethod
from typing import Optional


class INormalizer(ABC):
    """
    Normalizes content to a clean, processable format.

    Responsibilities:
    - Remove invisible characters (zero-width, RTL marks, etc.)
    - Convert HTML/rich text to Markdown or plain text
    - Remove unnecessary whitespace
    - Sanitize against prompt injection patterns
    """

    @abstractmethod
    def normalize(self, content: str, content_type: str = "text") -> str:
        """
        Clean and normalize content.

        Args:
            content: Raw content string
            content_type: Type hint ('text', 'html', 'code', 'markdown')

        Returns:
            Normalized content string
        """
        pass

    @abstractmethod
    def remove_invisible_chars(self, text: str) -> str:
        """
        Remove zero-width spaces, RTL marks, and other invisible characters.

        Critical for security (prevents prompt injection via hidden characters).
        """
        pass

    @abstractmethod
    def convert_to_markdown(self, content: str, source_format: str) -> str:
        """
        Convert content to clean Markdown.

        Args:
            content: Raw content
            source_format: 'html', 'docx', 'pdf', etc.

        Returns:
            Markdown string
        """
        pass

    @abstractmethod
    def extract_code_blocks(self, content: str) -> list[dict]:
        """
        Extract code blocks with language tags.

        Returns:
            List of dicts: [{'language': 'python', 'code': '...', 'line': 10}]
        """
        pass
