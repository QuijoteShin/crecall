# src/contracts/storage.py
"""Abstract interface for data persistence."""

from abc import ABC, abstractmethod
from typing import List, Optional
import numpy as np

from ..models.processed import ProcessedMessage
from ..models.search import SearchResult


class IStorage(ABC):
    """
    Persists processed messages and enables search.

    Implementations can use:
    - SQLite with vector extension
    - PostgreSQL with pgvector
    - Specialized vector DBs (Qdrant, Weaviate, etc.)
    """

    @abstractmethod
    def initialize(self) -> None:
        """
        Create database schema/indexes.

        Called once during system setup.
        """
        pass

    @abstractmethod
    def save_message(self, message: ProcessedMessage) -> str:
        """
        Persist a processed message.

        Args:
            message: ProcessedMessage with all fields populated

        Returns:
            Message ID (generated or from message.normalized.id)
        """
        pass

    @abstractmethod
    def save_messages_batch(self, messages: List[ProcessedMessage]) -> List[str]:
        """
        Bulk insert for efficiency.

        Args:
            messages: List of ProcessedMessage objects

        Returns:
            List of message IDs
        """
        pass

    @abstractmethod
    def search_by_vector(
        self,
        query_vector: np.ndarray,
        limit: int = 10,
        filters: Optional[dict] = None
    ) -> List[SearchResult]:
        """
        Vector similarity search.

        Args:
            query_vector: Embedding vector
            limit: Max results to return
            filters: Optional filters (e.g., {'conversation_id': 'abc123'})

        Returns:
            List of SearchResult objects ordered by similarity
        """
        pass

    @abstractmethod
    def search_by_text(
        self,
        query: str,
        limit: int = 10,
        filters: Optional[dict] = None
    ) -> List[SearchResult]:
        """
        Full-text search (fallback when no vectors).

        Args:
            query: Search query string
            limit: Max results
            filters: Optional filters

        Returns:
            List of SearchResult objects
        """
        pass

    @abstractmethod
    def get_message_by_id(self, message_id: str) -> Optional[ProcessedMessage]:
        """
        Retrieve single message by ID.
        """
        pass

    @abstractmethod
    def get_conversation_messages(self, conversation_id: str) -> List[ProcessedMessage]:
        """
        Get all messages in a conversation, ordered by timestamp.
        """
        pass

    @abstractmethod
    def get_statistics(self) -> dict:
        """
        Return storage statistics.

        Returns:
            Dict with keys: total_messages, total_conversations, size_mb, etc.
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """
        Close connections and cleanup.
        """
        pass
