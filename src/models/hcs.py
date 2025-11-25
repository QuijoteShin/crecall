# src/models/hcs.py
"""Hierarchical Context System (HCS) models."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any
import numpy as np


@dataclass
class TopicCacheEntry:
    """Entry in the Topic Cache (last N active topics)."""

    topic: str
    vector: np.ndarray
    last_seen: datetime
    relevance_score: float = 1.0  # Decays over time
    message_count: int = 1  # How many messages in this topic


@dataclass
class MicroNode:
    """
    Micro Node (Child Node) - Individual message within a macro context.

    Represents a single turn in the conversation.
    """

    id: str
    message_id: str  # Reference to ProcessedMessage
    parent_macro_id: str
    role: str  # 'user', 'assistant'
    content_preview: str  # First 200 chars
    timestamp: datetime
    is_question: bool = False


@dataclass
class MacroNode:
    """
    Macro Node (Root Context) - A topic-driven conversation thread.

    Created when:
    - User asks Explicit Question AND
    - Topic is NOT a continuation of recent context
    """

    # Campos requeridos (sin default) primero
    id: str
    main_topic: str
    summary: str
    timestamp_start: datetime

    # Campos con default
    type: str = "MACRO"
    timestamp_end: Optional[datetime] = None
    status: str = "active"
    vector: Optional[np.ndarray] = None
    children: List[MicroNode] = field(default_factory=list)
    total_messages: int = 0
    topics: List[str] = field(default_factory=list)
    entities: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self, include_vector: bool = False) -> dict:
        """Serialize to dict."""
        result = {
            'id': self.id,
            'type': self.type,
            'main_topic': self.main_topic,
            'summary': self.summary,
            'timestamp_start': self.timestamp_start.isoformat(),
            'timestamp_end': self.timestamp_end.isoformat() if self.timestamp_end else None,
            'status': self.status,
            'children': [
                {
                    'id': child.id,
                    'message_id': child.message_id,
                    'parent_macro_id': child.parent_macro_id,
                    'role': child.role,
                    'content_preview': child.content_preview,
                    'timestamp': child.timestamp.isoformat(),
                    'is_question': child.is_question,
                }
                for child in self.children
            ],
            'total_messages': self.total_messages,
            'topics': self.topics,
            'entities': self.entities,
            'metadata': self.metadata,
        }
        if include_vector and self.vector is not None:
            result['vector'] = self.vector.tolist()
        return result


@dataclass
class UserTimeline:
    """
    User's complete timeline (graph of macro contexts).

    This is the "memory" of the HCS system.
    """

    user_id: str

    # Active context
    active_macro_context_id: Optional[str] = None

    # Topic cache (last N topics for quick anchoring)
    topic_cache: List[TopicCacheEntry] = field(default_factory=list)
    topic_cache_size: int = 5

    # Graph of macro nodes
    nodes: List[MacroNode] = field(default_factory=list)

    def add_topic_to_cache(self, topic: str, vector: np.ndarray):
        """Add or update topic in cache."""
        # Check if topic already exists
        for entry in self.topic_cache:
            if entry.topic == topic:
                entry.last_seen = datetime.now()
                entry.relevance_score = 1.0
                entry.message_count += 1
                return

        # Add new topic
        entry = TopicCacheEntry(
            topic=topic,
            vector=vector,
            last_seen=datetime.now(),
        )

        self.topic_cache.append(entry)

        # Maintain cache size (remove oldest)
        if len(self.topic_cache) > self.topic_cache_size:
            # Sort by relevance and last_seen
            self.topic_cache.sort(
                key=lambda x: (x.relevance_score, x.last_seen),
                reverse=True
            )
            self.topic_cache = self.topic_cache[:self.topic_cache_size]

    def find_similar_topic(
        self,
        query_vector: np.ndarray,
        threshold: float = 0.75
    ) -> Optional[TopicCacheEntry]:
        """
        Find similar topic in cache using vector similarity.

        Returns:
            TopicCacheEntry if similarity > threshold, else None
        """
        best_match = None
        best_score = threshold

        for entry in self.topic_cache:
            similarity = float(np.dot(query_vector, entry.vector))

            if similarity > best_score:
                best_score = similarity
                best_match = entry

        return best_match

    def get_active_macro_node(self) -> Optional[MacroNode]:
        """Get currently active macro context."""
        if not self.active_macro_context_id:
            return None

        for node in self.nodes:
            if node.id == self.active_macro_context_id:
                return node

        return None

    def create_macro_node(
        self,
        topic: str,
        summary: str,
        message_id: str
    ) -> MacroNode:
        """Create new macro context node."""
        import uuid

        node = MacroNode(
            id=f"evt_{uuid.uuid4().hex[:8]}",
            main_topic=topic,
            summary=summary,
            timestamp_start=datetime.now(),
        )

        self.nodes.append(node)
        self.active_macro_context_id = node.id

        return node

    def add_child_to_macro(
        self,
        macro_id: str,
        message_id: str,
        role: str,
        content_preview: str,
        is_question: bool = False
    ) -> MicroNode:
        """Add child node to existing macro."""
        import uuid

        micro = MicroNode(
            id=f"msg_{uuid.uuid4().hex[:8]}",
            message_id=message_id,
            parent_macro_id=macro_id,
            role=role,
            content_preview=content_preview,
            timestamp=datetime.now(),
            is_question=is_question,
        )

        # Find macro and add child
        for node in self.nodes:
            if node.id == macro_id:
                node.children.append(micro)
                node.total_messages += 1
                node.timestamp_end = datetime.now()
                break

        return micro

    def to_dict(self) -> dict:
        """Serialize timeline."""
        return {
            'user_id': self.user_id,
            'active_macro_context_id': self.active_macro_context_id,
            'topic_cache': [
                {
                    'topic': entry.topic,
                    'vector': entry.vector.tolist(),
                    'last_seen': entry.last_seen.isoformat(),
                    'relevance_score': entry.relevance_score,
                    'message_count': entry.message_count,
                }
                for entry in self.topic_cache
            ],
            'nodes': [node.to_dict() for node in self.nodes],
        }
