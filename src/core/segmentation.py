# src/core/segmentation.py
"""Semantic Segmentation - HCS (Hierarchical Context System) Implementation."""

from typing import List, Dict, Any, Optional
import numpy as np
from collections import Counter

from ..models.processed import ProcessedMessage


class TopicSegment:
    """Represents a semantic segment within a conversation."""

    def __init__(self, segment_id: str, sequence: int):
        self.segment_id = segment_id
        self.sequence = sequence
        self.messages: List[ProcessedMessage] = []
        self.topic_label: Optional[str] = None
        self.avg_vector: Optional[np.ndarray] = None

    def add_message(self, message: ProcessedMessage) -> None:
        """Add a message to this segment."""
        self.messages.append(message)

    def update_avg_vector(self) -> None:
        """Update the average vector for this segment."""
        vectors = [msg.vector for msg in self.messages if msg.vector is not None]
        if vectors:
            self.avg_vector = np.mean(vectors, axis=0)
            # Normalize for cosine similarity
            norm = np.linalg.norm(self.avg_vector)
            if norm > 0:
                self.avg_vector = self.avg_vector / norm

    def generate_topic_label(self) -> str:
        """
        Generate a topic label for this segment based on message content.

        Uses frequency analysis of topics and keywords.
        """
        # Collect all topics from classifications
        all_topics = []
        for msg in self.messages:
            all_topics.extend(msg.classification.topics)

        if not all_topics:
            # Fallback: use first few words from first message
            if self.messages:
                words = self.messages[0].vectorizable_content.split()[:3]
                return " ".join(words).title()
            return "Untitled Segment"

        # Get most common topics
        topic_counts = Counter(all_topics)
        top_topics = [topic for topic, count in topic_counts.most_common(3)]

        # Generate label from top topics
        if len(top_topics) == 1:
            label = top_topics[0].title()
        elif len(top_topics) == 2:
            label = f"{top_topics[0].title()} & {top_topics[1].title()}"
        else:
            label = f"{top_topics[0].title()} / {top_topics[1].title()}"

        return label


class SemanticSegmenter:
    """
    Implements Topic Drift Detection using cosine similarity.

    Based on HCS (Hierarchical Context System) specification.
    """

    def __init__(self, similarity_threshold: float = 0.5, min_messages_per_segment: int = 1):
        """
        Initialize segmenter.

        Args:
            similarity_threshold: Cosine similarity threshold for topic drift (0.0-1.0)
                                  Values below this trigger a new segment.
            min_messages_per_segment: Minimum messages required to form a segment
        """
        self.similarity_threshold = similarity_threshold
        self.min_messages_per_segment = min_messages_per_segment

    def segment_conversation(
        self,
        conversation_id: str,
        messages: List[ProcessedMessage]
    ) -> List[TopicSegment]:
        """
        Segment a conversation based on topic drift.

        Algorithm (HCS 3.2):
        1. Iterate chronologically through messages
        2. Maintain running average vector for current segment
        3. Calculate similarity between new message and segment average
        4. If similarity < threshold, start new segment (topic drift detected)

        Args:
            conversation_id: ID of the conversation
            messages: Chronologically ordered messages

        Returns:
            List of TopicSegment objects
        """
        if not messages:
            return []

        segments: List[TopicSegment] = []
        current_segment_seq = 1

        # Initialize first segment
        current_segment = TopicSegment(
            segment_id=f"{conversation_id}_seg{current_segment_seq}",
            sequence=current_segment_seq
        )

        for i, message in enumerate(messages):
            # Skip messages without vectors
            if message.vector is None:
                continue

            # First message always goes to first segment
            if i == 0:
                current_segment.add_message(message)
                current_segment.update_avg_vector()
                continue

            # Calculate similarity with current segment
            if current_segment.avg_vector is not None:
                similarity = self._cosine_similarity(message.vector, current_segment.avg_vector)

                # Check for topic drift
                if similarity < self.similarity_threshold:
                    # Close current segment
                    current_segment.generate_topic_label()
                    segments.append(current_segment)

                    # Start new segment
                    current_segment_seq += 1
                    current_segment = TopicSegment(
                        segment_id=f"{conversation_id}_seg{current_segment_seq}",
                        sequence=current_segment_seq
                    )

            # Add message to current segment
            current_segment.add_message(message)
            current_segment.update_avg_vector()

        # Don't forget the last segment
        if current_segment.messages:
            current_segment.generate_topic_label()
            segments.append(current_segment)

        # Generate topic labels for all segments
        for segment in segments:
            if segment.topic_label is None:
                segment.topic_label = segment.generate_topic_label()

        return segments

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors.

        Assumes vectors are already normalized (which they should be from vectorizer).
        """
        # Dot product (vectors should already be normalized)
        similarity = float(np.dot(vec1, vec2))

        # Ensure in range [-1, 1] (account for floating point errors)
        return max(-1.0, min(1.0, similarity))

    def get_segment_updates(self, segments: List[TopicSegment]) -> List[Dict[str, Any]]:
        """
        Convert segments to database update format.

        Returns:
            List of dicts with: message_id, segment_id, segment_topic, segment_sequence
        """
        updates = []

        for segment in segments:
            for message in segment.messages:
                updates.append({
                    'message_id': message.normalized.id,
                    'segment_id': segment.segment_id,
                    'segment_topic': segment.topic_label or "Untitled",
                    'segment_sequence': segment.sequence
                })

        return updates
