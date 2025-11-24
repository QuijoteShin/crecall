# src/core/hcs_engine.py
"""Hierarchical Context System (HCS) Engine."""

from typing import Optional, List, Dict, Any
from datetime import datetime
import numpy as np

from ..models.hcs import UserTimeline, MacroNode, MicroNode, TopicCacheEntry
from ..models.processed import ProcessedMessage


class HCSEngine:
    """
    Hierarchical Context System Engine.

    Implements the 3-layer architecture:
    1. Admission Layer: Intent classification (already done by IClassifier)
    2. Contextualization Layer: Topic anchoring (similarity check)
    3. State Layer: Macro/Micro node management

    Based on the HCS specification from CONCEPT.md
    """

    SIMILARITY_THRESHOLD = 0.75  # Threshold for topic continuation

    def __init__(self, user_id: str = "default"):
        self.user_id = user_id
        self.timeline = UserTimeline(user_id=user_id)

    def process_message(
        self,
        message: ProcessedMessage,
        message_vector: np.ndarray
    ) -> Dict[str, Any]:
        """
        Process a message through the HCS pipeline.

        Returns:
            Dict with decision: {
                'decision': 'new_macro' | 'continuation',
                'macro_id': str,
                'micro_id': str,
                'similarity': float
            }
        """
        classification = message.classification

        # Extract main topic (first topic from classification)
        topic = classification.topics[0] if classification.topics else "General"

        # Layer 2: Context Anchoring (check if continuation)
        similar_topic = self.timeline.find_similar_topic(
            message_vector,
            threshold=self.SIMILARITY_THRESHOLD
        )

        # Layer 3: State Management (Macro/Micro decision)

        if classification.intent == "question_explicit" and similar_topic is None:
            # NEW MACRO NODE
            return self._create_new_macro(message, topic, message_vector)

        elif similar_topic is not None:
            # CONTINUATION - Add to existing macro
            return self._add_to_existing_macro(message, similar_topic, message_vector)

        else:
            # Default: Enrich current macro (social, knowledge_share, etc.)
            return self._enrich_current_macro(message, topic, message_vector)

    def _create_new_macro(
        self,
        message: ProcessedMessage,
        topic: str,
        message_vector: np.ndarray
    ) -> Dict:
        """Create new macro context node."""
        # Generate summary (simple for now, could use LLM later)
        summary = f"Conversación sobre {topic}"

        macro = self.timeline.create_macro_node(
            topic=topic,
            summary=summary,
            message_id=message.normalized.id
        )

        # Add first message as child
        micro = self.timeline.add_child_to_macro(
            macro_id=macro.id,
            message_id=message.normalized.id,
            role=message.normalized.author_role,
            content_preview=message.clean_content[:200],
            is_question=message.classification.is_question
        )

        # Update topic cache
        self.timeline.add_topic_to_cache(topic, message_vector)

        return {
            'decision': 'new_macro',
            'macro_id': macro.id,
            'micro_id': micro.id,
            'topic': topic,
            'similarity': 0.0,
        }

    def _add_to_existing_macro(
        self,
        message: ProcessedMessage,
        similar_topic: TopicCacheEntry,
        message_vector: np.ndarray
    ) -> Dict:
        """Add message to existing macro context."""
        # Get active macro
        active_macro = self.timeline.get_active_macro_node()

        if active_macro is None:
            # Fallback: create new macro
            return self._create_new_macro(message, similar_topic.topic, message_vector)

        # Add as child to active macro
        micro = self.timeline.add_child_to_macro(
            macro_id=active_macro.id,
            message_id=message.normalized.id,
            role=message.normalized.author_role,
            content_preview=message.clean_content[:200],
            is_question=message.classification.is_question
        )

        # Update topic cache
        self.timeline.add_topic_to_cache(similar_topic.topic, message_vector)

        return {
            'decision': 'continuation',
            'macro_id': active_macro.id,
            'micro_id': micro.id,
            'topic': similar_topic.topic,
            'similarity': float(np.dot(message_vector, similar_topic.vector)),
        }

    def _enrich_current_macro(
        self,
        message: ProcessedMessage,
        topic: str,
        message_vector: np.ndarray
    ) -> Dict:
        """Enrich current macro with non-question content."""
        active_macro = self.timeline.get_active_macro_node()

        if active_macro is None:
            # No active macro, create one
            return self._create_new_macro(message, topic, message_vector)

        # Add as enrichment (knowledge_share, social, etc.)
        micro = self.timeline.add_child_to_macro(
            macro_id=active_macro.id,
            message_id=message.normalized.id,
            role=message.normalized.author_role,
            content_preview=message.clean_content[:200],
            is_question=False
        )

        return {
            'decision': 'enrichment',
            'macro_id': active_macro.id,
            'micro_id': micro.id,
            'topic': topic,
            'similarity': 0.0,
        }

    def get_context_for_prompt(self) -> str:
        """
        Generate context summary for LLM prompt.

        Instead of feeding raw chat history, feed structured macro context.
        """
        active_macro = self.timeline.get_active_macro_node()

        if not active_macro:
            return "No hay contexto activo."

        # Build context string
        context = f"[CONTEXTO ACTIVO: MACRO-NODE]\n"
        context += f"Tópico: {active_macro.main_topic}\n"
        context += f"Inicio: {active_macro.timestamp_start.strftime('%Y-%m-%d %H:%M')}\n"
        context += f"Resumen: {active_macro.summary}\n\n"

        context += "[HISTORIAL DENTRO DEL TÓPICO]\n"

        for child in active_macro.children[-10:]:  # Last 10 messages
            context += f"{child.role}: {child.content_preview}\n"

        return context

    def save_timeline(self, file_path: str):
        """Save timeline to JSON file."""
        import json
        from pathlib import Path

        Path(file_path).parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, 'w') as f:
            json.dump(self.timeline.to_dict(), f, indent=2)

    def load_timeline(self, file_path: str):
        """Load timeline from JSON file."""
        import json

        with open(file_path, 'r') as f:
            data = json.load(f)

        # TODO: Deserialize timeline from data
        # For now, just load user_id
        self.timeline = UserTimeline(user_id=data['user_id'])
