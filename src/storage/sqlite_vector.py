# src/storage/sqlite_vector.py
"""SQLite-based vector storage with optional sqlite-vec extension."""

import sqlite3
import json
import pickle
from pathlib import Path
from typing import List, Optional, Dict, Any
import numpy as np

from ..contracts.storage import IStorage
from ..models.processed import ProcessedMessage, Classification
from ..models.normalized import NormalizedMessage
from ..models.search import SearchResult


class SQLiteVectorStorage(IStorage):
    """
    SQLite storage with vector search capabilities.

    Uses:
    - FTS5 for full-text search
    - Pickle-serialized numpy arrays for vectors (fallback if sqlite-vec not available)
    - Optional sqlite-vec extension for optimized vector search
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.database_path = Path(self.config.get("database", "data/recall.db"))
        self.database_path.parent.mkdir(parents=True, exist_ok=True)

        self.conn: Optional[sqlite3.Connection] = None
        self._sqlite_vec_available = False

    def _connect(self) -> sqlite3.Connection:
        """Get or create database connection."""
        if self.conn is None:
            self.conn = sqlite3.connect(str(self.database_path))
            self.conn.row_factory = sqlite3.Row

            # Try to load sqlite-vec extension
            try:
                self.conn.enable_load_extension(True)
                import sqlite_vec
                self.conn.load_extension(sqlite_vec.loadable_path())
                self._sqlite_vec_available = True
                # print("✓ sqlite-vec extension loaded")
            except Exception as e:
                self._sqlite_vec_available = False
                print(f"⚠ sqlite-vec not available, using fallback vector search: {e}")

        return self.conn

    def initialize(self) -> None:
        """Create database schema."""
        conn = self._connect()

        # Messages table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                author_role TEXT NOT NULL,
                content TEXT NOT NULL,
                clean_content TEXT NOT NULL,
                vectorizable_content TEXT NOT NULL,
                content_type TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                parent_message_id TEXT,

                -- Classification
                intent TEXT NOT NULL,
                intent_confidence REAL NOT NULL,
                topics TEXT NOT NULL,  -- JSON array
                is_question INTEGER NOT NULL,
                is_command INTEGER NOT NULL,

                -- Semantic Segmentation (HCS Layer)
                segment_id TEXT,  -- Segment identifier (e.g., "conv123_seg1")
                segment_topic TEXT,  -- Auto-generated topic label (e.g., "Salud Visual")
                segment_sequence INTEGER,  -- Sequence number within conversation (1, 2, 3...)

                -- Metadata
                metadata TEXT NOT NULL,  -- JSON object

                -- Deduplication & versioning
                content_hash TEXT,  -- SHA256 of content for deduplication
                vectorizer_version TEXT,  -- Tracks which vectorizer was used

                -- Vector (pickled numpy array)
                vector BLOB,

                -- Indexes
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Source files tracking (for deduplication)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS source_files (
                file_hash TEXT PRIMARY KEY,
                file_path TEXT NOT NULL,
                file_size INTEGER NOT NULL,
                processed_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                total_messages INTEGER NOT NULL,
                vectorizer_version TEXT,
                metadata TEXT  -- JSON with additional info
            )
        """)

        # Vectorizer versions tracking
        conn.execute("""
            CREATE TABLE IF NOT EXISTS vectorizer_versions (
                version_id TEXT PRIMARY KEY,
                model_name TEXT NOT NULL,
                embedding_dim INTEGER NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                config TEXT  -- JSON with vectorizer config
            )
        """)

        # Indexes
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_conversation
            ON messages(conversation_id)
        """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp
            ON messages(timestamp DESC)
        """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_intent
            ON messages(intent)
        """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_segment_topic
            ON messages(segment_topic)
        """)

        # Full-text search table
        conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
                id UNINDEXED,
                content,
                clean_content,
                vectorizable_content,
                content='messages',
                content_rowid='rowid'
            )
        """)

        # FTS triggers
        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS messages_fts_insert AFTER INSERT ON messages BEGIN
                INSERT INTO messages_fts(rowid, id, content, clean_content, vectorizable_content)
                VALUES (new.rowid, new.id, new.content, new.clean_content, new.vectorizable_content);
            END
        """)

        conn.commit()
        print("✓ Database initialized")

    def save_message(self, message: ProcessedMessage) -> str:
        """Persist a processed message."""
        conn = self._connect()

        # Serialize vector
        vector_blob = None
        if message.vector is not None:
            vector_blob = pickle.dumps(message.vector)

        # Prepare data
        data = {
            'id': message.normalized.id,
            'conversation_id': message.normalized.conversation_id,
            'author_role': message.normalized.author_role,
            'content': message.normalized.content,
            'clean_content': message.clean_content,
            'vectorizable_content': message.vectorizable_content,
            'content_type': message.normalized.content_type,
            'timestamp': message.normalized.timestamp.isoformat(),
            'parent_message_id': message.normalized.parent_message_id,
            'intent': message.classification.intent,
            'intent_confidence': message.classification.confidence,
            'topics': json.dumps(message.classification.topics),
            'is_question': int(message.classification.is_question),
            'is_command': int(message.classification.is_command),
            'metadata': json.dumps(message.normalized.metadata),
            'vector': vector_blob,
        }

        conn.execute("""
            INSERT OR REPLACE INTO messages (
                id, conversation_id, author_role, content, clean_content, vectorizable_content,
                content_type, timestamp, parent_message_id, intent, intent_confidence,
                topics, is_question, is_command, metadata, vector
            ) VALUES (
                :id, :conversation_id, :author_role, :content, :clean_content, :vectorizable_content,
                :content_type, :timestamp, :parent_message_id, :intent, :intent_confidence,
                :topics, :is_question, :is_command, :metadata, :vector
            )
        """, data)

        conn.commit()
        return message.normalized.id

    def save_messages_batch(self, messages: List[ProcessedMessage]) -> List[str]:
        """Bulk insert for efficiency."""
        message_ids = []

        conn = self._connect()

        for message in messages:
            # Serialize vector
            vector_blob = None
            if message.vector is not None:
                vector_blob = pickle.dumps(message.vector)

            data = {
                'id': message.normalized.id,
                'conversation_id': message.normalized.conversation_id,
                'author_role': message.normalized.author_role,
                'content': message.normalized.content,
                'clean_content': message.clean_content,
                'vectorizable_content': message.vectorizable_content,
                'content_type': message.normalized.content_type,
                'timestamp': message.normalized.timestamp.isoformat(),
                'parent_message_id': message.normalized.parent_message_id,
                'intent': message.classification.intent,
                'intent_confidence': message.classification.confidence,
                'topics': json.dumps(message.classification.topics),
                'is_question': int(message.classification.is_question),
                'is_command': int(message.classification.is_command),
                'metadata': json.dumps(message.normalized.metadata),
                'vector': vector_blob,
            }

            conn.execute("""
                INSERT OR REPLACE INTO messages (
                    id, conversation_id, author_role, content, clean_content, vectorizable_content,
                    content_type, timestamp, parent_message_id, intent, intent_confidence,
                    topics, is_question, is_command, metadata, vector
                ) VALUES (
                    :id, :conversation_id, :author_role, :content, :clean_content, :vectorizable_content,
                    :content_type, :timestamp, :parent_message_id, :intent, :intent_confidence,
                    :topics, :is_question, :is_command, :metadata, :vector
                )
            """, data)

            message_ids.append(message.normalized.id)

        conn.commit()
        return message_ids

    def search_by_vector(
        self,
        query_vector: np.ndarray,
        limit: int = 10,
        filters: Optional[dict] = None
    ) -> List[SearchResult]:
        """Vector similarity search."""
        conn = self._connect()

        # Build filter WHERE clause
        where_clauses = ["vector IS NOT NULL"]
        params = {}

        if filters:
            if 'conversation_id' in filters:
                where_clauses.append("conversation_id = :conversation_id")
                params['conversation_id'] = filters['conversation_id']
            if 'intent' in filters:
                where_clauses.append("intent = :intent")
                params['intent'] = filters['intent']
            if 'segment_topic' in filters:
                where_clauses.append("segment_topic = :segment_topic")
                params['segment_topic'] = filters['segment_topic']

        where_sql = f"WHERE {' AND '.join(where_clauses)}"

        # Fetch all messages with vectors
        cursor = conn.execute(f"""
            SELECT * FROM messages
            {where_sql}
        """, params)

        results = []
        for row in cursor:
            # Deserialize vector
            stored_vector = pickle.loads(row['vector'])

            # Calculate cosine similarity (vectors are normalized)
            similarity = float(np.dot(query_vector, stored_vector))

            # Reconstruct ProcessedMessage
            message = self._row_to_processed_message(row)

            results.append(SearchResult(
                message=message,
                score=similarity,
                rank=0,  # Will be set after sorting
                matched_on='vector'
            ))

        # Sort by similarity
        results.sort(key=lambda x: x.score, reverse=True)

        # Set ranks and limit
        for i, result in enumerate(results[:limit], 1):
            result.rank = i

        return results[:limit]

    def search_by_text(
        self,
        query: str,
        limit: int = 10,
        filters: Optional[dict] = None
    ) -> List[SearchResult]:
        """Full-text search."""
        conn = self._connect()

        # Build filter WHERE clause
        where_clauses = ["messages_fts.id = messages.id"]
        params = {'query': query}

        if filters:
            if 'conversation_id' in filters:
                where_clauses.append("conversation_id = :conversation_id")
                params['conversation_id'] = filters['conversation_id']
            if 'intent' in filters:
                where_clauses.append("intent = :intent")
                params['intent'] = filters['intent']
            if 'segment_topic' in filters:
                where_clauses.append("segment_topic = :segment_topic")
                params['segment_topic'] = filters['segment_topic']

        where_sql = ' AND '.join(where_clauses)

        cursor = conn.execute(f"""
            SELECT messages.*, rank
            FROM messages_fts
            JOIN messages ON messages_fts.id = messages.id
            WHERE {where_sql} AND messages_fts MATCH :query
            ORDER BY rank
            LIMIT :limit
        """, {**params, 'limit': limit})

        results = []
        for i, row in enumerate(cursor, 1):
            message = self._row_to_processed_message(row)

            results.append(SearchResult(
                message=message,
                score=1.0 / (i + 1),  # Simple rank-based score
                rank=i,
                matched_on='text'
            ))

        return results

    def get_message_by_id(self, message_id: str) -> Optional[ProcessedMessage]:
        """Retrieve single message by ID."""
        conn = self._connect()

        cursor = conn.execute("""
            SELECT * FROM messages WHERE id = ?
        """, (message_id,))

        row = cursor.fetchone()
        if row:
            return self._row_to_processed_message(row)
        return None

    def get_conversation_messages(self, conversation_id: str) -> List[ProcessedMessage]:
        """Get all messages in a conversation."""
        conn = self._connect()

        cursor = conn.execute("""
            SELECT * FROM messages
            WHERE conversation_id = ?
            ORDER BY timestamp ASC
        """, (conversation_id,))

        return [self._row_to_processed_message(row) for row in cursor]

    def get_statistics(self) -> dict:
        """Return storage statistics."""
        conn = self._connect()

        stats = {}

        # Total messages
        cursor = conn.execute("SELECT COUNT(*) as count FROM messages")
        stats['total_messages'] = cursor.fetchone()['count']

        # Total conversations
        cursor = conn.execute("SELECT COUNT(DISTINCT conversation_id) as count FROM messages")
        stats['total_conversations'] = cursor.fetchone()['count']

        # Database size
        stats['size_mb'] = self.database_path.stat().st_size / 1024 / 1024 if self.database_path.exists() else 0

        # Intent distribution
        cursor = conn.execute("""
            SELECT intent, COUNT(*) as count
            FROM messages
            GROUP BY intent
        """)
        stats['intent_distribution'] = {row['intent']: row['count'] for row in cursor}

        return stats

    def close(self) -> None:
        """Close connection."""
        if self.conn:
            self.conn.close()
            self.conn = None

    def _row_to_processed_message(self, row: sqlite3.Row) -> ProcessedMessage:
        """Convert database row to ProcessedMessage."""
        from datetime import datetime

        # Deserialize vector
        vector = None
        if row['vector']:
            vector = pickle.loads(row['vector'])

        # Reconstruct normalized message with segment info
        metadata = json.loads(row['metadata'])

        # Add segment information if available
        if 'segment_topic' in row.keys() and row['segment_topic']:
            metadata['segment_topic'] = row['segment_topic']
            metadata['segment_id'] = row.get('segment_id')
            metadata['segment_sequence'] = row.get('segment_sequence')

        normalized = NormalizedMessage(
            id=row['id'],
            conversation_id=row['conversation_id'],
            author_role=row['author_role'],
            content=row['content'],
            content_type=row['content_type'],
            timestamp=datetime.fromisoformat(row['timestamp']),
            metadata=metadata,
            parent_message_id=row['parent_message_id'],
        )

        # Reconstruct classification
        classification = Classification(
            intent=row['intent'],
            confidence=row['intent_confidence'],
            topics=json.loads(row['topics']),
        )

        return ProcessedMessage(
            normalized=normalized,
            clean_content=row['clean_content'],
            vectorizable_content=row['vectorizable_content'],
            classification=classification,
            vector=vector,
        )

    # Deduplication & Re-indexing methods

    def check_file_processed(self, file_hash: str) -> bool:
        """Check if source file was already processed."""
        conn = self._connect()
        cursor = conn.execute("""
            SELECT 1 FROM source_files WHERE file_hash = ?
        """, (file_hash,))
        return cursor.fetchone() is not None

    def register_source_file(
        self,
        file_hash: str,
        file_path: str,
        file_size: int,
        total_messages: int,
        vectorizer_version: str,
        metadata: Optional[Dict] = None
    ) -> None:
        """Register processed source file."""
        conn = self._connect()
        conn.execute("""
            INSERT OR REPLACE INTO source_files
            (file_hash, file_path, file_size, total_messages, vectorizer_version, metadata)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            file_hash,
            file_path,
            file_size,
            total_messages,
            vectorizer_version,
            json.dumps(metadata or {})
        ))
        conn.commit()

    def register_vectorizer_version(
        self,
        version_id: str,
        model_name: str,
        embedding_dim: int,
        config: Optional[Dict] = None
    ) -> None:
        """Register vectorizer version."""
        conn = self._connect()
        conn.execute("""
            INSERT OR REPLACE INTO vectorizer_versions
            (version_id, model_name, embedding_dim, config)
            VALUES (?, ?, ?, ?)
        """, (
            version_id,
            model_name,
            embedding_dim,
            json.dumps(config or {})
        ))
        conn.commit()

    def reindex_vectors(
        self,
        new_vectors: Dict[str, np.ndarray],
        vectorizer_version: str
    ) -> int:
        """
        Update vectors for existing messages (re-indexing).

        Args:
            new_vectors: Dict mapping message_id to new vector
            vectorizer_version: New vectorizer version ID

        Returns:
            Number of messages updated
        """
        conn = self._connect()
        count = 0

        for message_id, vector in new_vectors.items():
            vector_blob = pickle.dumps(vector)

            conn.execute("""
                UPDATE messages
                SET vector = ?, vectorizer_version = ?
                WHERE id = ?
            """, (vector_blob, vectorizer_version, message_id))

            count += 1

        conn.commit()
        return count

    def get_messages_for_reindex(
        self,
        vectorizer_version: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[ProcessedMessage]:
        """
        Get messages that need re-indexing.

        Args:
            vectorizer_version: If provided, only get messages with different version
            limit: Max messages to return

        Returns:
            List of ProcessedMessage objects
        """
        conn = self._connect()

        where_clause = ""
        params = []

        if vectorizer_version:
            where_clause = "WHERE vectorizer_version != ? OR vectorizer_version IS NULL"
            params.append(vectorizer_version)

        limit_clause = f"LIMIT {limit}" if limit else ""

        cursor = conn.execute(f"""
            SELECT * FROM messages
            {where_clause}
            ORDER BY created_at DESC
            {limit_clause}
        """, params)

        return [self._row_to_processed_message(row) for row in cursor]

    def migrate_add_segmentation_columns(self) -> None:
        """Add segmentation columns to existing database (migration)."""
        conn = self._connect()

        try:
            conn.execute("ALTER TABLE messages ADD COLUMN segment_id TEXT")
            print("✓ Added column: segment_id")
        except:
            pass  # Column already exists

        try:
            conn.execute("ALTER TABLE messages ADD COLUMN segment_topic TEXT")
            print("✓ Added column: segment_topic")
        except:
            pass

        try:
            conn.execute("ALTER TABLE messages ADD COLUMN segment_sequence INTEGER")
            print("✓ Added column: segment_sequence")
        except:
            pass

        try:
            conn.execute("CREATE INDEX IF NOT EXISTS idx_segment_topic ON messages(segment_topic)")
            print("✓ Created index: idx_segment_topic")
        except:
            pass

        conn.commit()

    def get_conversations_for_segmentation(self) -> List[str]:
        """Get list of conversation IDs that need segmentation."""
        conn = self._connect()

        cursor = conn.execute("""
            SELECT DISTINCT conversation_id
            FROM messages
            WHERE segment_id IS NULL AND vector IS NOT NULL
            ORDER BY conversation_id
        """)

        return [row[0] for row in cursor]

    def update_message_segment(
        self,
        message_id: str,
        segment_id: str,
        segment_topic: str,
        segment_sequence: int
    ) -> None:
        """Update segment information for a message."""
        conn = self._connect()

        conn.execute("""
            UPDATE messages
            SET segment_id = ?, segment_topic = ?, segment_sequence = ?
            WHERE id = ?
        """, (segment_id, segment_topic, segment_sequence, message_id))

        conn.commit()

    def update_segments_batch(self, segments: List[Dict[str, Any]]) -> int:
        """
        Batch update segment information.

        Args:
            segments: List of dicts with keys: message_id, segment_id, segment_topic, segment_sequence

        Returns:
            Number of messages updated
        """
        conn = self._connect()

        updates = [
            (s['segment_id'], s['segment_topic'], s['segment_sequence'], s['message_id'])
            for s in segments
        ]

        conn.executemany("""
            UPDATE messages
            SET segment_id = ?, segment_topic = ?, segment_sequence = ?
            WHERE id = ?
        """, updates)

        conn.commit()
        return len(updates)
