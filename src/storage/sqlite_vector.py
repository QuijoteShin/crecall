# src/storage/sqlite_vector.py
"""SQLite-based vector storage with optional sqlite-vec extension."""

import sqlite3
import json
import pickle
import re
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Any
import numpy as np

try:
    import sqlite_vec
    SQLITE_VEC_AVAILABLE = True
except ImportError:
    SQLITE_VEC_AVAILABLE = False
    sqlite_vec = None

from ..contracts.storage import IStorage
from ..models.processed import ProcessedMessage, Classification
from ..models.normalized import NormalizedMessage
from ..models.search import SearchResult
from ..models.hcs import MacroNode, MicroNode


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
        self.vec_dim = self.config.get("dimension", 256)

        self.conn: Optional[sqlite3.Connection] = None
        self._sqlite_vec_available = False

    def _connect(self) -> sqlite3.Connection:
        """Get or create database connection."""
        if self.conn is None:
            self.conn = sqlite3.connect(str(self.database_path))
            self.conn.row_factory = sqlite3.Row

            # Cargar sqlite-vec si disponible
            if SQLITE_VEC_AVAILABLE and sqlite_vec is not None:
                try:
                    self.conn.enable_load_extension(True)
                    sqlite_vec.load(self.conn)
                    self.conn.enable_load_extension(False)
                    self._sqlite_vec_available = True
                    print("✓ sqlite-vec cargado")
                except Exception as e:
                    self._sqlite_vec_available = False
                    print(f"⚠ sqlite-vec no disponible: {e}")
            else:
                self._sqlite_vec_available = False
                print("⚠ sqlite-vec no instalado, usando búsqueda vectorial fallback")

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

        # HCS: Tabla de MacroNodes (tópicos)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS hcs_macro_nodes (
                id TEXT PRIMARY KEY,
                main_topic TEXT NOT NULL,
                summary TEXT,
                status TEXT DEFAULT 'active',
                created_at TIMESTAMP NOT NULL,
                last_active TIMESTAMP,
                total_messages INTEGER DEFAULT 0,
                topics TEXT,
                entities TEXT,
                metadata TEXT,
                vector BLOB
            )
        """)

        # HCS: Tabla de MicroNodes (mensajes dentro de un macro)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS hcs_micro_nodes (
                id TEXT PRIMARY KEY,
                message_id TEXT NOT NULL,
                parent_macro_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content_preview TEXT,
                timestamp TIMESTAMP NOT NULL,
                is_question INTEGER DEFAULT 0,
                FOREIGN KEY (parent_macro_id) REFERENCES hcs_macro_nodes(id)
            )
        """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_micro_parent
            ON hcs_micro_nodes(parent_macro_id)
        """)

        # HCS: FTS para búsqueda de tópicos
        conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS hcs_macro_fts USING fts5(
                main_topic,
                summary,
                content='hcs_macro_nodes',
                content_rowid='rowid'
            )
        """)

        # HCS: Tabla vectorial con sqlite-vec (si disponible)
        if self._sqlite_vec_available:
            try:
                conn.execute(f"""
                    CREATE VIRTUAL TABLE IF NOT EXISTS vec_hcs_macro USING vec0(
                        macro_rowid INTEGER PRIMARY KEY,
                        vector float[{self.vec_dim}]
                    )
                """)
                print(f"✓ Tabla vectorial HCS creada (dim={self.vec_dim})")
            except Exception as e:
                print(f"⚠ No se pudo crear tabla vectorial: {e}")

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
        filters: Optional[dict] = None,
        query_text: Optional[str] = None,
        hybrid_boost: float = 0.3
    ) -> List[SearchResult]:
        """
        Vector similarity search with optional hybrid boosting.

        Args:
            query_vector: Embedding vector
            limit: Max results
            filters: Optional filters
            query_text: Original query text (for hybrid search)
            hybrid_boost: Boost factor for exact keyword matches (0.0-1.0)
        """
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

        where_sql = f"WHERE {' AND '.join(where_clauses)}"

        # Fetch all messages with vectors
        cursor = conn.execute(f"""
            SELECT * FROM messages
            {where_sql}
        """, params)

        results = []
        keywords = self._extract_keywords(query_text) if query_text else []

        for row in cursor:
            # Deserialize vector
            stored_vector = pickle.loads(row['vector'])

            # Calculate cosine similarity (vectors are normalized)
            vector_score = float(np.dot(query_vector, stored_vector))

            # Hybrid boosting: check for exact keyword matches
            keyword_boost = 0.0
            if keywords and hybrid_boost > 0:
                content_lower = row['vectorizable_content'].lower()

                for keyword in keywords:
                    if keyword.lower() in content_lower:
                        # Boost per keyword found
                        keyword_boost += hybrid_boost / len(keywords)

            # Combined score
            final_score = min(1.0, vector_score + keyword_boost)

            # Reconstruct ProcessedMessage
            message = self._row_to_processed_message(row)

            results.append(SearchResult(
                message=message,
                score=final_score,
                rank=0,  # Will be set after sorting
                matched_on='hybrid' if keyword_boost > 0 else 'vector',
                metadata={
                    'vector_score': vector_score,
                    'keyword_boost': keyword_boost,
                    'keywords_found': [k for k in keywords if k.lower() in row['vectorizable_content'].lower()]
                }
            ))

        # Sort by combined score
        results.sort(key=lambda x: x.score, reverse=True)

        # Set ranks and limit
        for i, result in enumerate(results[:limit], 1):
            result.rank = i

        return results[:limit]

    def _extract_keywords(self, query_text: Optional[str]) -> List[str]:
        """
        Extract meaningful keywords from query.

        Preserva acrónimos (SQL, API, AWS) y filtra stopwords.
        """
        if not query_text:
            return []

        # Extraer palabras preservando case original
        words = re.findall(r'\b\w+\b', query_text)

        # Stopwords bilingüe
        stopwords = {
            'es', 'de', 'la', 'el', 'en', 'y', 'a', 'que', 'los', 'las',
            'un', 'una', 'por', 'para', 'con', 'del', 'al',
            'is', 'the', 'of', 'and', 'to', 'in', 'for', 'with',
            'this', 'that', 'from', 'are', 'was', 'were', 'be', 'been'
        }

        # Lógica corregida: preserva acrónimos (mayúsculas) y palabras > 2 chars
        keywords = [
            w.lower() for w in words
            if w.lower() not in stopwords and (len(w) > 2 or w.isupper())
        ]

        # Remove duplicates while preserving order
        seen = set()
        unique_keywords = []
        for kw in keywords:
            if kw not in seen:
                seen.add(kw)
                unique_keywords.append(kw)

        return unique_keywords

    def search_by_topic(
        self,
        topic_query: str,
        limit: int = 10,
        use_similarity: bool = True
    ) -> List[SearchResult]:
        """
        Search by topic keywords.

        Args:
            topic_query: Topic to search for
            limit: Max results
            use_similarity: If True, also match similar topics (fuzzy)

        Returns:
            List of SearchResult objects
        """
        conn = self._connect()

        # Normalize topic query
        topic_lower = topic_query.lower().strip()

        results = []

        # Fetch all messages
        cursor = conn.execute("""
            SELECT * FROM messages
            WHERE topics != '[]'
        """)

        for row in cursor:
            topics = json.loads(row['topics'])

            # Check for exact or partial match
            match_score = 0.0

            for topic in topics:
                topic_clean = topic.lower()

                # Exact match
                if topic_clean == topic_lower:
                    match_score = 1.0
                    break

                # Partial match (contains)
                elif topic_lower in topic_clean or topic_clean in topic_lower:
                    match_score = max(match_score, 0.8)

                # Fuzzy similarity (simple)
                elif use_similarity:
                    # Calculate overlap ratio
                    common_chars = set(topic_lower) & set(topic_clean)
                    if common_chars:
                        similarity = len(common_chars) / max(len(topic_lower), len(topic_clean))
                        if similarity > 0.5:
                            match_score = max(match_score, similarity * 0.6)

            if match_score > 0:
                message = self._row_to_processed_message(row)

                results.append(SearchResult(
                    message=message,
                    score=match_score,
                    rank=0,
                    matched_on='topic',
                    metadata={
                        'matched_topics': [t for t in topics if topic_lower in t.lower()],
                        'all_topics': topics
                    }
                ))

        # Sort by score
        results.sort(key=lambda x: x.score, reverse=True)

        # Set ranks
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

        # Reconstruct normalized message
        normalized = NormalizedMessage(
            id=row['id'],
            conversation_id=row['conversation_id'],
            author_role=row['author_role'],
            content=row['content'],
            content_type=row['content_type'],
            timestamp=datetime.fromisoformat(row['timestamp']),
            metadata=json.loads(row['metadata']),
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

    # ==================== HCS Methods ====================

    def save_macro_node(self, node: MacroNode) -> str:
        """Persistir MacroNode con su vector."""
        conn = self._connect()
        cursor = conn.cursor()

        vector_blob = pickle.dumps(node.vector) if node.vector is not None else None

        cursor.execute("""
            INSERT OR REPLACE INTO hcs_macro_nodes
            (id, main_topic, summary, status, created_at, last_active,
             total_messages, topics, entities, metadata, vector)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            node.id,
            node.main_topic,
            node.summary,
            node.status,
            node.timestamp_start.isoformat(),
            node.timestamp_end.isoformat() if node.timestamp_end else None,
            node.total_messages,
            json.dumps(node.topics),
            json.dumps(node.entities),
            json.dumps(node.metadata),
            vector_blob
        ))

        # Obtener rowid para tabla vectorial
        cursor.execute("SELECT rowid FROM hcs_macro_nodes WHERE id = ?", (node.id,))
        row = cursor.fetchone()
        rowid = row[0] if row else None

        # Insertar en tabla vectorial sqlite-vec si disponible
        if self._sqlite_vec_available and node.vector is not None and rowid:
            try:
                vec_bytes = node.vector.astype(np.float32).tobytes()
                cursor.execute("""
                    INSERT OR REPLACE INTO vec_hcs_macro (macro_rowid, vector)
                    VALUES (?, ?)
                """, (rowid, vec_bytes))
            except Exception as e:
                print(f"⚠ Error insertando vector en sqlite-vec: {e}")

        conn.commit()
        return node.id

    def save_micro_node(self, node: MicroNode) -> str:
        """Persistir MicroNode."""
        conn = self._connect()

        conn.execute("""
            INSERT OR REPLACE INTO hcs_micro_nodes
            (id, message_id, parent_macro_id, role, content_preview, timestamp, is_question)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            node.id,
            node.message_id,
            node.parent_macro_id,
            node.role,
            node.content_preview,
            node.timestamp.isoformat(),
            int(node.is_question)
        ))

        conn.commit()
        return node.id

    def get_macro_node_by_id(self, macro_id: str) -> Optional[MacroNode]:
        """Obtener MacroNode por ID."""
        conn = self._connect()

        cursor = conn.execute("""
            SELECT * FROM hcs_macro_nodes WHERE id = ?
        """, (macro_id,))

        row = cursor.fetchone()
        if not row:
            return None

        return self._row_to_macro_node(row)

    def get_recent_macro_nodes(self, limit: int = 5, status: str = 'active') -> List[MacroNode]:
        """Obtener MacroNodes recientes."""
        conn = self._connect()

        cursor = conn.execute("""
            SELECT * FROM hcs_macro_nodes
            WHERE status = ?
            ORDER BY last_active DESC NULLS LAST, created_at DESC
            LIMIT ?
        """, (status, limit))

        return [self._row_to_macro_node(row) for row in cursor]

    def get_micro_nodes_for_macro(self, macro_id: str) -> List[MicroNode]:
        """Obtener MicroNodes de un MacroNode."""
        conn = self._connect()

        cursor = conn.execute("""
            SELECT * FROM hcs_micro_nodes
            WHERE parent_macro_id = ?
            ORDER BY timestamp ASC
        """, (macro_id,))

        return [self._row_to_micro_node(row) for row in cursor]

    def search_macro_nodes_hybrid(
        self,
        query_vector: np.ndarray,
        query_text: str,
        limit: int = 10,
        similarity_threshold: float = 0.4
    ) -> List[Dict[str, Any]]:
        """
        Búsqueda híbrida de MacroNodes: vectorial + keywords.

        Returns:
            Lista de dicts con 'node', 'score', 'matched_on'
        """
        conn = self._connect()
        results = []
        keywords = self._extract_keywords(query_text)

        if self._sqlite_vec_available:
            try:
                vec_bytes = query_vector.astype(np.float32).tobytes()
                cursor = conn.execute("""
                    SELECT
                        m.*,
                        vec_distance_cosine(v.vector, ?) as distance
                    FROM vec_hcs_macro v
                    JOIN hcs_macro_nodes m ON m.rowid = v.macro_rowid
                    WHERE m.status = 'active'
                    ORDER BY distance ASC
                    LIMIT ?
                """, (vec_bytes, limit * 2))

                for row in cursor:
                    similarity = 1 - row['distance']
                    if similarity >= similarity_threshold:
                        node = self._row_to_macro_node(row)
                        keyword_boost = self._calc_keyword_boost(node, keywords)
                        results.append({
                            'node': node,
                            'score': similarity + keyword_boost,
                            'vector_score': similarity,
                            'keyword_boost': keyword_boost,
                            'matched_on': 'sqlite-vec'
                        })
            except Exception as e:
                print(f"⚠ sqlite-vec search failed, using fallback: {e}")
                results = self._search_macro_fallback(query_vector, keywords, limit, similarity_threshold)
        else:
            results = self._search_macro_fallback(query_vector, keywords, limit, similarity_threshold)

        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:limit]

    def _search_macro_fallback(
        self,
        query_vector: np.ndarray,
        keywords: List[str],
        limit: int,
        threshold: float
    ) -> List[Dict[str, Any]]:
        """Búsqueda vectorial fallback en Python."""
        conn = self._connect()
        results = []

        cursor = conn.execute("""
            SELECT * FROM hcs_macro_nodes
            WHERE status = 'active' AND vector IS NOT NULL
        """)

        for row in cursor:
            stored_vector = pickle.loads(row['vector'])
            similarity = float(np.dot(query_vector, stored_vector))

            if similarity >= threshold:
                node = self._row_to_macro_node(row)
                keyword_boost = self._calc_keyword_boost(node, keywords)
                results.append({
                    'node': node,
                    'score': similarity + keyword_boost,
                    'vector_score': similarity,
                    'keyword_boost': keyword_boost,
                    'matched_on': 'fallback'
                })

        return results

    def _calc_keyword_boost(self, node: MacroNode, keywords: List[str]) -> float:
        """Calcular boost por coincidencia de keywords."""
        if not keywords:
            return 0.0

        content = f"{node.main_topic} {node.summary or ''} {' '.join(node.topics)}".lower()
        matches = sum(1 for kw in keywords if kw in content)

        if matches > 0:
            return 0.15 + (0.05 * min(matches, 5))
        return 0.0

    def _row_to_macro_node(self, row: sqlite3.Row) -> MacroNode:
        """Convertir row de BD a MacroNode."""
        vector = None
        if row['vector']:
            vector = pickle.loads(row['vector'])

        created_at = row['created_at']
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)

        last_active = row['last_active']
        if last_active and isinstance(last_active, str):
            last_active = datetime.fromisoformat(last_active)

        return MacroNode(
            id=row['id'],
            main_topic=row['main_topic'],
            summary=row['summary'] or "",
            status=row['status'],
            timestamp_start=created_at,
            timestamp_end=last_active,
            total_messages=row['total_messages'] or 0,
            topics=json.loads(row['topics']) if row['topics'] else [],
            entities=json.loads(row['entities']) if row['entities'] else [],
            metadata=json.loads(row['metadata']) if row['metadata'] else {},
            vector=vector
        )

    def _row_to_micro_node(self, row: sqlite3.Row) -> MicroNode:
        """Convertir row de BD a MicroNode."""
        timestamp = row['timestamp']
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)

        return MicroNode(
            id=row['id'],
            message_id=row['message_id'],
            parent_macro_id=row['parent_macro_id'],
            role=row['role'],
            content_preview=row['content_preview'] or "",
            timestamp=timestamp,
            is_question=bool(row['is_question'])
        )
