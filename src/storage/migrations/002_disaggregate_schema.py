# src/storage/migrations/002_disaggregate_schema.py
"""
Migration: Disaggregate schema for MariaDB compatibility.

Creates:
- message_content: TEXT fields separated
- message_chunks: VARCHAR-safe chunks for long messages
- vectors: Dedicated vector storage (migrable to VECTOR type)

Includes hotfix to migrate existing refined data.
"""

import sqlite3
import pickle
from pathlib import Path
from typing import Optional


def migrate(db_path: str, verbose: bool = True) -> dict:
    """
    Run the disaggregation migration.

    Returns:
        dict with migration stats
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    stats = {
        'content_migrated': 0,
        'vectors_migrated': 0,
        'chunks_created': 0,
        'tables_created': []
    }

    if verbose:
        print("\n=== Migration 002: Disaggregate Schema ===\n")

    # 1. Create message_content table
    if verbose:
        print("Creating message_content table...")

    conn.execute("""
        CREATE TABLE IF NOT EXISTS message_content (
            message_id TEXT PRIMARY KEY,
            full_content TEXT,
            clean_content TEXT,
            vectorizable_content TEXT,
            refined_content TEXT,
            FOREIGN KEY (message_id) REFERENCES messages(id)
        )
    """)
    stats['tables_created'].append('message_content')

    # 2. Create message_chunks table
    if verbose:
        print("Creating message_chunks table...")

    conn.execute("""
        CREATE TABLE IF NOT EXISTS message_chunks (
            chunk_id TEXT PRIMARY KEY,
            message_id TEXT NOT NULL,
            sequence INTEGER NOT NULL,
            chunk_text VARCHAR(4000),
            refined_text VARCHAR(4000),
            topic_inherited TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (message_id) REFERENCES messages(id)
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_message ON message_chunks(message_id)")
    stats['tables_created'].append('message_chunks')

    # 3. Create vectors table
    if verbose:
        print("Creating vectors table...")

    conn.execute("""
        CREATE TABLE IF NOT EXISTS vectors (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            entity_id TEXT NOT NULL,
            entity_type TEXT NOT NULL CHECK(entity_type IN ('message', 'chunk')),
            vector BLOB,
            vector_dim INTEGER,
            model_version TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(entity_id, entity_type)
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_vectors_entity ON vectors(entity_id, entity_type)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_vectors_model ON vectors(model_version)")
    stats['tables_created'].append('vectors')

    # 4. Migrate existing content to message_content
    if verbose:
        print("\nMigrating content from messages table...")

    cursor = conn.execute("""
        SELECT id, content, clean_content, vectorizable_content, refined_content
        FROM messages
        WHERE id NOT IN (SELECT message_id FROM message_content)
    """)

    batch = []
    for row in cursor:
        batch.append((
            row['id'],
            row['content'],
            row['clean_content'],
            row['vectorizable_content'],
            row['refined_content']
        ))

        if len(batch) >= 1000:
            conn.executemany("""
                INSERT OR IGNORE INTO message_content
                (message_id, full_content, clean_content, vectorizable_content, refined_content)
                VALUES (?, ?, ?, ?, ?)
            """, batch)
            stats['content_migrated'] += len(batch)
            if verbose:
                print(f"  Migrated {stats['content_migrated']} content rows...")
            batch = []

    if batch:
        conn.executemany("""
            INSERT OR IGNORE INTO message_content
            (message_id, full_content, clean_content, vectorizable_content, refined_content)
            VALUES (?, ?, ?, ?, ?)
        """, batch)
        stats['content_migrated'] += len(batch)

    if verbose:
        print(f"  ✓ Migrated {stats['content_migrated']} content rows")

    # 5. Migrate existing vectors
    if verbose:
        print("\nMigrating vectors from messages table...")

    cursor = conn.execute("""
        SELECT id, vector, vectorizer_version
        FROM messages
        WHERE vector IS NOT NULL
        AND id NOT IN (SELECT entity_id FROM vectors WHERE entity_type = 'message')
    """)

    batch = []
    for row in cursor:
        vector_blob = row['vector']
        if vector_blob:
            # Get dimension from unpickled vector
            try:
                vec = pickle.loads(vector_blob)
                dim = len(vec)
            except:
                dim = None

            batch.append((
                row['id'],
                'message',
                vector_blob,
                dim,
                row['vectorizer_version']
            ))

        if len(batch) >= 1000:
            conn.executemany("""
                INSERT OR IGNORE INTO vectors
                (entity_id, entity_type, vector, vector_dim, model_version)
                VALUES (?, ?, ?, ?, ?)
            """, batch)
            stats['vectors_migrated'] += len(batch)
            if verbose:
                print(f"  Migrated {stats['vectors_migrated']} vectors...")
            batch = []

    if batch:
        conn.executemany("""
            INSERT OR IGNORE INTO vectors
            (entity_id, entity_type, vector, vector_dim, model_version)
            VALUES (?, ?, ?, ?, ?)
        """, batch)
        stats['vectors_migrated'] += len(batch)

    if verbose:
        print(f"  ✓ Migrated {stats['vectors_migrated']} vectors")

    # 6. Create chunks for long messages (> 4000 chars)
    if verbose:
        print("\nCreating chunks for long messages...")

    CHUNK_SIZE = 3800  # Leave margin for VARCHAR(4000)

    cursor = conn.execute("""
        SELECT m.id, mc.vectorizable_content, m.segment_topic
        FROM messages m
        JOIN message_content mc ON m.id = mc.message_id
        WHERE LENGTH(mc.vectorizable_content) > 4000
        AND m.id NOT IN (SELECT DISTINCT message_id FROM message_chunks)
    """)

    for row in cursor:
        message_id = row['id']
        content = row['vectorizable_content'] or ''
        topic = row['segment_topic']

        # Split into chunks
        chunks = []
        for i in range(0, len(content), CHUNK_SIZE):
            chunk_text = content[i:i + CHUNK_SIZE]
            chunks.append(chunk_text)

        # Insert chunks
        for seq, chunk_text in enumerate(chunks, 1):
            chunk_id = f"{message_id}_chunk{seq}"
            conn.execute("""
                INSERT OR IGNORE INTO message_chunks
                (chunk_id, message_id, sequence, chunk_text, topic_inherited)
                VALUES (?, ?, ?, ?, ?)
            """, (chunk_id, message_id, seq, chunk_text, topic))
            stats['chunks_created'] += 1

    if verbose:
        print(f"  ✓ Created {stats['chunks_created']} chunks")

    # 7. Create FTS for chunks
    if verbose:
        print("\nCreating FTS index for chunks...")

    conn.execute("DROP TABLE IF EXISTS message_chunks_fts")
    conn.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS message_chunks_fts USING fts5(
            chunk_id UNINDEXED,
            message_id UNINDEXED,
            chunk_text,
            refined_text,
            topic_inherited,
            content='message_chunks',
            content_rowid='rowid'
        )
    """)

    # Populate FTS
    conn.execute("""
        INSERT INTO message_chunks_fts(rowid, chunk_id, message_id, chunk_text, refined_text, topic_inherited)
        SELECT rowid, chunk_id, message_id, chunk_text, refined_text, topic_inherited FROM message_chunks
    """)

    # Create trigger for new chunks
    conn.execute("DROP TRIGGER IF EXISTS chunks_fts_insert")
    conn.execute("""
        CREATE TRIGGER chunks_fts_insert AFTER INSERT ON message_chunks BEGIN
            INSERT INTO message_chunks_fts(rowid, chunk_id, message_id, chunk_text, refined_text, topic_inherited)
            VALUES (new.rowid, new.chunk_id, new.message_id, new.chunk_text, new.refined_text, new.topic_inherited);
        END
    """)

    conn.commit()

    if verbose:
        print("\n=== Migration Complete ===")
        print(f"  Tables created: {', '.join(stats['tables_created'])}")
        print(f"  Content migrated: {stats['content_migrated']}")
        print(f"  Vectors migrated: {stats['vectors_migrated']}")
        print(f"  Chunks created: {stats['chunks_created']}")

    conn.close()
    return stats


def verify(db_path: str) -> dict:
    """Verify migration was successful."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    result = {
        'message_content_count': 0,
        'vectors_count': 0,
        'chunks_count': 0,
        'vectors_by_dim': {},
        'orphaned_vectors': 0
    }

    # Count records
    result['message_content_count'] = conn.execute(
        "SELECT COUNT(*) FROM message_content"
    ).fetchone()[0]

    result['vectors_count'] = conn.execute(
        "SELECT COUNT(*) FROM vectors"
    ).fetchone()[0]

    result['chunks_count'] = conn.execute(
        "SELECT COUNT(*) FROM message_chunks"
    ).fetchone()[0]

    # Vectors by dimension
    cursor = conn.execute("""
        SELECT vector_dim, COUNT(*) as cnt
        FROM vectors
        GROUP BY vector_dim
    """)
    for row in cursor:
        result['vectors_by_dim'][row['vector_dim']] = row['cnt']

    conn.close()
    return result


if __name__ == "__main__":
    import sys
    db_path = sys.argv[1] if len(sys.argv) > 1 else "data/recall.db"

    stats = migrate(db_path)
    print("\nVerifying...")
    verify_result = verify(db_path)
    print(f"  message_content: {verify_result['message_content_count']}")
    print(f"  vectors: {verify_result['vectors_count']}")
    print(f"  chunks: {verify_result['chunks_count']}")
    print(f"  vectors by dim: {verify_result['vectors_by_dim']}")
