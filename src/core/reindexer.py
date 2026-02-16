# src/core/reindexer.py
"""Re-indexer with checkpointing support for resumable batch processing."""

import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn

from .registry import ComponentRegistry
from ..contracts.refiner import IRefiner
from ..contracts.vectorizer import IVectorizer
from ..contracts.storage import IStorage
from ..models.processed import Classification

console = Console()

CHECKPOINT_FILE = Path("data/reindex_checkpoint.json")


class Reindexer:
    """
    Re-indexes messages with checkpointing for resumable processing.

    Features:
    - Checkpoint every N messages (configurable)
    - Resume from last checkpoint on failure/restart
    - Progress tracking with ETA
    - Memory-efficient batch processing
    """

    def __init__(self, config: Dict[str, Any], registry: ComponentRegistry):
        self.config = config
        self.registry = registry
        self.checkpoint_interval = config.get('reindex', {}).get('checkpoint_interval', 100)

    def reindex_all(
        self,
        force: bool = False,
        resume: bool = True,
        batch_size: int = 8
    ) -> Dict[str, Any]:
        """
        Re-index all messages with new refiner prompt.

        Args:
            force: If True, re-index ALL messages. If False, only unprocessed.
            resume: If True, resume from checkpoint. If False, start fresh.
            batch_size: Messages per batch (default 8)

        Returns:
            Statistics dictionary
        """
        stats = {
            'start_time': datetime.now().isoformat(),
            'processed': 0,
            'errors': 0,
            'skipped': 0,
            'resumed_from': None,
        }

        # Load checkpoint if resuming
        checkpoint = self._load_checkpoint() if resume else None
        if checkpoint:
            stats['resumed_from'] = checkpoint.get('last_processed_id')
            console.print(f"[yellow]Resuming from checkpoint: {checkpoint['processed']} already done[/yellow]")

        # Get storage
        storage = self._get_storage()

        # Get messages to process
        messages = self._get_messages_to_reindex(storage, force, checkpoint)
        total = len(messages)

        if total == 0:
            console.print("[green]✓ All messages already indexed[/green]")
            return stats

        console.print(f"\n[bold]Re-indexing {total} messages[/bold]")
        console.print(f"  Checkpoint interval: {self.checkpoint_interval}")
        console.print(f"  Batch size: {batch_size}")

        # Load models
        refiner = self._get_refiner()
        vectorizer = self._get_vectorizer()

        console.print("\n[bold]Loading models...[/bold]")
        refiner.load()
        vectorizer.load()

        start_time = time.time()
        next_checkpoint = self.checkpoint_interval

        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TimeElapsedColumn(),
                console=console,
                refresh_per_second=1
            ) as progress:
                task = progress.add_task("Re-indexing...", total=total)

                for i in range(0, total, batch_size):
                    batch = messages[i:i + batch_size]

                    try:
                        self._process_batch(batch, refiner, vectorizer, storage)
                        stats['processed'] += len(batch)
                    except Exception as e:
                        console.print(f"\n[red]Error in batch {i}: {e}[/red]")
                        stats['errors'] += len(batch)

                    progress.update(task, advance=len(batch))

                    # Checkpoint (uses >= threshold to work with any batch size)
                    if stats['processed'] >= next_checkpoint:
                        self._save_checkpoint({
                            'processed': stats['processed'],
                            'last_processed_id': batch[-1]['id'],
                            'timestamp': datetime.now().isoformat(),
                            'force': force,
                        })
                        next_checkpoint = stats['processed'] + self.checkpoint_interval

                        # Show progress
                        elapsed = time.time() - start_time
                        rate = stats['processed'] / elapsed if elapsed > 0 else 0
                        remaining = (total - stats['processed']) / rate if rate > 0 else 0
                        console.print(f"\n  [dim]Checkpoint: {stats['processed']}/{total} ({rate:.2f} msg/s, ~{remaining/60:.0f}min remaining)[/dim]")

        finally:
            # Unload models
            console.print("\n[bold]Unloading models...[/bold]")
            refiner.unload()
            vectorizer.unload()

        # Clear checkpoint on success
        if stats['errors'] == 0:
            self._clear_checkpoint()

        # Rebuild FTS index para incluir refined_content actualizado
        if stats['processed'] > 0:
            console.print("\n[bold]Rebuilding FTS index...[/bold]")
            self._rebuild_fts_index(storage)
            console.print("  [green]✓[/green] FTS index rebuilt")

        # Final stats
        elapsed = time.time() - start_time
        stats['end_time'] = datetime.now().isoformat()
        stats['elapsed_seconds'] = elapsed
        stats['messages_per_second'] = stats['processed'] / elapsed if elapsed > 0 else 0

        console.print(f"\n[bold green]✓ Re-indexing complete[/bold green]")
        console.print(f"  Processed: {stats['processed']}")
        console.print(f"  Errors: {stats['errors']}")
        console.print(f"  Time: {elapsed/60:.1f} minutes")
        console.print(f"  Speed: {stats['messages_per_second']:.2f} msg/s")

        return stats

    def _get_messages_to_reindex(
        self,
        storage: IStorage,
        force: bool,
        checkpoint: Optional[Dict]
    ) -> List[Dict]:
        """Get list of messages that need re-indexing."""
        import sqlite3

        db_path = self.config['components']['storage']['config']['database']
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()

        if force:
            # All messages
            query = """
                SELECT m.id, m.conversation_id, mc.vectorizable_content
                FROM messages m
                JOIN message_content mc ON m.id = mc.message_id
                ORDER BY m.created_at
            """
        else:
            # Only messages without new format (Intención: + Términos:)
            query = """
                SELECT m.id, m.conversation_id, mc.vectorizable_content
                FROM messages m
                JOIN message_content mc ON m.id = mc.message_id
                WHERE mc.refined_content IS NULL
                   OR mc.refined_content = ''
                   OR NOT (mc.refined_content LIKE 'Intención:%' AND mc.refined_content LIKE '%Términos:%')
                ORDER BY m.created_at
            """

        cur.execute(query)
        messages = [dict(row) for row in cur.fetchall()]
        conn.close()

        # Skip already processed if resuming
        if checkpoint and checkpoint.get('last_processed_id'):
            last_id = checkpoint['last_processed_id']
            # Find index of last processed
            for i, msg in enumerate(messages):
                if msg['id'] == last_id:
                    messages = messages[i + 1:]
                    break

        return messages

    def _process_batch(
        self,
        batch: List[Dict],
        refiner: IRefiner,
        vectorizer: IVectorizer,
        storage: IStorage
    ):
        """Process a batch of messages through refine + vectorize."""
        import sqlite3
        import numpy as np

        # Extract texts
        texts = [msg['vectorizable_content'] or '' for msg in batch]

        # Refine
        refinements = refiner.refine_batch(texts)

        # Vectorize refined content
        refined_texts = [r.refined_content for r in refinements]
        vectors = vectorizer.vectorize_batch(refined_texts)

        # Update database
        db_path = self.config['components']['storage']['config']['database']
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()

        for msg, refinement, vector in zip(batch, refinements, vectors):
            msg_id = msg['id']

            # Update message_content
            cur.execute("""
                UPDATE message_content
                SET refined_content = ?
                WHERE message_id = ?
            """, (refinement.refined_content, msg_id))

            # Update messages table with classification from Qwen + refined_content para FTS
            has_question = refinement.metadata.get('has_question', False)
            topics_str = ','.join(refinement.entities[:10]) if refinement.entities else ''

            # Extraer solo Términos expandidos para semantic_keywords (mejor para FTS)
            semantic_keywords = self._extract_semantic_keywords(refinement)

            cur.execute("""
                UPDATE messages
                SET intent = ?, topics = ?, is_question = ?, refined_content = ?
                WHERE id = ?
            """, (refinement.intent or 'unknown', topics_str, has_question, semantic_keywords, msg_id))

            # Update vector
            vector_blob = vector.astype(np.float32).tobytes()
            cur.execute("""
                UPDATE vectors
                SET vector = ?, model_version = 'reindex_v2'
                WHERE entity_id = ? AND entity_type = 'message'
            """, (vector_blob, msg_id))

            # Insert if not exists
            if cur.rowcount == 0:
                cur.execute("""
                    INSERT INTO vectors (entity_id, entity_type, vector, vector_dim, model_version)
                    VALUES (?, 'message', ?, ?, 'reindex_v2')
                """, (msg_id, vector_blob, len(vector)))

        conn.commit()
        conn.close()

    def _rebuild_fts_index(self, storage: IStorage):
        """Rebuild FTS index to include updated refined_content."""
        import sqlite3

        db_path = self.config['components']['storage']['config']['database']
        conn = sqlite3.connect(db_path)

        # Rebuild FTS index from scratch
        conn.execute("DELETE FROM messages_fts")
        conn.execute("""
            INSERT INTO messages_fts(rowid, id, content, clean_content, vectorizable_content, refined_content)
            SELECT rowid, id, content, clean_content, vectorizable_content, refined_content FROM messages
        """)
        conn.commit()
        conn.close()

    def _extract_semantic_keywords(self, refinement) -> str:
        """
        Extrae términos expandidos para FTS.

        Input (refinement.entities):
            ['SaaS (Software as a Service)', 'API (Application Programming Interface)', ...]

        Output:
            'SaaS Software as a Service API Application Programming Interface ...'

        Esto permite que FTS encuentre tanto "SaaS" como "Software as a Service".
        """
        keywords = []

        # Agregar intención
        if refinement.intent:
            keywords.append(refinement.intent)

        # Agregar entidades/términos expandidos
        for entity in (refinement.entities or []):
            # Limpiar paréntesis y separar sigla de expansión
            # 'SaaS (Software as a Service)' → 'SaaS Software as a Service'
            clean = entity.replace('(', ' ').replace(')', ' ').strip()
            keywords.append(clean)

        # Agregar resumen si existe
        if refinement.summary:
            keywords.append(refinement.summary)

        return ' '.join(keywords)

    def _get_storage(self) -> IStorage:
        """Get storage instance."""
        storage_config = self.config['components']['storage']
        return self.registry.create_instance(
            name='storage',
            class_path=storage_config['class'],
            config=storage_config.get('config', {})
        )

    def _get_refiner(self) -> IRefiner:
        """Get refiner instance."""
        refiner_config = self.config['components']['refiner']
        return self.registry.create_instance(
            name='refiner',
            class_path=refiner_config['class'],
            config=refiner_config.get('config', {})
        )

    def _get_vectorizer(self) -> IVectorizer:
        """Get vectorizer instance."""
        vectorizer_config = self.config['components']['vectorizer']
        return self.registry.create_instance(
            name='vectorizer',
            class_path=vectorizer_config['class'],
            config=vectorizer_config.get('config', {})
        )

    def _load_checkpoint(self) -> Optional[Dict]:
        """Load checkpoint from file."""
        if CHECKPOINT_FILE.exists():
            try:
                with open(CHECKPOINT_FILE) as f:
                    return json.load(f)
            except Exception:
                return None
        return None

    def _save_checkpoint(self, data: Dict):
        """Save checkpoint to file."""
        CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(CHECKPOINT_FILE, 'w') as f:
            json.dump(data, f, indent=2)

    def _clear_checkpoint(self):
        """Remove checkpoint file."""
        if CHECKPOINT_FILE.exists():
            CHECKPOINT_FILE.unlink()
            console.print("[dim]Checkpoint cleared[/dim]")

    def get_status(self) -> Dict[str, Any]:
        """Get current re-indexing status."""
        checkpoint = self._load_checkpoint()

        # Count messages needing reindex
        import sqlite3
        db_path = self.config['components']['storage']['config']['database']
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()

        cur.execute("SELECT COUNT(*) FROM messages")
        total = cur.fetchone()[0]

        # Nuevo formato tiene "Intención:" y "Términos:" (con expansión de siglas)
        cur.execute("""
            SELECT COUNT(*) FROM message_content
            WHERE refined_content LIKE 'Intención:%'
              AND refined_content LIKE '%Términos:%'
        """)
        indexed_new = cur.fetchone()[0]

        conn.close()

        return {
            'total_messages': total,
            'indexed_new_format': indexed_new,
            'pending': total - indexed_new,
            'checkpoint': checkpoint,
        }
