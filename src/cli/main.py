# src/cli/main.py
"""Chat Recall (crec) - Deep Research Chat Recall CLI."""

# Set CUDA memory config BEFORE importing torch (via sentence_transformers)
import os
os.environ.setdefault('PYTORCH_ALLOC_CONF', 'expandable_segments:True')

from pathlib import Path
from typing import Optional
import sys

import typer
from rich.console import Console
from rich.table import Table

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.config import ConfigLoader
from src.core.registry import ComponentRegistry
from src.core.pipeline import Pipeline
from src.storage.sqlite_vector import SQLiteVectorStorage
from src.vectorizers.sentence_transformer import SentenceTransformerVectorizer

app = typer.Typer(
    name="crec",
    help="Chat Recall - Deep Research Chat Recall",
    add_completion=False,
)
console = Console()


@app.command()
def init(
    source: Path = typer.Argument(..., help="Path to source file (ZIP, directory, etc.)"),
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Custom config file"),
    profile: Optional[str] = typer.Option(None, "--profile", "-p", help="Config profile: default, lite (rule-based), express (no classifier)"),
    cpu_only: bool = typer.Option(False, "--cpu-only", help="Force CPU processing (no CUDA)"),
    batch_size: Optional[int] = typer.Option(None, "--batch-size", help="Override batch size"),
):
    """
    Process and index a data source.

    Profiles:
        default - Full AI (transformer classifier + vectorizer)
        lite    - Rule-based classifier (fast, no GPU for classifier)
        express - Vectorization only (ultra-fast, add classifier later with reindex)

    Examples:
        drecall init data.zip
        drecall init data.zip --profile lite
        drecall init data.zip --profile express
        drecall init data.zip --cpu-only
    """
    if not source.exists():
        console.print(f"[red]Error:[/red] Source not found: {source}")
        raise typer.Exit(1)

    try:
        # Load configuration based on profile
        if profile:
            profile_path = Path(f"config/{profile}.yaml")
            if not profile_path.exists():
                console.print(f"[red]Error:[/red] Profile '{profile}' not found")
                console.print("Available profiles: default, lite, express")
                raise typer.Exit(1)
            cfg = ConfigLoader.load(profile_path)
            console.print(f"[dim]Using profile: {profile}[/dim]")
        else:
            cfg = ConfigLoader.load(config)

        # Apply CLI overrides
        if cpu_only:
            cfg['components']['classifier']['config']['device'] = 'cpu'
            cfg['components']['vectorizer']['config']['device'] = 'cpu'

        if batch_size:
            if 'classifier' in cfg['components']:
                cfg['components']['classifier']['config']['batch_size'] = batch_size
            if 'vectorizer' in cfg['components']:
                cfg['components']['vectorizer']['config']['batch_size'] = batch_size

        # Create registry and pipeline
        registry = ComponentRegistry()
        pipeline = Pipeline(cfg, registry)

        # Process source
        stats = pipeline.process_source(source)

        # Show final stats
        console.print("\n[bold green]✓ Processing complete![/bold green]")

    except Exception as e:
        console.print(f"\n[bold red]✗ Error:[/bold red] {e}")
        raise typer.Exit(1)


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query"),
    limit: int = typer.Option(10, "--limit", "-n", help="Number of results"),
    mode: str = typer.Option("vector", "--mode", "-m", help="Search mode: vector, text, or hybrid"),
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Custom config file"),
    topic: Optional[str] = typer.Option(None, "--topic", "-t", help="Filter by segment topic"),
):
    """
    Search indexed messages.

    Examples:
        drecall search "how to implement authentication"
        drecall search "python async" --limit 20
        drecall search "error handling" --mode text
        drecall search "architecture" --topic "Salud Visual"
    """
    try:
        # Load configuration
        cfg = ConfigLoader.load(config)

        # Initialize storage
        storage_config = cfg['components']['storage']['config']
        storage = SQLiteVectorStorage(config=storage_config)

        # Build filters
        filters = {}
        if topic:
            filters['segment_topic'] = topic

        if mode == "text":
            # Text search
            results = storage.search_by_text(query, limit=limit, filters=filters)

        elif mode == "vector":
            # Vector search - need to vectorize query first
            console.print("[dim]Initializing vectorizer...[/dim]")

            vectorizer_config = cfg['components']['vectorizer']['config']
            vectorizer = SentenceTransformerVectorizer(config=vectorizer_config)
            vectorizer.load()

            query_vector = vectorizer.vectorize(query)
            results = storage.search_by_vector(query_vector, limit=limit, filters=filters)

            vectorizer.unload()

        else:
            console.print(f"[red]Invalid mode:[/red] {mode}. Use 'vector' or 'text'")
            raise typer.Exit(1)

        # Display results
        if not results:
            console.print("[yellow]No results found.[/yellow]")
            return

        console.print(f"\n[bold]Found {len(results)} results:[/bold]\n")

        for i, result in enumerate(results, 1):
            msg = result.message
            score = result.score

            console.print(f"[cyan]{i}. Score: {score:.3f}[/cyan]")
            console.print(f"   Conversation: {msg.normalized.conversation_id}")

            # Show segment topic if available
            segment_topic = msg.normalized.metadata.get('segment_topic')
            if segment_topic:
                console.print(f"   Topic: [bold]{segment_topic}[/bold]")

            console.print(f"   Author: {msg.normalized.author_role}")
            console.print(f"   Intent: {msg.classification.intent} ({msg.classification.confidence:.2f})")
            console.print(f"   Timestamp: {msg.normalized.timestamp}")

            # Show snippet
            snippet = msg.clean_content[:200]
            if len(msg.clean_content) > 200:
                snippet += "..."
            console.print(f"   Content: {snippet}")
            console.print()

    except Exception as e:
        console.print(f"\n[bold red]✗ Error:[/bold red] {e}")
        raise typer.Exit(1)


@app.command()
def stats(
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Custom config file"),
):
    """Show database statistics."""
    try:
        # Load configuration
        cfg = ConfigLoader.load(config)

        # Initialize storage
        storage_config = cfg['components']['storage']['config']
        storage = SQLiteVectorStorage(config=storage_config)

        stats = storage.get_statistics()

        # Create table
        table = Table(title="Database Statistics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Total Messages", str(stats['total_messages']))
        table.add_row("Total Conversations", str(stats['total_conversations']))
        table.add_row("Database Size", f"{stats['size_mb']:.2f} MB")

        if 'intent_distribution' in stats:
            table.add_row("", "")  # Separator
            table.add_row("[bold]Intent Distribution[/bold]", "")
            for intent, count in stats['intent_distribution'].items():
                percentage = (count / stats['total_messages'] * 100) if stats['total_messages'] > 0 else 0
                table.add_row(f"  {intent}", f"{count} ({percentage:.1f}%)")

        console.print(table)

    except Exception as e:
        console.print(f"\n[bold red]✗ Error:[/bold red] {e}")
        raise typer.Exit(1)


@app.command()
def reindex(
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Custom config file"),
    limit: Optional[int] = typer.Option(None, "--limit", help="Max messages to reindex (for testing)"),
    force: bool = typer.Option(False, "--force", help="Re-vectorize all messages, not just outdated"),
):
    """
    Re-vectorize existing messages without reimporting.

    Use this when you change the vectorizer model in config (e.g., switching from
    MiniLM to e5-large). Messages are NOT reimported, only vectors are regenerated.

    Examples:
        drecall reindex
        drecall reindex --limit 1000  # Test with first 1000 messages
        drecall reindex --force  # Re-vectorize everything
    """
    try:
        console.print("[bold cyan]Re-indexing vectors...[/bold cyan]\n")

        # Load configuration
        cfg = ConfigLoader.load(config)

        # Initialize storage
        storage_config = cfg['components']['storage']['config']
        storage = SQLiteVectorStorage(config=storage_config)

        # Initialize vectorizer
        vectorizer_config = cfg['components']['vectorizer']['config']
        vectorizer = SentenceTransformerVectorizer(config=vectorizer_config)
        vectorizer.load()

        # Generate version ID
        vectorizer_version = f"{vectorizer_config['model']}_dim{vectorizer.get_embedding_dim()}"

        # Register this vectorizer version
        storage.register_vectorizer_version(
            version_id=vectorizer_version,
            model_name=vectorizer_config['model'],
            embedding_dim=vectorizer.get_embedding_dim(),
            config=vectorizer_config
        )

        # Get messages to reindex
        if force:
            console.print("Fetching all messages for re-indexing...")
            messages = storage.get_messages_for_reindex(vectorizer_version=None, limit=limit)
        else:
            console.print(f"Fetching messages with outdated vectors (not {vectorizer_version})...")
            messages = storage.get_messages_for_reindex(vectorizer_version=vectorizer_version, limit=limit)

        if not messages:
            console.print("[yellow]No messages need re-indexing.[/yellow]")
            vectorizer.unload()
            return

        console.print(f"Found {len(messages)} messages to re-vectorize\n")

        # Re-vectorize in batches
        batch_size = vectorizer_config.get('batch_size', 64)
        new_vectors = {}

        from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console
        ) as progress:
            task = progress.add_task("Re-vectorizing...", total=len(messages))

            for i in range(0, len(messages), batch_size):
                batch = messages[i:i + batch_size]

                # Extract vectorizable content
                texts = [msg.vectorizable_content for msg in batch]

                # Generate new vectors
                vectors = vectorizer.vectorize_batch(texts)

                # Store in dict
                for msg, vector in zip(batch, vectors):
                    new_vectors[msg.normalized.id] = vector

                progress.update(task, advance=len(batch))

        # Update database
        console.print("\nUpdating database...")
        updated_count = storage.reindex_vectors(new_vectors, vectorizer_version)

        # Unload vectorizer
        vectorizer.unload()

        console.print(f"\n[bold green]✓ Re-indexed {updated_count} messages[/bold green]")
        console.print(f"  New vectorizer: {vectorizer_version}")

    except Exception as e:
        console.print(f"\n[bold red]✗ Error:[/bold red] {e}")
        raise typer.Exit(1)


@app.command(name="refine-reindex")
def refine_reindex(
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Custom config file"),
    profile: Optional[str] = typer.Option(None, "--profile", "-p", help="Config profile"),
    limit: Optional[int] = typer.Option(None, "--limit", help="Max items to process (for testing)"),
    batch_size: int = typer.Option(8, "--batch-size", help="Batch size for SLM processing"),
    force: bool = typer.Option(False, "--force", "-f", help="Re-refine all (not just unrefined)"),
    chunks_only: bool = typer.Option(False, "--chunks-only", help="Only process chunks (skip messages)"),
    include_chunks: bool = typer.Option(True, "--chunks/--no-chunks", help="Also process message chunks"),
    confirm: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation"),
):
    """
    Re-process existing messages through Semantic Refiner + new Vectorizer.

    This command reads from SQLite (not source files), runs the SLM to extract
    semantic intent, and generates new embeddings. The app can continue serving
    queries with old vectors while this runs in the background.

    Use Cases:
    - Initial refinement after upgrading to semantic pipeline
    - Re-refine with improved SLM prompt
    - Upgrade vectorizer model (e.g., MiniLM → GTE)

    Examples:
        crec refine-reindex --yes
        crec refine-reindex --profile default --limit 100
        crec refine-reindex --force --yes  # Re-refine everything
    """
    import pickle

    try:
        if not confirm:
            total = typer.confirm(
                "This will refine and re-vectorize messages. Continue?",
                default=False
            )
            if not total:
                console.print("[yellow]Cancelled[/yellow]")
                return

        console.print("\n[bold cyan]Semantic Re-indexing[/bold cyan]\n")

        # Load configuration
        if profile:
            profile_path = Path(f"config/{profile}.yaml")
            if not profile_path.exists():
                console.print(f"[red]Error:[/red] Profile '{profile}' not found")
                raise typer.Exit(1)
            cfg = ConfigLoader.load(profile_path)
            console.print(f"[dim]Using profile: {profile}[/dim]")
        else:
            cfg = ConfigLoader.load(config)

        # Check refiner is configured
        if 'refiner' not in cfg.get('components', {}):
            console.print("[red]Error:[/red] No refiner configured in selected profile")
            console.print("Add 'refiner' component to your config or use --profile default")
            raise typer.Exit(1)

        # Initialize storage
        storage_config = cfg['components']['storage']['config']
        storage = SQLiteVectorStorage(config=storage_config)

        # Run migration if needed
        console.print("Checking database schema...")
        storage.migrate_add_refined_content_column()

        # Get messages to process (skip if chunks_only)
        messages = []
        if not chunks_only:
            console.print("\nFetching messages...")
            messages = storage.get_messages_for_refinement(
                limit=limit,
                only_unrefined=not force
            )
            console.print(f"Found {len(messages)} messages to process")
        else:
            console.print("[dim]Skipping messages (--chunks-only)[/dim]")

        # Check if there's work to do
        conn_check = storage._connect()
        chunks_pending = conn_check.execute(
            "SELECT COUNT(*) FROM message_chunks WHERE refined_text IS NULL"
        ).fetchone()[0] if include_chunks or chunks_only else 0

        if not messages and chunks_pending == 0:
            console.print("[yellow]Nothing to refine.[/yellow]")
            return

        console.print(f"Chunks pending: {chunks_pending}\n")

        # Initialize refiner
        registry = ComponentRegistry()
        refiner_config = cfg['components']['refiner']
        refiner = registry.create_instance(
            name='refiner',
            class_path=refiner_config['class'],
            config=refiner_config.get('config', {})
        )

        vectorizer_config = cfg['components']['vectorizer']

        from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
        import pickle

        total_refined = 0
        total_vectorized = 0
        MAX_SLM_INPUT = 6000  # SLM context limit

        # ============================================================
        # PHASE 1: REFINEMENT ONLY (Qwen loaded, GTE unloaded)
        # ============================================================
        console.print("\n[bold cyan]═══ PHASE 1: Semantic Refinement (Qwen) ═══[/bold cyan]")
        refiner.load()

        # 1a. Refine messages
        if messages:
            console.print(f"\nRefining {len(messages)} messages...")
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=console
            ) as progress:
                task = progress.add_task("Messages...", total=len(messages))

                for i in range(0, len(messages), batch_size):
                    batch = messages[i:i + batch_size]

                    texts = []
                    for msg in batch:
                        text = msg.vectorizable_content or ''
                        if len(text) > MAX_SLM_INPUT:
                            text = text[:MAX_SLM_INPUT] + '...'
                        texts.append(text)

                    refinements = refiner.refine_batch(texts)

                    # Save refined_content ONLY (no vector yet)
                    conn = storage._connect()
                    for msg, refinement in zip(batch, refinements):
                        conn.execute("""
                            UPDATE messages SET refined_content = ? WHERE id = ?
                        """, (refinement.refined_content, msg.normalized.id))
                    conn.commit()

                    total_refined += len(batch)
                    progress.update(task, advance=len(batch))

        # 1b. Refine chunks
        if include_chunks or chunks_only:
            conn = storage._connect()
            if force:
                cursor = conn.execute("SELECT chunk_id, chunk_text FROM message_chunks")
            else:
                cursor = conn.execute("SELECT chunk_id, chunk_text FROM message_chunks WHERE refined_text IS NULL")
            chunks_to_process = cursor.fetchall()

            if chunks_to_process:
                console.print(f"\nRefining {len(chunks_to_process)} chunks...")
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    TaskProgressColumn(),
                    console=console
                ) as progress:
                    task = progress.add_task("Chunks...", total=len(chunks_to_process))

                    for i in range(0, len(chunks_to_process), batch_size):
                        batch = chunks_to_process[i:i + batch_size]
                        chunk_ids = [c[0] for c in batch]
                        chunk_texts = [c[1] or '' for c in batch]

                        refinements = refiner.refine_batch(chunk_texts)

                        # Save refined_text ONLY (no vector yet)
                        for chunk_id, refinement in zip(chunk_ids, refinements):
                            conn.execute("""
                                UPDATE message_chunks SET refined_text = ? WHERE chunk_id = ?
                            """, (refinement.refined_content, chunk_id))
                        conn.commit()

                        total_refined += len(batch)
                        progress.update(task, advance=len(batch))

        # Unload Qwen to free VRAM
        refiner.unload()
        console.print(f"[green]✓ Refined {total_refined} items[/green]")

        # ============================================================
        # PHASE 2: VECTORIZATION ONLY (GTE loaded, Qwen unloaded)
        # ============================================================
        console.print("\n[bold cyan]═══ PHASE 2: Vectorization (GTE) ═══[/bold cyan]")

        vectorizer = registry.create_instance(
            name='vectorizer',
            class_path=vectorizer_config['class'],
            config=vectorizer_config.get('config', {})
        )
        vectorizer.load()
        model_version = vectorizer_config['config'].get('model', 'unknown')

        conn = storage._connect()

        # 2a. Vectorize messages with refined_content
        cursor = conn.execute("""
            SELECT m.id, m.refined_content
            FROM messages m
            LEFT JOIN vectors v ON m.id = v.entity_id AND v.entity_type = 'message'
            WHERE m.refined_content IS NOT NULL
            AND (v.vector IS NULL OR v.model_version != ?)
        """, (model_version,))
        messages_to_vec = cursor.fetchall()

        if messages_to_vec:
            console.print(f"\nVectorizing {len(messages_to_vec)} messages...")
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=console
            ) as progress:
                task = progress.add_task("Messages...", total=len(messages_to_vec))

                for i in range(0, len(messages_to_vec), batch_size * 2):  # Larger batches OK for GTE
                    batch = messages_to_vec[i:i + batch_size * 2]
                    msg_ids = [m[0] for m in batch]
                    texts = [m[1] or '' for m in batch]

                    vectors = vectorizer.vectorize_batch(texts)

                    for msg_id, vec in zip(msg_ids, vectors):
                        conn.execute("""
                            INSERT OR REPLACE INTO vectors (entity_id, entity_type, vector, vector_dim, model_version)
                            VALUES (?, 'message', ?, 768, ?)
                        """, (msg_id, pickle.dumps(vec), model_version))
                    conn.commit()

                    total_vectorized += len(batch)
                    progress.update(task, advance=len(batch))

        # 2b. Vectorize chunks with refined_text
        cursor = conn.execute("""
            SELECT mc.chunk_id, mc.refined_text
            FROM message_chunks mc
            LEFT JOIN vectors v ON mc.chunk_id = v.entity_id AND v.entity_type = 'chunk'
            WHERE mc.refined_text IS NOT NULL
            AND (v.vector IS NULL OR v.model_version != ?)
        """, (model_version,))
        chunks_to_vec = cursor.fetchall()

        if chunks_to_vec:
            console.print(f"\nVectorizing {len(chunks_to_vec)} chunks...")
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=console
            ) as progress:
                task = progress.add_task("Chunks...", total=len(chunks_to_vec))

                for i in range(0, len(chunks_to_vec), batch_size * 2):
                    batch = chunks_to_vec[i:i + batch_size * 2]
                    chunk_ids = [c[0] for c in batch]
                    texts = [c[1] or '' for c in batch]

                    vectors = vectorizer.vectorize_batch(texts)

                    for chunk_id, vec in zip(chunk_ids, vectors):
                        conn.execute("""
                            INSERT OR REPLACE INTO vectors (entity_id, entity_type, vector, vector_dim, model_version)
                            VALUES (?, 'chunk', ?, 768, ?)
                        """, (chunk_id, pickle.dumps(vec), model_version))
                    conn.commit()

                    total_vectorized += len(batch)
                    progress.update(task, advance=len(batch))

        vectorizer.unload()
        console.print(f"[green]✓ Vectorized {total_vectorized} items[/green]")

        # Rebuild FTS index to sync with updated content
        console.print("\nRebuilding FTS index...")
        storage.rebuild_fts_index()
        console.print("[green]✓[/green] FTS index rebuilt")

        console.print(f"\n[bold green]═══ COMPLETE ═══[/bold green]")
        console.print(f"  Refined: {total_refined} items")
        console.print(f"  Vectorized: {total_vectorized} items")
        console.print(f"  Refiner: {refiner_config['config'].get('model_path', 'Qwen2.5-3B')}")
        console.print(f"  Vectorizer: {vectorizer_config['config'].get('model', 'GTE')}")

    except Exception as e:
        console.print(f"\n[bold red]✗ Error:[/bold red] {e}")
        import traceback
        traceback.print_exc()
        raise typer.Exit(1)


@app.command()
def migrate(
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Custom config file"),
):
    """
    Run database migrations to add new columns.

    This is safe to run multiple times - it will skip columns that already exist.

    Examples:
        crec migrate
    """
    try:
        console.print("[bold cyan]Running database migrations...[/bold cyan]\n")

        # Load configuration
        cfg = ConfigLoader.load(config)

        # Initialize storage
        storage_config = cfg['components']['storage']['config']
        storage = SQLiteVectorStorage(config=storage_config)

        # Run migrations
        console.print("Adding segmentation columns...")
        storage.migrate_add_segmentation_columns()

        console.print("\nAdding refined_content column...")
        storage.migrate_add_refined_content_column()

        console.print("\n[bold green]✓ Migrations complete[/bold green]")

    except Exception as e:
        console.print(f"\n[bold red]✗ Error:[/bold red] {e}")
        raise typer.Exit(1)


@app.command()
def revectorize(
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Custom config file"),
    batch_size: int = typer.Option(4, "--batch-size", "-b", help="Batch size for vectorization (4 safe for 8GB GPUs with GTE)"),
    limit: Optional[int] = typer.Option(None, "--limit", help="Max items to process"),
    include_chunks: bool = typer.Option(True, "--chunks/--no-chunks", help="Also vectorize chunks"),
    confirm: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation"),
):
    """
    Fast re-vectorization using GTE (no SLM refinement).

    Updates vectors table with new 768-dim embeddings.
    Much faster than refine-reindex (minutes vs hours).

    Examples:
        crec revectorize --yes
        crec revectorize --limit 1000 --yes
        crec revectorize --no-chunks --yes
    """
    import pickle

    try:
        console.print("[bold cyan]Fast Re-vectorization (GTE only)[/bold cyan]\n")

        # Load configuration
        cfg = ConfigLoader.load(config)

        # Initialize storage
        storage_config = cfg['components']['storage']['config']
        storage = SQLiteVectorStorage(config=storage_config)
        conn = storage._connect()

        # Count items to process
        cursor = conn.execute("""
            SELECT COUNT(*) FROM vectors WHERE vector_dim != 768 OR vector_dim IS NULL
        """)
        messages_pending = cursor.fetchone()[0]

        chunks_pending = 0
        if include_chunks:
            cursor = conn.execute("""
                SELECT COUNT(*) FROM message_chunks
                WHERE chunk_id NOT IN (SELECT entity_id FROM vectors WHERE entity_type = 'chunk')
            """)
            chunks_pending = cursor.fetchone()[0]

        total = messages_pending + chunks_pending
        if limit:
            total = min(total, limit)

        console.print(f"Messages pending: {messages_pending:,}")
        if include_chunks:
            console.print(f"Chunks pending: {chunks_pending:,}")
        console.print(f"Total to process: {total:,}")

        if total == 0:
            console.print("\n[green]All items already vectorized with 768-dim![/green]")
            return

        if not confirm:
            console.print("\n[yellow]Use --yes to confirm[/yellow]")
            return

        # CUDA memory management
        import gc
        import torch

        # Clear CUDA cache before loading
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Load vectorizer
        vectorizer_config = cfg['components']['vectorizer']['config']
        vectorizer = SentenceTransformerVectorizer(config=vectorizer_config)
        vectorizer.load()

        model_version = f"{vectorizer_config.get('model', 'unknown')}_dim768"

        from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

        processed = 0

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console
        ) as progress:
            task = progress.add_task("Vectorizing...", total=total)

            # Process messages
            limit_clause = f"LIMIT {limit}" if limit else ""
            cursor = conn.execute(f"""
                SELECT v.entity_id, mc.vectorizable_content
                FROM vectors v
                JOIN message_content mc ON v.entity_id = mc.message_id
                WHERE v.entity_type = 'message' AND (v.vector_dim != 768 OR v.vector_dim IS NULL)
                {limit_clause}
            """)

            batch_ids = []
            batch_texts = []

            MAX_TEXT_LEN = 8000  # Truncate to avoid OOM on very long texts

            for row in cursor:
                batch_ids.append(row[0])
                text = row[1] or ''
                if len(text) > MAX_TEXT_LEN:
                    text = text[:MAX_TEXT_LEN]
                batch_texts.append(text)

                if len(batch_ids) >= batch_size:
                    # Vectorize batch
                    vectors = vectorizer.vectorize_batch(batch_texts)

                    # Update vectors table
                    for entity_id, vec in zip(batch_ids, vectors):
                        vec_blob = pickle.dumps(vec)
                        conn.execute("""
                            UPDATE vectors
                            SET vector = ?, vector_dim = 768, model_version = ?
                            WHERE entity_id = ? AND entity_type = 'message'
                        """, (vec_blob, model_version, entity_id))

                    conn.commit()
                    processed += len(batch_ids)
                    progress.update(task, advance=len(batch_ids))
                    batch_ids = []
                    batch_texts = []

                    # Clear CUDA cache periodically to prevent fragmentation
                    if processed % (batch_size * 10) == 0:
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                    if limit and processed >= limit:
                        break

            # Process remaining batch
            if batch_ids and (not limit or processed < limit):
                vectors = vectorizer.vectorize_batch(batch_texts)
                for entity_id, vec in zip(batch_ids, vectors):
                    vec_blob = pickle.dumps(vec)
                    conn.execute("""
                        UPDATE vectors
                        SET vector = ?, vector_dim = 768, model_version = ?
                        WHERE entity_id = ? AND entity_type = 'message'
                    """, (vec_blob, model_version, entity_id))
                conn.commit()
                processed += len(batch_ids)
                progress.update(task, advance=len(batch_ids))

            # Process chunks if enabled
            if include_chunks and (not limit or processed < limit):
                remaining = (limit - processed) if limit else None
                chunk_limit = f"LIMIT {remaining}" if remaining else ""

                cursor = conn.execute(f"""
                    SELECT chunk_id, chunk_text
                    FROM message_chunks
                    WHERE chunk_id NOT IN (SELECT entity_id FROM vectors WHERE entity_type = 'chunk')
                    {chunk_limit}
                """)

                batch_ids = []
                batch_texts = []

                for row in cursor:
                    batch_ids.append(row[0])
                    batch_texts.append(row[1] or '')

                    if len(batch_ids) >= batch_size:
                        vectors = vectorizer.vectorize_batch(batch_texts)

                        for chunk_id, vec in zip(batch_ids, vectors):
                            vec_blob = pickle.dumps(vec)
                            conn.execute("""
                                INSERT OR REPLACE INTO vectors
                                (entity_id, entity_type, vector, vector_dim, model_version)
                                VALUES (?, 'chunk', ?, 768, ?)
                            """, (chunk_id, vec_blob, model_version))

                        conn.commit()
                        processed += len(batch_ids)
                        progress.update(task, advance=len(batch_ids))
                        batch_ids = []
                        batch_texts = []

                        # Clear CUDA cache periodically
                        if processed % (batch_size * 10) == 0:
                            gc.collect()
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()

                # Remaining chunks
                if batch_ids:
                    vectors = vectorizer.vectorize_batch(batch_texts)
                    for chunk_id, vec in zip(batch_ids, vectors):
                        vec_blob = pickle.dumps(vec)
                        conn.execute("""
                            INSERT OR REPLACE INTO vectors
                            (entity_id, entity_type, vector, vector_dim, model_version)
                            VALUES (?, 'chunk', ?, 768, ?)
                        """, (chunk_id, vec_blob, model_version))
                    conn.commit()
                    processed += len(batch_ids)
                    progress.update(task, advance=len(batch_ids))

        vectorizer.unload()

        console.print(f"\n[bold green]✓ Vectorized {processed:,} items[/bold green]")
        console.print(f"  Model: {vectorizer_config.get('model')}")
        console.print(f"  Dimension: 768")

    except Exception as e:
        console.print(f"\n[bold red]✗ Error:[/bold red] {e}")
        import traceback
        traceback.print_exc()
        raise typer.Exit(1)


@app.command()
def interactive():
    """
    Launch interactive TUI (Terminal User Interface).

    Features:
    - REPL-style search interface with Vector First strategy
    - Numbered results [1-5] with keyboard navigation
    - Slash commands (/stats, /help, /theme)
    - Topic search: ..topic.. or query ..topic..
    - ESC returns without losing context
    """
    from src.tui.tui_app import run_tui
    try:
        run_tui()
    except Exception as e:
        import traceback
        traceback.print_exc()
        console.print(f"[red]Failed to start TUI:[/red] {e}")


@app.command()
def rebuild(
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Custom config file"),
    profile: Optional[str] = typer.Option(None, "--profile", "-p", help="Config profile"),
    confirm: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation"),
    vectors_only: bool = typer.Option(False, "--vectors-only", "-v", help="Only re-vectorize, skip classification"),
    resume: bool = typer.Option(True, "--resume/--no-resume", help="Resume from checkpoint (skip already converted)"),
    segment: bool = typer.Option(False, "--segment", "-s", help="Apply semantic segmentation (Topic Drift detection)"),
    segment_threshold: float = typer.Option(0.5, "--segment-threshold", help="Similarity threshold for topic drift (0.0-1.0)"),
):
    """
    Rebuild vectors and/or classifications without reimporting.

    Features:
        --resume (default): Detects already-converted vectors by dimension and skips them
        --no-resume: Force re-vectorize all messages
        --segment: Apply semantic segmentation to detect topic changes within conversations

    Examples:
        crec rebuild --vectors-only --yes
        crec rebuild --vectors-only --yes --no-resume
        crec rebuild --vectors-only --yes --segment
        crec rebuild --segment --segment-threshold 0.6
    """
    import pickle
    import sqlite3
    try:
        if not confirm:
            total = typer.confirm(
                "This will rebuild vectors/classifications. Continue?",
                default=False
            )
            if not total:
                console.print("[yellow]Cancelled[/yellow]")
                return

        console.print("\n[bold cyan]Rebuild: Re-vectorizing...[/bold cyan]\n")

        if profile:
            profile_path = Path(f"config/{profile}.yaml")
            cfg = ConfigLoader.load(profile_path)
            console.print(f"[dim]Using profile: {profile}[/dim]")
        else:
            cfg = ConfigLoader.load(config)

        storage_config = cfg['components']['storage']['config']
        storage = SQLiteVectorStorage(config=storage_config)

        console.print("Loading messages...")
        conn = sqlite3.connect(storage_config['database'])

        # Check if table exists
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='messages'")
        if not cursor.fetchone():
            console.print("[red]Error: Database is empty (no 'messages' table).[/red]")
            console.print("You must import data first using: [bold]crec init <path_to_zip>[/bold]")
            return

        cursor = conn.execute("SELECT COUNT(*) FROM messages")
        total_messages = cursor.fetchone()[0]
        console.print(f"Total messages: {total_messages}\n")

        registry = ComponentRegistry()

        # Skip classification if --vectors-only
        if not vectors_only:
            classifier_config = cfg['components']['classifier']
            classifier = registry.create_instance(
                name='classifier',
                class_path=classifier_config['class'],
                config=classifier_config.get('config', {})
            )

            console.print("[bold]Step 1/2: Re-classifying...[/bold]")
            classifier.load()

            cursor = conn.execute("SELECT id, vectorizable_content FROM messages")
            batch_size = classifier_config.get('config', {}).get('batch_size', 32)
            rows = cursor.fetchall()

            from rich.progress import Progress
            import json
            with Progress() as progress:
                task = progress.add_task("Classifying...", total=len(rows))
                updates = []

                for i in range(0, len(rows), batch_size):
                    batch = rows[i:i+batch_size]
                    texts = [row[1] for row in batch]
                    classifications = classifier.classify_batch(texts)

                    for (msg_id, _), classification in zip(batch, classifications):
                        updates.append((
                            classification.intent,
                            classification.confidence,
                            json.dumps(classification.topics),
                            int(classification.is_question),
                            int(classification.is_command),
                            msg_id
                        ))
                    progress.update(task, advance=len(batch))

            classifier.unload()

            conn.executemany("""
                UPDATE messages
                SET intent = ?, intent_confidence = ?, topics = ?, is_question = ?, is_command = ?
                WHERE id = ?
            """, updates)
            conn.commit()
            console.print(f"[green]✓[/green] {len(updates)} messages reclassified\n")
            step_prefix = "Step 2/2"
        else:
            console.print("[dim]Skipping classification (--vectors-only)[/dim]\n")
            step_prefix = "Step 1/1"

        # Vectorize
        vectorizer_config = cfg['components']['vectorizer']
        vectorizer = registry.create_instance(
            name='vectorizer',
            class_path=vectorizer_config['class'],
            config=vectorizer_config.get('config', {})
        )

        console.print(f"[bold]{step_prefix}: Re-vectorizing...[/bold]")
        vectorizer.load()

        target_dim = vectorizer.get_embedding_dim()
        batch_size = vectorizer_config.get('config', {}).get('batch_size', 64)

        # Checkpoint support
        if resume:
            console.print(f"[dim]Checkpoint: finding messages without {target_dim}d vectors...[/dim]")
            cursor = conn.execute("SELECT id, vectorizable_content, vector FROM messages")
            all_rows = cursor.fetchall()

            rows = []
            already_done = 0
            for row in all_rows:
                msg_id, content, vector_blob = row
                if vector_blob:
                    try:
                        existing_vector = pickle.loads(vector_blob)
                        if len(existing_vector) == target_dim:
                            already_done += 1
                            continue
                    except:
                        pass
                rows.append((msg_id, content))

            if already_done > 0:
                console.print(f"[green]✓[/green] {already_done} messages already have {target_dim}d vectors (skipped)")

            if not rows:
                console.print("[green]✓[/green] All messages already have correct vectors")
                vectorizer.unload()
                conn.close()
                return
        else:
            cursor = conn.execute("SELECT id, vectorizable_content FROM messages")
            rows = cursor.fetchall()

        total = len(rows)
        console.print(f"Messages to vectorize: {total}")

        processed = 0
        for i in range(0, total, batch_size):
            batch = rows[i:i+batch_size]
            texts = [row[1] for row in batch]
            vectors = vectorizer.vectorize_batch(texts)

            updates = []
            for (msg_id, _), vector in zip(batch, vectors):
                updates.append((pickle.dumps(vector), msg_id))

            conn.executemany("UPDATE messages SET vector = ? WHERE id = ?", updates)
            conn.commit()

            processed += len(batch)
            console.print(f"  Vectorized: {processed}/{total} ({100*processed/total:.0f}%)")

        vectorizer.unload()
        console.print(f"[green]✓[/green] {total} vectors regenerated\n")

        # Semantic Segmentation (Topic Drift Detection)
        if segment:
            from src.core.segmentation import SemanticSegmenter
            from rich.progress import Progress

            console.print(f"\n[bold]Semantic Segmentation (HCS)[/bold]")
            console.print(f"Similarity threshold: {segment_threshold}")

            # Ensure segmentation columns exist (migration)
            storage.migrate_add_segmentation_columns()

            # Get conversations that need segmentation
            conversation_ids = storage.get_conversations_for_segmentation()

            if not conversation_ids:
                console.print("[yellow]No conversations need segmentation[/yellow]")
            else:
                console.print(f"Found {len(conversation_ids)} conversations to segment\n")

                segmenter = SemanticSegmenter(similarity_threshold=segment_threshold)
                total_segments = 0

                with Progress() as progress:
                    task = progress.add_task("Segmenting conversations...", total=len(conversation_ids))

                    for conv_id in conversation_ids:
                        # Get all messages for this conversation
                        conv_messages = storage.get_conversation_messages(conv_id)

                        # Segment the conversation
                        segments = segmenter.segment_conversation(conv_id, conv_messages)

                        # Update database
                        updates = segmenter.get_segment_updates(segments)
                        if updates:
                            storage.update_segments_batch(updates)
                            total_segments += len(segments)

                        progress.update(task, advance=1)

                console.print(f"[green]✓[/green] Created {total_segments} topic segments from {len(conversation_ids)} conversations\n")

                # Show segment distribution
                cursor = conn.execute("""
                    SELECT segment_topic, COUNT(*) as msg_count
                    FROM messages
                    WHERE segment_topic IS NOT NULL
                    GROUP BY segment_topic
                    ORDER BY msg_count DESC
                    LIMIT 10
                """)

                console.print("[bold]Top 10 Topics:[/bold]")
                for row in cursor:
                    console.print(f"  • {row[0]}: {row[1]} messages")
                console.print()

        console.print("[bold green]✓ Rebuild complete[/bold green]")
        conn.close()

    except Exception as e:
        console.print(f"\n[bold red]✗ Error:[/bold red] {e}")
        raise typer.Exit(1)


@app.command()
def topics(
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Custom config file"),
    limit: int = typer.Option(20, "--limit", "-n", help="Number of topics to show"),
):
    """
    List all available topic segments.

    Examples:
        crec topics
        crec topics --limit 50
    """
    try:
        # Load configuration
        cfg = ConfigLoader.load(config)

        # Initialize storage
        storage_config = cfg['components']['storage']['config']
        storage = SQLiteVectorStorage(config=storage_config)

        conn = storage._connect()

        # Get topic distribution
        cursor = conn.execute(f"""
            SELECT
                segment_topic,
                COUNT(*) as message_count,
                COUNT(DISTINCT conversation_id) as conversation_count,
                MIN(timestamp) as first_seen,
                MAX(timestamp) as last_seen
            FROM messages
            WHERE segment_topic IS NOT NULL
            GROUP BY segment_topic
            ORDER BY message_count DESC
            LIMIT ?
        """, (limit,))

        topics_data = cursor.fetchall()

        if not topics_data:
            console.print("[yellow]No topics found. Run 'crec rebuild --segment' first.[/yellow]")
            return

        # Create table
        table = Table(title=f"Topic Segments (Top {len(topics_data)})")
        table.add_column("Topic", style="cyan")
        table.add_column("Messages", style="green", justify="right")
        table.add_column("Conversations", style="yellow", justify="right")
        table.add_column("Date Range", style="dim")

        for row in topics_data:
            topic = row[0]
            msg_count = row[1]
            conv_count = row[2]
            first = row[3][:10] if row[3] else "N/A"
            last = row[4][:10] if row[4] else "N/A"

            date_range = f"{first} to {last}" if first != last else first

            table.add_row(topic, str(msg_count), str(conv_count), date_range)

        console.print(table)

        # Show total stats
        cursor = conn.execute("""
            SELECT
                COUNT(DISTINCT segment_topic) as total_topics,
                COUNT(*) as total_segmented_messages
            FROM messages
            WHERE segment_topic IS NOT NULL
        """)
        stats = cursor.fetchone()

        console.print(f"\n[dim]Total topics: {stats[0]} | Segmented messages: {stats[1]}[/dim]")

    except Exception as e:
        console.print(f"\n[bold red]✗ Error:[/bold red] {e}")
        raise typer.Exit(1)


@app.command()
def version():
    """Show version information."""
    from src import __version__, __full_name__
    console.print(f"{__full_name__} (crec) v{__version__}")
    console.print("Deep Research Chat Recall")


if __name__ == "__main__":
    app()
