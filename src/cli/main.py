# src/cli/main.py
"""Chat Recall (crec) - Deep Research Chat Recall CLI."""

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
