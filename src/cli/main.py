# src/cli/main.py
"""DRecall CLI application."""

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
    name="drecall",
    help="Tech-agnostic cognitive RAG system for document and chat history",
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
    mode: str = typer.Option("vector", "--mode", "-m", help="Search mode: vector, text, topic"),
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Custom config file"),
):
    """
    Search indexed messages.

    Search Modes:
        vector - Semantic search (default, hybrid with keyword boost)
        text   - Full-text keyword search
        topic  - Search by topic tags

    Special Syntax:
        ..topico..  - Auto-detect topic search (e.g., ..energía..)
        @file.pdf   - Search in file context

    Examples:
        drecall search "how to implement authentication"
        drecall search "python async" --limit 20
        drecall search "..energía.." --mode topic
        drecall search "error handling" --mode text
    """
    try:
        # Load configuration
        cfg = ConfigLoader.load(config)

        # Initialize storage
        storage_config = cfg['components']['storage']['config']
        storage = SQLiteVectorStorage(config=storage_config)

        # Auto-detect topic search syntax: ..topic..
        import re
        topic_match = re.search(r'\.\.([^.]+)\.\.', query)

        if topic_match:
            topic_filter = topic_match.group(1)
            # Remove topic syntax from query
            query_without_topic = re.sub(r'\.\.([^.]+)\.\.', '', query).strip()

            if query_without_topic:
                # Mixed search: semantic + topic filter
                console.print(f"[dim]Búsqueda mixta: '{query_without_topic}' en tópico '{topic_filter}'[/dim]")

                # Get vectorizer
                vectorizer_config = cfg['components']['vectorizer']['config']
                vectorizer = SentenceTransformerVectorizer(config=vectorizer_config)
                vectorizer.load()

                query_vector = vectorizer.vectorize(query_without_topic)

                # Get topic results first
                topic_results = storage.search_by_topic(topic_filter, limit=1000)

                # Rank by semantic similarity
                import numpy as np
                for result in topic_results:
                    if result.message.vector is not None:
                        similarity = float(np.dot(query_vector, result.message.vector))
                        result.score = (result.score * 0.3) + (similarity * 0.7)
                        result.matched_on = 'mixed'

                # Sort and limit
                topic_results.sort(key=lambda x: x.score, reverse=True)
                results = topic_results[:limit]

                vectorizer.unload()

            else:
                # Pure topic search
                console.print(f"[dim]Buscando tópico: {topic_filter}[/dim]")
                results = storage.search_by_topic(topic_filter, limit=limit)

            # Show matched topics in results
            for result in results:
                matched_topics = result.metadata.get('matched_topics', [])
                if matched_topics:
                    result.snippet = f"Tópicos: {', '.join(matched_topics)}"

        elif mode == "topic":
            # Explicit topic mode
            console.print(f"[dim]Buscando tópico: {query}[/dim]")
            results = storage.search_by_topic(query, limit=limit)

        elif mode == "text":
            # Text search
            results = storage.search_by_text(query, limit=limit)

        elif mode == "vector":
            # Hybrid search (vector + keyword boosting)
            console.print("[dim]Initializing vectorizer...[/dim]")

            vectorizer_config = cfg['components']['vectorizer']['config']
            vectorizer = SentenceTransformerVectorizer(config=vectorizer_config)
            vectorizer.load()

            query_vector = vectorizer.vectorize(query)

            # Hybrid search: vector similarity + keyword boosting
            results = storage.search_by_vector(
                query_vector,
                limit=limit,
                query_text=query,  # Enable hybrid boosting
                hybrid_boost=0.3  # 30% boost for exact keyword matches
            )

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

            # Show matched topics for topic search
            if result.matched_on == 'topic' and result.metadata.get('matched_topics'):
                matched = ', '.join(result.metadata['matched_topics'])
                all_topics = ', '.join(result.metadata.get('all_topics', []))
                console.print(f"   Matched Topics: [yellow]{matched}[/yellow]")
                console.print(f"   All Topics: [dim]{all_topics}[/dim]")

            console.print(f"   Conversation: {msg.normalized.metadata.get('conversation_title', msg.normalized.conversation_id[:20])}")
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
    - REPL-style search interface
    - Numbered results [1-5] with keyboard navigation
    - Slash commands (/connect, /files, @file)
    - State management (ESC returns without losing context)
    - Lazy loading for efficiency

    Examples:
        drecall interactive
    """
    from src.tui.app import run_interactive
    run_interactive()


@app.command()
def rebuild(
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Custom config file"),
    profile: Optional[str] = typer.Option(None, "--profile", "-p", help="Config profile"),
    confirm: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation"),
    vectors_only: bool = typer.Option(False, "--vectors-only", "-v", help="Only re-vectorize, skip classification"),
    resume: bool = typer.Option(True, "--resume/--no-resume", help="Resume from checkpoint (skip already converted)"),
):
    """
    Rebuild vectors and/or classifications without reimporting.

    Features:
        --resume (default): Detects already-converted vectors by dimension and skips them
        --no-resume: Force re-vectorize all messages

    Examples:
        drecall rebuild --vectors-only --yes
        drecall rebuild --vectors-only --yes --no-resume
        drecall rebuild --profile nomic-384 --yes
    """
    try:
        # Confirmation
        if not confirm:
            total = typer.confirm(
                "⚠️  Esto borrará vectores y reclasificará TODOS los mensajes. ¿Continuar?",
                default=False
            )
            if not total:
                console.print("[yellow]Operación cancelada[/yellow]")
                return

        console.print("\n[bold cyan]🔄 Rebuild: Re-clasificando y re-vectorizando...[/bold cyan]\n")

        # Load config
        if profile:
            profile_path = Path(f"config/{profile}.yaml")
            cfg = ConfigLoader.load(profile_path)
            console.print(f"[dim]Usando profile: {profile}[/dim]")
        else:
            cfg = ConfigLoader.load(config)

        # Initialize storage
        storage_config = cfg['components']['storage']['config']
        storage = SQLiteVectorStorage(config=storage_config)

        # Get all messages
        console.print("Cargando mensajes desde DB...")
        import sqlite3
        conn = sqlite3.connect(storage_config['database'])

        cursor = conn.execute("SELECT COUNT(*) FROM messages")
        total_messages = cursor.fetchone()[0]

        console.print(f"Total mensajes: {total_messages}\n")

        from src.core.registry import ComponentRegistry
        registry = ComponentRegistry()

        # Step 1: Re-classify (skip if --vectors-only)
        if not vectors_only:
            classifier_config = cfg['components']['classifier']
            classifier = registry.create_instance(
                name='classifier',
                class_path=classifier_config['class'],
                config=classifier_config.get('config', {})
            )

            console.print("[bold]Step 1/2: Re-clasificando...[/bold]")
            classifier.load()

            cursor = conn.execute("SELECT id, vectorizable_content FROM messages")
            batch_size = classifier_config.get('config', {}).get('batch_size', 32)
            rows = cursor.fetchall()

            from rich.progress import Progress
            with Progress() as progress:
                task = progress.add_task("Clasificando...", total=len(rows))
                updates = []

                for i in range(0, len(rows), batch_size):
                    batch = rows[i:i+batch_size]
                    texts = [row[1] for row in batch]
                    classifications = classifier.classify_batch(texts)

                    for (msg_id, _), classification in zip(batch, classifications):
                        import json
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
            console.print(f"[green]✓[/green] {len(updates)} mensajes reclasificados\n")
            step_prefix = "Step 2/2"
        else:
            console.print("[dim]Saltando clasificación (--vectors-only)[/dim]\n")
            step_prefix = "Step 1/1"

        # Step 2: Re-vectorize
        vectorizer_config = cfg['components']['vectorizer']
        vectorizer = registry.create_instance(
            name='vectorizer',
            class_path=vectorizer_config['class'],
            config=vectorizer_config.get('config', {})
        )

        console.print(f"[bold]{step_prefix}: Re-vectorizando...[/bold]")
        vectorizer.load()

        target_dim = vectorizer.get_embedding_dim()
        batch_size = vectorizer_config.get('config', {}).get('batch_size', 64)
        import pickle

        # Get messages to process (with checkpoint support)
        if resume:
            console.print(f"[dim]Checkpoint: buscando mensajes sin vector de {target_dim} dims...[/dim]")
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
                console.print(f"[green]✓[/green] {already_done} mensajes ya tienen vectores de {target_dim}d (saltados)")

            if not rows:
                console.print("[green]✓[/green] Todos los mensajes ya tienen vectores correctos")
                vectorizer.unload()
                conn.close()
                return
        else:
            cursor = conn.execute("SELECT id, vectorizable_content FROM messages")
            rows = cursor.fetchall()

        total = len(rows)
        console.print(f"Mensajes a vectorizar: {total}")

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
            console.print(f"  Vectorizado: {processed}/{total} ({100*processed/total:.0f}%)")

        vectorizer.unload()

        console.print(f"[green]✓[/green] {total} vectores regenerados\n")

        # Step 3: Stats
        console.print("[bold]Step 3/3: Verificando...[/bold]")
        stats = storage.get_statistics()

        console.print(f"\n[bold green]✓ Rebuild completado[/bold green]")
        console.print(f"  Mensajes: {stats['total_messages']}")
        console.print(f"  Conversaciones: {stats['total_conversations']}")

        conn.close()

    except Exception as e:
        console.print(f"\n[bold red]✗ Error:[/bold red] {e}")
        raise typer.Exit(1)


@app.command()
def version():
    """Show version information."""
    from src import __version__
    console.print(f"DRecall version {__version__}")
    console.print("Tech-agnostic cognitive RAG system")


if __name__ == "__main__":
    app()
