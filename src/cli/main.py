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
    mode: str = typer.Option("vector", "--mode", "-m", help="Search mode: vector, text, or hybrid"),
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Custom config file"),
):
    """
    Search indexed messages.

    Examples:
        drecall search "how to implement authentication"
        drecall search "python async" --limit 20
        drecall search "error handling" --mode text
    """
    try:
        # Load configuration
        cfg = ConfigLoader.load(config)

        # Initialize storage
        storage_config = cfg['components']['storage']['config']
        storage = SQLiteVectorStorage(config=storage_config)

        if mode == "text":
            # Text search
            results = storage.search_by_text(query, limit=limit)

        elif mode == "vector":
            # Vector search - need to vectorize query first
            console.print("[dim]Initializing vectorizer...[/dim]")

            vectorizer_config = cfg['components']['vectorizer']['config']
            vectorizer = SentenceTransformerVectorizer(config=vectorizer_config)
            vectorizer.load()

            query_vector = vectorizer.vectorize(query)
            results = storage.search_by_vector(query_vector, limit=limit)

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
def version():
    """Show version information."""
    from src import __version__
    console.print(f"DRecall version {__version__}")
    console.print("Tech-agnostic cognitive RAG system")


if __name__ == "__main__":
    app()
