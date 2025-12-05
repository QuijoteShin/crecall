"""Minimal interactive TUI for Chat Recall."""

from __future__ import annotations

import re
from typing import List, Tuple

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from src.core.config import ConfigLoader
from src.core.registry import ComponentRegistry
from src.contracts.vectorizer import IVectorizer
from src.storage.sqlite_vector import SQLiteVectorStorage

console = Console()


def _parse_topic_query(raw_query: str) -> Tuple[str, List[str]]:
    """
    Extract topics expressed as ..topic.. and return the combined query.

    Example: "bintel ..sql.." -> ("bintel sql", ["sql"])
    """
    topics = re.findall(r"\.\.(.+?)\.\.", raw_query)
    cleaned = raw_query
    for topic in topics:
        cleaned = cleaned.replace(f"..{topic}..", "").strip()

    parts = [cleaned, " ".join(topics)]
    combined = " ".join(part for part in parts if part).strip()
    return (combined or raw_query).strip(), topics


def _format_snippet(text: str, max_length: int = 200) -> str:
    """Compact multi-line content into a single preview line."""
    flattened = " ".join(text.split())
    return flattened[: max_length - 3] + "..." if len(flattened) > max_length else flattened


def _print_help(current_limit: int, mode: str) -> None:
    console.print(
        Panel(
            "\n".join(
                [
                    "Comandos:",
                    "  /help            Mostrar esta ayuda",
                    "  /stats           Estadisticas de la base",
                    "  /mode vector|text Cambiar modo de busqueda",
                    "  /limit N         Cambiar limite de resultados",
                    "  /exit            Salir",
                    "",
                    "Sintaxis:",
                    "  query            Busqueda semantica (vector first)",
                    "  ..tema..         Pista de tema, se mezcla en el query",
                    "",
                    f"Modo actual: {mode} | Limite: {current_limit}",
                ]
            ),
            title="Ayuda",
            expand=False,
        )
    )


def _print_stats(storage: SQLiteVectorStorage) -> None:
    stats = storage.get_statistics()
    table = Table(title="Base de datos", header_style="bold cyan")
    table.add_column("Metric", style="cyan")
    table.add_column("Valor", style="green")

    table.add_row("Mensajes", str(stats.get("total_messages", 0)))
    table.add_row("Conversaciones", str(stats.get("total_conversations", 0)))
    size_mb = stats.get("size_mb", 0)
    table.add_row("Tamanio", f"{size_mb:.2f} MB")

    intent_dist = stats.get("intent_distribution") or {}
    if intent_dist:
        table.add_row("", "")
        table.add_row("Intents", "")
        for intent, count in intent_dist.items():
            table.add_row(f"  {intent}", str(count))

    console.print(table)


def _print_results(results) -> None:
    if not results:
        console.print("[yellow]No se encontraron resultados[/yellow]")
        return

    table = Table(
        show_header=True,
        header_style="bold cyan",
        box=None,
        title="Resultados",
    )
    table.add_column("#", style="dim", width=4)
    table.add_column("Score", style="cyan", width=8)
    table.add_column("Intent", style="yellow", width=12)
    table.add_column("Autor", style="green", width=10)
    table.add_column("Conv", style="magenta", width=12)
    table.add_column("Tema", style="blue", width=14)
    table.add_column("Snippet", style="white")

    for result in results:
        msg = result.message
        metadata = msg.normalized.metadata or {}
        segment_topic = metadata.get("segment_topic") or msg.classification.topics[:1]
        topic_label = (
            segment_topic if isinstance(segment_topic, str) else ", ".join(segment_topic)
        )
        snippet = _format_snippet(msg.clean_content)
        table.add_row(
            str(result.rank or ""),
            f"{result.score:.3f}",
            msg.classification.intent,
            msg.normalized.author_role,
            msg.normalized.conversation_id,
            topic_label or "-",
            snippet,
        )

    console.print(table)


def run_interactive() -> None:
    """Entry point for `python crec.py interactive`."""
    console.print(
        Panel(
            "\n".join(
                [
                    "Chat Recall - modo interactivo",
                    "Escribe un query y presiona Enter.",
                    "Comandos: /help, /stats, /mode vector|text, /limit N, /exit",
                    "Tips: usa ..tema.. para dar contexto semantico (vector first).",
                ]
            ),
            title="crec",
            expand=False,
        )
    )

    try:
        cfg = ConfigLoader.load()
    except Exception as exc:
        console.print(f"[red]Error cargando config:[/red] {exc}")
        return

    registry = ComponentRegistry()
    storage_cfg = cfg["components"]["storage"]
    vectorizer_cfg = cfg["components"]["vectorizer"]

    try:
        storage: SQLiteVectorStorage = registry.create_instance(
            name="storage",
            class_path=storage_cfg["class"],
            config=storage_cfg.get("config", {}),
        )
        conn = storage._connect()
        table_exists = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='messages'"
        ).fetchone()
        if not table_exists:
            storage.initialize()
    except Exception as exc:
        console.print(f"[red]No se pudo iniciar el storage:[/red] {exc}")
        return

    vectorizer: IVectorizer | None = None
    mode = "vector"
    limit = 5

    # Show quick stats up front
    try:
        stats = storage.get_statistics()
        console.print(
            f"[dim]{stats.get('total_messages', 0)} mensajes | "
            f"{stats.get('total_conversations', 0)} conversaciones[/dim]"
        )
    except Exception:
        console.print("[yellow]Advertencia: no pude leer estadisticas de la base[/yellow]")

    def ensure_vectorizer_loaded() -> IVectorizer:
        nonlocal vectorizer
        if vectorizer is None:
            vec_instance = registry.create_instance(
                name="vectorizer",
                class_path=vectorizer_cfg["class"],
                config=vectorizer_cfg.get("config", {}),
            )
            vec_instance.load()
            vectorizer = vec_instance
        return vectorizer

    try:
        while True:
            try:
                raw_query = Prompt.ask("[bold magenta]buscar[/bold magenta]").strip()
            except (KeyboardInterrupt, EOFError):
                console.print("\n[dim]Saliendo...[/dim]")
                break

            if not raw_query:
                continue

            lowered = raw_query.lower()
            if lowered in {"exit", "/exit", "quit", "/quit", ":q"}:
                break

            if raw_query.startswith("/"):
                if lowered.startswith("/help"):
                    _print_help(limit, mode)
                elif lowered.startswith("/stats"):
                    _print_stats(storage)
                elif lowered.startswith("/limit"):
                    parts = raw_query.split()
                    if len(parts) == 2 and parts[1].isdigit():
                        limit = max(1, min(50, int(parts[1])))
                        console.print(f"[green]Limite actualizado a {limit}[/green]")
                    else:
                        console.print("[yellow]Uso: /limit 10[/yellow]")
                elif lowered.startswith("/mode"):
                    parts = raw_query.split()
                    if len(parts) == 2 and parts[1] in {"vector", "text"}:
                        mode = parts[1]
                        console.print(f"[green]Modo: {mode}[/green]")
                    else:
                        console.print("[yellow]Modos validos: vector, text[/yellow]")
                else:
                    console.print("[yellow]Comando no reconocido. Usa /help[/yellow]")
                continue

            query, topics = _parse_topic_query(raw_query)
            if topics:
                console.print(f"[dim]Contexto de tema: {', '.join(topics)}[/dim]")

            try:
                if mode == "text":
                    results = storage.search_by_text(query, limit=limit)
                else:
                    vec = ensure_vectorizer_loaded().vectorize(query)
                    results = storage.search_by_vector(vec, limit=limit)
            except Exception as exc:
                console.print(f"[red]Error en la busqueda:[/red] {exc}")
                continue

            _print_results(results)
    finally:
        if vectorizer is not None:
            try:
                vectorizer.unload()
            except Exception:
                pass
        try:
            storage.close()
        except Exception:
            pass
