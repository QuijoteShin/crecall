# src/tui/app.py
"""Main Textual TUI application - Improved with navigation."""

from textual.app import App, ComposeResult
from textual.containers import Container, ScrollableContainer, Vertical
from textual.widgets import Header, Footer, Input, Static, ListView, ListItem, Label, RichLog
from textual.binding import Binding
from textual.reactive import reactive
from rich.text import Text
from rich.markdown import Markdown
from rich.markup import escape
from pathlib import Path
from datetime import datetime
import sys
import re

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.config import ConfigLoader
from src.core.user_config import UserConfig
from src.core.registry import ComponentRegistry
from src.storage.sqlite_vector import SQLiteVectorStorage
from src import __version__


class SearchResultItem(ListItem):
    """List item for search results."""

    def __init__(self, result, rank: int):
        super().__init__()
        self.result = result
        self.rank = rank
        self.message = result.message

        # Format content
        msg = result.message
        score = result.score

        # Build label using Text object (safer than string markup)
        content = Text()

        # Header line
        title = msg.normalized.metadata.get('conversation_title', 'Untitled')
        content.append(f"[{rank}] ", style="bold cyan")
        content.append(f"⭐ {score:.3f} ", style="yellow")
        content.append(f"- {title}\n")

        # Intent and timestamp
        content.append(f"    {msg.classification.intent}", style="dim")
        content.append(f" | {msg.normalized.timestamp.strftime('%Y-%m-%d %H:%M')}\n", style="dim")

        # Preview content
        preview = msg.clean_content[:500]
        if len(msg.clean_content) > 500:
            preview += "..."
        content.append(f"    {preview}\n")

        # Keywords if hybrid search
        if result.matched_on == 'hybrid' and result.metadata.get('keywords_found'):
            keywords = ', '.join(result.metadata['keywords_found'])
            content.append(f"    Keywords: {keywords}", style="dim")

        self._label = Static(content)

    def compose(self) -> ComposeResult:
        yield self._label


class DRecallApp(App):
    """
    DRecall Interactive Terminal UI - Improved.

    Navigation:
    - Tab: Switch between input and results
    - ↑↓: Navigate results
    - Enter: View full conversation
    - 1-5: Quick select result
    - ESC: Go back
    """

    CSS = """
    #search-input {
        dock: top;
        height: 3;
        border: solid $primary;
        margin: 1;
    }

    #results-list {
        height: 1fr;
        border: solid $accent;
        margin: 1;
    }

    #status-bar {
        dock: bottom;
        height: 1;
        background: $primary;
        color: $text;
        padding: 0 1;
    }

    ListItem {
        padding: 1;
        height: auto;
    }

    ListItem:hover {
        background: $accent 30%;
    }

    ListItem.-active {
        background: $accent 50%;
    }

    .matched-message {
        background: $warning 20%;
        border-left: thick $warning;
    }

    .conversation-view {
        padding: 1;
        border: solid $primary;
    }
    """

    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit", show=True),
        Binding("escape", "back", "Back", show=True),
        Binding("tab", "focus_next", "Next Panel", show=True),
        Binding("shift+tab", "focus_previous", "Prev Panel", show=True),
        Binding("ctrl+f", "find_in_conversation", "Find in Conversation", show=True),
        Binding("f3", "find_next", "Next Match", show=False),  # Navigate to next match
        Binding("shift+f3", "find_previous", "Previous Match", show=False),
        Binding("ctrl+h", "show_help", "Help/Palette", show=True),
        Binding("ctrl+j", "jump_to_match", "Jump to Match", show=True),
        Binding("ctrl+m", "load_more", "Load More", show=True),
        Binding("ctrl+p", "export", "Export", show=True),
        Binding("ctrl+o", "show_options", "Options", show=True),
        ("1", "select_1", "[1]"),
        ("2", "select_2", "[2]"),
        ("3", "select_3", "[3]"),
        ("4", "select_4", "[4]"),
        ("5", "select_5", "[5]"),
        ("n", "find_next_alias", "Next"),  # Alias for F3
        ("N", "find_previous_alias", "Prev"),  # Alias for Shift+F3
    ]

    def __init__(self):
        super().__init__()
        self.user_config = UserConfig.load()
        self.system_config = ConfigLoader.load()
        self.registry = ComponentRegistry()

        # Apply theme from config
        if self.user_config.ui.theme == "light":
            self.theme = "textual-light"
        elif self.user_config.ui.theme == "dark":
            self.theme = "textual-dark"
        # "auto" uses default

        # Initialize storage
        storage_config = self.system_config['components']['storage']['config']
        self.storage = SQLiteVectorStorage(config=storage_config)

        # Vectorizer (lazy-loaded)
        self.vectorizer = None

        # State (not reactive to avoid numpy array comparison issues)
        self.current_results = []
        self.current_offset = 0
        self.history_stack = []
        self.last_query = ""
        self.current_conversation_id = None
        self.matched_message_id = None  # For highlighting
        self.current_conversation_messages = []  # Full conversation cache

        # Query history (like bash history)
        self.query_history = self._load_query_history()  # Load from file
        self.history_index = -1  # Current position in history (-1 = not navigating)
        self.current_draft = ""  # Save current text when navigating history

        # Find in conversation
        self.find_mode = False
        self.find_term = ""
        self.find_matches = []
        self.find_current_index = 0

    def compose(self) -> ComposeResult:
        """Create UI layout."""
        # Get stats
        stats = self.storage.get_statistics()
        total_messages = stats.get('total_messages', 0)

        welcome = self.user_config.prompts.welcome.format(
            version=__version__,
            total_messages=total_messages
        )

        yield Header(show_clock=True)
        yield Input(
            placeholder="Busca en tu memoria cognitiva... (o /help para comandos)",
            id="search-input"
        )
        yield ListView(id="results-list")
        yield Static(welcome, id="status-bar")
        yield Footer()

    def _load_query_history(self) -> list:
        """Load query history from file."""
        history_file = Path(".drecall/search_history.txt")

        if not history_file.exists():
            return []

        try:
            with open(history_file, 'r', encoding='utf-8') as f:
                # Load last 100 queries
                lines = f.readlines()
                return [line.strip() for line in lines if line.strip()][-100:]
        except:
            return []

    def _save_query_history(self):
        """Save query history to file."""
        history_file = Path(".drecall/search_history.txt")
        history_file.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(history_file, 'w', encoding='utf-8') as f:
                # Save last 100 queries
                for query in self.query_history[-100:]:
                    f.write(f"{query}\n")
        except Exception as e:
            pass  # Silent fail

    def on_mount(self) -> None:
        """Initialize on app start."""
        self.query_one("#search-input").focus()

    def on_unmount(self) -> None:
        """Save history on exit."""
        self._save_query_history()

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle search query submission."""
        query = event.value.strip()

        if not query:
            return

        # Handle find in conversation mode
        if self.find_mode:
            await self.perform_find_in_conversation(query)
            self.query_one("#results-list").focus()
            return

        # Add to history (avoid duplicates of last query)
        if not self.query_history or self.query_history[-1] != query:
            self.query_history.append(query)

        # Reset history navigation
        self.history_index = -1
        self.current_draft = ""

        self.last_query = query

        # Handle slash commands
        if query.startswith('/'):
            await self.handle_slash_command(query)
            return

        # Handle topic search syntax: ..topic..
        import re
        topic_match = re.search(r'\.\.([^.]+)\.\.', query)

        if topic_match:
            topic_filter = topic_match.group(1)
            # Remove topic syntax from query
            query_without_topic = re.sub(r'\.\.([^.]+)\.\.', '', query).strip()

            if query_without_topic:
                # Mixed: semantic search + topic filter
                await self.perform_mixed_search(query_without_topic, topic_filter)
            else:
                # Pure topic search
                await self.perform_topic_search(topic_filter)

            self.query_one("#results-list").focus()
            return

        # Handle @file references
        if '@' in query:
            await self.handle_file_reference(query)
            return

        # Regular search
        await self.perform_search(query)

        # Focus results after search
        self.query_one("#results-list").focus()

    async def perform_mixed_search(self, query: str, topic_hint: str):
        """
        Mixed search: VECTOR FIRST, topics as bonus (not filter).

        Strategy: "Vector Truth"
        - Combine query + topic_hint into semantic search
        - Find by meaning, not by tags
        - Add bonus for topic tag matches (highlighting)

        Example: "bintel ..sql.." → Search "bintel sql" semantically
        """
        status_bar = self.query_one("#status-bar", Static)
        results_list = self.query_one("#results-list", ListView)

        # Combine query with topic hint for richer semantic search
        semantic_query = f"{query} {topic_hint}"
        status_bar.update(f"Buscando '{semantic_query}' (vector first)...")

        if self.vectorizer is None:
            vectorizer_cfg = self.system_config['components']['vectorizer']
            registry = ComponentRegistry()
            self.vectorizer = registry.create_instance(
                name='vectorizer',
                class_path=vectorizer_cfg['class'],
                config=vectorizer_cfg.get('config', {})
            )
            self.vectorizer.load()

        # Vectorize the combined semantic query
        if hasattr(self.vectorizer, 'vectorize_query'):
            query_vector = self.vectorizer.vectorize_query(semantic_query)
        else:
            query_vector = self.vectorizer.vectorize(semantic_query)

        # VECTOR FIRST: Search by meaning, with keyword bonus
        results = self.storage.search_by_vector(
            query_vector,
            limit=self.user_config.search.default_limit * 3,
            query_text=semantic_query,
            hybrid_boost=0.3
        )

        # Add bonus for topic tag matches (highlighting, not filtering)
        import numpy as np
        topic_lower = topic_hint.lower()

        for result in results:
            topics = result.message.classification.topics
            topic_bonus = 0.0

            for topic in topics:
                if topic_lower in topic.lower() or topic.lower() in topic_lower:
                    topic_bonus = 0.15
                    break

            if topic_bonus > 0:
                result.score = min(1.0, result.score + topic_bonus)
                result.matched_on = 'mixed'
                result.metadata['topic_bonus'] = topic_bonus
                result.metadata['matched_topics'] = [t for t in topics if topic_lower in t.lower()]

        # Re-sort after topic bonus
        results.sort(key=lambda x: x.score, reverse=True)

        # Filter by min similarity
        results = [
            r for r in results
            if r.score >= self.user_config.search.min_similarity
        ]

        # Limit
        results = results[:self.user_config.search.default_limit * 2]

        # Update ranks
        for i, result in enumerate(results, 1):
            result.rank = i

        self.current_results = results
        results_list.clear()

        if not results:
            status_bar.update(f"No hay resultados para '{semantic_query}'")
            return

        status_bar.update(
            f"'{query}' ..{topic_hint}.. ({len(results)} resultados) | Vector First | Tab=navegar"
        )

        for i, result in enumerate(results, 1):
            results_list.append(SearchResultItem(result, i))

    async def perform_topic_search(self, topic_query: str):
        """
        Topic search: VECTOR FIRST strategy.

        ..sql.. now searches semantically for "sql" meaning,
        not just messages with "sql" in their tags.
        """
        status_bar = self.query_one("#status-bar", Static)
        results_list = self.query_one("#results-list", ListView)

        status_bar.update(f"Buscando ..{topic_query}.. (vector first)...")

        # Load vectorizer
        if self.vectorizer is None:
            vectorizer_cfg = self.system_config['components']['vectorizer']
            registry = ComponentRegistry()
            self.vectorizer = registry.create_instance(
                name='vectorizer',
                class_path=vectorizer_cfg['class'],
                config=vectorizer_cfg.get('config', {})
            )
            self.vectorizer.load()

        # Vectorize the topic as semantic query
        if hasattr(self.vectorizer, 'vectorize_query'):
            query_vector = self.vectorizer.vectorize_query(topic_query)
        else:
            query_vector = self.vectorizer.vectorize(topic_query)

        # VECTOR FIRST: Search by meaning
        results = self.storage.search_by_vector(
            query_vector,
            limit=self.user_config.search.default_limit * 2,
            query_text=topic_query,
            hybrid_boost=0.3
        )

        # Add bonus for topic tag matches
        topic_lower = topic_query.lower()
        for result in results:
            topics = result.message.classification.topics
            for topic in topics:
                if topic_lower in topic.lower() or topic.lower() in topic_lower:
                    result.score = min(1.0, result.score + 0.15)
                    result.metadata['matched_topics'] = [t for t in topics if topic_lower in t.lower()]
                    break

        # Re-sort after bonus
        results.sort(key=lambda x: x.score, reverse=True)

        # Filter by min similarity
        results = [r for r in results if r.score >= self.user_config.search.min_similarity]

        self.current_results = results
        results_list.clear()

        if not results:
            status_bar.update(f"No hay resultados para ..{topic_query}..")
            return

        status_bar.update(
            f"..{topic_query}.. ({len(results)} resultados) | Vector First | Tab=navegar"
        )

        for i, result in enumerate(results, 1):
            results_list.append(SearchResultItem(result, i))

    async def perform_search(self, query: str, offset: int = 0, append: bool = False):
        """
        Perform hybrid search and display results.

        Args:
            query: Search query
            offset: Offset for pagination
            append: If True, append to existing results (infinite scroll)
        """
        status_bar = self.query_one("#status-bar", Static)
        results_list = self.query_one("#results-list", ListView)

        # Show loading
        status_bar.update(self.user_config.prompts.loading)

        if self.vectorizer is None:
            vectorizer_cfg = self.system_config['components']['vectorizer']
            registry = ComponentRegistry()
            self.vectorizer = registry.create_instance(
                name='vectorizer',
                class_path=vectorizer_cfg['class'],
                config=vectorizer_cfg.get('config', {})
            )
            self.vectorizer.load()
            status_bar.update("Vectorizer cargado. Buscando...")

        if hasattr(self.vectorizer, 'vectorize_query'):
            query_vector = self.vectorizer.vectorize_query(query)
        else:
            query_vector = self.vectorizer.vectorize(query)

        # Hybrid search with increased limit for pagination
        limit = self.user_config.search.default_limit
        if append:
            limit = limit + offset
        else:
            limit = limit + offset

        results = self.storage.search_by_vector(
            query_vector,
            limit=limit,
            query_text=query,
            hybrid_boost=0.3
        )

        # Filter by min similarity
        results = [
            r for r in results
            if r.score >= self.user_config.search.min_similarity
        ]

        # For append mode, only show new results
        if append and self.current_results:
            # Get only new results
            existing_ids = {r.message.normalized.id for r in self.current_results}
            new_results = [r for r in results if r.message.normalized.id not in existing_ids]

            if not new_results:
                status_bar.update("No hay más resultados")
                return

            self.current_results.extend(new_results)

            # Append to list
            current_count = len(self.current_results) - len(new_results)
            for i, result in enumerate(new_results, current_count + 1):
                results_list.append(SearchResultItem(result, i))

            status_bar.update(
                f"Cargados {len(new_results)} resultados más | Total: {len(self.current_results)} | Ctrl+M=más"
            )

        else:
            # Fresh search
            self.current_results = results
            self.current_offset = offset

            # Clear previous results
            results_list.clear()

            if not results:
                status_bar.update(self.user_config.prompts.no_results)
                return

            # Display results
            status_bar.update(
                f"{self.user_config.prompts.search_intro} ({len(results)} resultados) | Tab=navegar, Ctrl+M=más, Enter=ver"
            )

            for i, result in enumerate(results, 1):
                results_list.append(SearchResultItem(result, i))

    async def handle_slash_command(self, command: str):
        """Handle slash commands."""
        status_bar = self.query_one("#status-bar", Static)
        results_list = self.query_one("#results-list", ListView)

        # Theme command
        if command.startswith("/theme"):
            parts = command.split()
            if len(parts) == 2 and parts[1] in ['dark', 'light', 'auto']:
                new_theme = parts[1]
                self.user_config.ui.theme = new_theme
                self.user_config.save()  # Save to file

                status_bar.update(f"✓ Tema cambiado a '{new_theme}'. Reinicia el TUI para aplicar (Ctrl+C y relanza)")
            else:
                status_bar.update("Uso: /theme <dark|light|auto>")

        # Limit command
        elif command.startswith("/limit"):
            parts = command.split()
            if len(parts) == 2 and parts[1].isdigit():
                new_limit = int(parts[1])
                self.user_config.search.default_limit = new_limit
                self.user_config.save()

                status_bar.update(f"✓ Límite cambiado a {new_limit}. Aplicado inmediatamente.")
            else:
                status_bar.update("Uso: /limit <número>")

        # Language command
        elif command.startswith("/lang"):
            parts = command.split()
            if len(parts) == 2 and parts[1] in ['es', 'en']:
                new_lang = parts[1]
                self.user_config.personality.language = new_lang
                self.user_config.save()

                status_bar.update(f"✓ Idioma cambiado a '{new_lang}'. Aplicado inmediatamente.")
            else:
                status_bar.update("Uso: /lang <es|en>")

        # Stats command
        elif command == "/stats":
            await self.show_stats()

        # Files command
        elif command == "/files":
            status_bar.update("Comando /files - Próximamente: listar archivos indexados")

        # Connect command
        elif command.startswith("/connect"):
            parts = command.split()
            if len(parts) > 1:
                service = parts[1]
                status_bar.update(f"Comando /connect {service} - Próximamente: OAuth integration")
            else:
                status_bar.update("Uso: /connect <gemini|chatgpt|claude>")

        # Help command
        elif command == "/help":
            await self.show_help()

        else:
            status_bar.update(f"Comando desconocido: {command}. Usa /help")

    async def show_stats(self):
        """Display database statistics."""
        results_list = self.query_one("#results-list", ListView)
        status_bar = self.query_one("#status-bar", Static)

        stats = self.storage.get_statistics()

        results_list.clear()

        # Build stats display
        stats_text = Text()
        stats_text.append("Estadísticas de la Base de Datos\n\n", style="bold cyan")

        stats_text.append(f"Total mensajes: ", style="bold")
        stats_text.append(f"{stats['total_messages']}\n")

        stats_text.append(f"Total conversaciones: ", style="bold")
        stats_text.append(f"{stats['total_conversations']}\n")

        stats_text.append(f"Tamaño: ", style="bold")
        stats_text.append(f"{stats['size_mb']:.2f} MB\n\n")

        if 'intent_distribution' in stats:
            stats_text.append("Distribución de Intenciones:\n", style="bold")
            for intent, count in stats['intent_distribution'].items():
                pct = (count / stats['total_messages'] * 100) if stats['total_messages'] > 0 else 0
                stats_text.append(f"  {intent}: ", style="cyan")
                stats_text.append(f"{count} ({pct:.1f}%)\n")

        # Add as static item
        results_list.append(ListItem(Static(stats_text)))

        status_bar.update("Estadísticas | ESC=volver")

    async def show_help(self):
        """Display help."""
        results_list = self.query_one("#results-list", ListView)
        status_bar = self.query_one("#status-bar", Static)

        results_list.clear()

        help_text = Text()
        help_text.append("DRecall - Comandos Disponibles\n\n", style="bold cyan")

        help_text.append("Búsqueda:\n", style="bold green")
        help_text.append("  <texto>               - Búsqueda semántica híbrida\n")
        help_text.append("  ..tópico..            - Buscar por tópicos (ej: ..energía..)\n")
        help_text.append("  bintel ..sass..       - Mixta: buscar 'bintel' EN tópico 'sass'\n")
        help_text.append("  @archivo.pdf <texto>  - Buscar en contexto de archivo\n\n")

        help_text.append("Comandos:\n", style="bold green")
        help_text.append("  /stats                - Ver estadísticas de la base de datos\n")
        help_text.append("  /theme <dark|light>   - Cambiar tema y guardar\n")
        help_text.append("  /limit <número>       - Cambiar límite de resultados\n")
        help_text.append("  /lang <es|en>         - Cambiar idioma\n")
        help_text.append("  /files                - Listar archivos indexados\n")
        help_text.append("  /connect <servicio>   - Conectar a AI externa\n")
        help_text.append("  /help                 - Esta ayuda\n\n")

        help_text.append("Navegación:\n", style="bold green")
        help_text.append("  Tab / Shift+Tab       - Cambiar entre paneles\n")
        help_text.append("  ↑↓ (en Input)         - Historial de búsquedas (como bash)\n")
        help_text.append("  ↑↓ (en Resultados)    - Navegar lista\n")
        help_text.append("  Enter                 - Ver conversación (auto-scroll a match)\n")
        help_text.append("  1-5                   - Selección rápida de resultado\n")
        help_text.append("  ESC                   - Volver atrás\n\n")

        help_text.append("Acciones:\n", style="bold green")
        help_text.append("  Ctrl+F                - Buscar DENTRO de conversación actual\n")
        help_text.append("    n / F3              - Siguiente ocurrencia\n")
        help_text.append("    N / Shift+F3        - Anterior ocurrencia\n")
        help_text.append("  Ctrl+H                - Mostrar ayuda/palette (este menú)\n")
        help_text.append("  Ctrl+J                - Saltar al mensaje matcheado\n")
        help_text.append("  Ctrl+M                - Cargar más resultados (infinite scroll)\n")
        help_text.append("  Ctrl+P                - Exportar conversación a Markdown\n")
        help_text.append("  Ctrl+O                - Ver/cambiar opciones\n")
        help_text.append("  Ctrl+C                - Salir\n\n")

        help_text.append("Tip: La búsqueda combina similitud semántica + keywords exactos\n", style="dim italic")

        results_list.append(ListItem(Static(help_text)))

        status_bar.update("Ayuda | ESC=volver")

    async def handle_file_reference(self, query: str):
        """Handle @file references."""
        status_bar = self.query_one("#status-bar", Static)

        import re
        file_refs = re.findall(r'@(\S+)', query)

        if file_refs:
            filename = file_refs[0]
            # Remove @filename from query
            clean_query = re.sub(r'@\S+', '', query).strip()

            if clean_query:
                # TODO: Add filter for files
                await self.perform_search(clean_query)
                status_bar.update(f"Búsqueda en contexto de @{filename}")
            else:
                status_bar.update(f"Buscar archivo: @{filename} - Próximamente")

    async def on_list_view_selected(self, event: ListView.Selected) -> None:
        """Handle result selection (Enter key only, not Ctrl+M)."""
        if isinstance(event.item, SearchResultItem):
            self.view_conversation(event.item.result)

    async def on_key(self, event) -> None:
        """Handle key events before widgets process them."""
        # Get currently focused widget
        focused = self.focused

        # History navigation when Input is focused
        if focused and focused.id == "search-input":
            search_input = self.query_one("#search-input", Input)

            if event.key == "up":
                # Navigate backward in history
                if self.query_history:
                    # Save current draft if starting navigation
                    if self.history_index == -1:
                        self.current_draft = search_input.value

                    # Move backward in history
                    if self.history_index < len(self.query_history) - 1:
                        self.history_index += 1
                        search_input.value = self.query_history[-(self.history_index + 1)]
                        search_input.cursor_position = len(search_input.value)
                    elif self.history_index == -1 and self.query_history:
                        # First up press
                        self.history_index = 0
                        search_input.value = self.query_history[-1]
                        search_input.cursor_position = len(search_input.value)

                event.prevent_default()
                event.stop()
                return

            elif event.key == "down":
                # Navigate forward in history
                if self.history_index > 0:
                    self.history_index -= 1
                    search_input.value = self.query_history[-(self.history_index + 1)]
                    search_input.cursor_position = len(search_input.value)
                elif self.history_index == 0:
                    # Return to draft
                    self.history_index = -1
                    search_input.value = self.current_draft
                    search_input.cursor_position = len(search_input.value)

                event.prevent_default()
                event.stop()
                return

        # Intercept Ctrl+M to prevent ListView from handling it
        if event.key == "ctrl+m":
            await self.action_load_more()
            event.prevent_default()
            event.stop()

    def view_conversation(self, result):
        """View full conversation with highlighted match."""
        results_list = self.query_one("#results-list", ListView)
        status_bar = self.query_one("#status-bar", Static)

        # Save current state
        self.history_stack.append({
            'type': 'results',
            'results': self.current_results,
            'query': self.last_query
        })

        # Get full conversation
        conversation_id = result.message.normalized.conversation_id
        self.current_conversation_id = conversation_id
        self.matched_message_id = result.message.normalized.id  # For highlighting

        messages = self.storage.get_conversation_messages(conversation_id)
        self.current_conversation_messages = messages  # Cache for Ctrl+F search

        # Clear and show conversation
        results_list.clear()

        conv_title = result.message.normalized.metadata.get('conversation_title', 'Untitled')

        # Header
        header = Text()
        header.append(f"📖 {conv_title}\n", style="bold cyan")
        header.append(f"{len(messages)} mensajes | ", style="dim")
        header.append(f"Iniciada: {messages[0].normalized.timestamp.strftime('%Y-%m-%d %H:%M')}\n", style="dim")
        header.append("─" * 80 + "\n", style="dim")

        results_list.append(ListItem(Static(header)))

        # Show all messages with highlighting
        matched_index = None

        for idx, msg in enumerate(messages):
            is_matched = msg.normalized.id == self.matched_message_id

            if is_matched:
                matched_index = idx + 1  # +1 because header is index 0

            msg_text = Text()

            # Author with color
            if msg.normalized.author_role == "user":
                msg_text.append("👤 Usuario", style="bold green")
            else:
                msg_text.append("🤖 Assistant", style="bold blue")

            msg_text.append(f" [{msg.normalized.timestamp.strftime('%H:%M')}]", style="dim")

            # Highlight indicator if matched
            if is_matched:
                msg_text.append(" ⬅ MATCHED", style="bold yellow on red")

            msg_text.append(f"\n{msg.clean_content}\n", style="")

            # Create list item with special class for matched message
            item = ListItem(Static(msg_text))
            if is_matched:
                item.add_class("matched-message")

            results_list.append(item)

        # Auto-scroll to matched message
        if matched_index is not None:
            # Focus the matched item
            try:
                results_list.index = matched_index
            except:
                pass

        status_bar.update(f"Conversación: {conv_title} | {len(messages)} msgs | Ctrl+F=buscar | Ctrl+J=match | Ctrl+P=exportar | ESC=volver")

    def action_back(self):
        """Go back (ESC)."""
        # Cancel find mode
        if self.find_mode:
            self.find_mode = False
            search_input = self.query_one("#search-input", Input)
            search_input.placeholder = "Busca en tu memoria cognitiva... (o /help para comandos)"
            search_input.value = ""
            status_bar = self.query_one("#status-bar", Static)
            status_bar.update("Búsqueda en conversación cancelada")
            return

        if not self.history_stack:
            # Clear results
            results_list = self.query_one("#results-list", ListView)
            results_list.clear()
            status_bar = self.query_one("#status-bar", Static)
            status_bar.update("Listo para buscar")
            self.query_one("#search-input").focus()
            return

        # Restore previous state
        previous = self.history_stack.pop()

        if previous['type'] == 'results':
            # Restore search results
            results_list = self.query_one("#results-list", ListView)
            results_list.clear()

            for i, result in enumerate(previous['results'], 1):
                results_list.append(SearchResultItem(result, i))

            self.current_results = previous['results']
            status_bar = self.query_one("#status-bar", Static)
            status_bar.update(f"Resultados: {previous['query']} | Tab=navegar, Enter=ver, ESC=volver")

            results_list.focus()

    def action_select_1(self):
        self._quick_select(0)

    def action_select_2(self):
        self._quick_select(1)

    def action_select_3(self):
        self._quick_select(2)

    def action_select_4(self):
        self._quick_select(3)

    def action_select_5(self):
        self._quick_select(4)

    def _quick_select(self, index: int):
        """Quick select result by number."""
        if index < len(self.current_results):
            self.view_conversation(self.current_results[index])
        else:
            status_bar = self.query_one("#status-bar", Static)
            status_bar.update(f"Resultado [{index + 1}] no disponible")

    async def action_load_more(self):
        """Load more results (Ctrl+M)."""
        status_bar = self.query_one("#status-bar", Static)

        if not self.last_query or self.last_query.startswith('/'):
            status_bar.update("No hay búsqueda activa para cargar más resultados")
            return

        # Increase offset
        self.current_offset += self.user_config.search.default_limit

        # Re-search with append mode
        await self.perform_search(self.last_query, offset=self.current_offset, append=True)

    async def action_export(self):
        """Export conversation (Ctrl+P)."""
        status_bar = self.query_one("#status-bar", Static)

        if not self.current_conversation_id:
            status_bar.update("No hay conversación abierta para exportar")
            return

        # Export conversation to markdown
        messages = self.storage.get_conversation_messages(self.current_conversation_id)

        if not messages:
            status_bar.update("No se pudo cargar la conversación")
            return

        # Build markdown export
        conv_title = messages[0].normalized.metadata.get('conversation_title', 'Untitled')
        filename = f"export_{self.current_conversation_id[:8]}.md"

        from pathlib import Path
        export_dir = Path("data/exports")
        export_dir.mkdir(parents=True, exist_ok=True)

        export_path = export_dir / filename

        with open(export_path, 'w', encoding='utf-8') as f:
            f.write(f"# {conv_title}\n\n")
            f.write(f"Exportado: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
            f.write(f"Mensajes: {len(messages)}\n\n")
            f.write("---\n\n")

            for msg in messages:
                role_label = "Usuario" if msg.normalized.author_role == "user" else "Assistant"
                timestamp = msg.normalized.timestamp.strftime('%Y-%m-%d %H:%M')

                f.write(f"## {role_label} - {timestamp}\n\n")

                # Highlight if matched
                if msg.normalized.id == self.matched_message_id:
                    f.write(f"> **[RESULTADO DE BÚSQUEDA]**\n\n")

                f.write(f"{msg.clean_content}\n\n")
                f.write("---\n\n")

        status_bar.update(f"✓ Exportado: {export_path}")

    async def action_show_help(self):
        """Show help palette (Ctrl+H)."""
        await self.show_help()
        self.query_one("#results-list").focus()

    async def action_show_options(self):
        """Show options panel (Ctrl+O) with interactive selectors."""
        results_list = self.query_one("#results-list", ListView)
        status_bar = self.query_one("#status-bar", Static)

        results_list.clear()

        options_text = Text()
        options_text.append("⚙️  Opciones de Configuración\n\n", style="bold cyan")

        # Current settings
        options_text.append("Configuración Actual:\n", style="bold")
        options_text.append(f"  Tema: ", style="dim")
        options_text.append(f"{self.user_config.ui.theme}\n", style="yellow bold")
        options_text.append(f"  Idioma: {self.user_config.personality.language}\n")
        options_text.append(f"  Límite búsqueda: {self.user_config.search.default_limit}\n")
        options_text.append(f"  Similitud mínima: {self.user_config.search.min_similarity}\n\n")

        options_text.append("Cambiar Opciones (escribe el comando):\n\n", style="bold green")

        options_text.append("  /theme dark          - Cambiar a tema oscuro\n")
        options_text.append("  /theme light         - Cambiar a tema claro\n")
        options_text.append("  /theme auto          - Tema automático\n\n")

        options_text.append("  /limit 10            - Cambiar límite de resultados\n")
        options_text.append("  /lang es             - Cambiar idioma (es, en)\n\n")

        options_text.append("Ubicación del archivo:\n", style="dim")

        # Check where config is
        config_path = Path.home() / ".drecall" / "config.toml"
        if not config_path.exists():
            config_path = Path(".drecall/config.toml")

        options_text.append(f"  {config_path}\n", style="cyan")

        results_list.append(ListItem(Static(options_text)))

        status_bar.update("Opciones | Escribe comando (ej: /theme light) | ESC=volver")

    def action_jump_to_match(self):
        """Jump to matched message in conversation (Ctrl+J)."""
        if not self.matched_message_id or not self.current_conversation_id:
            status_bar = self.query_one("#status-bar", Static)
            status_bar.update("No hay mensaje matcheado. Abre una conversación desde resultados primero.")
            return

        results_list = self.query_one("#results-list", ListView)

        # Find matched message index
        for idx, item in enumerate(results_list.children):
            if isinstance(item, ListItem) and item.has_class("matched-message"):
                results_list.index = idx
                status_bar = self.query_one("#status-bar", Static)
                status_bar.update("Saltado al mensaje matcheado ⬅")
                return

        status_bar = self.query_one("#status-bar", Static)
        status_bar.update("No se encontró el mensaje matcheado en la vista actual")

    async def action_find_in_conversation(self):
        """Find text within current conversation (Ctrl+F)."""
        if not self.current_conversation_id or not self.current_conversation_messages:
            status_bar = self.query_one("#status-bar", Static)
            status_bar.update("Abre una conversación primero para buscar dentro de ella (Ctrl+F)")
            return

        # Switch to find mode
        self.find_mode = True

        # Focus input and change placeholder
        search_input = self.query_one("#search-input", Input)
        search_input.placeholder = f"🔍 Buscar en conversación ({len(self.current_conversation_messages)} mensajes)... ESC=cancelar"
        search_input.value = ""
        search_input.focus()

        status_bar = self.query_one("#status-bar", Static)
        status_bar.update("Modo búsqueda en conversación | Escribe término y Enter | ESC=cancelar")

    async def perform_find_in_conversation(self, find_term: str):
        """Search for term in current conversation."""
        results_list = self.query_one("#results-list", ListView)
        status_bar = self.query_one("#status-bar", Static)

        find_term_lower = find_term.lower()

        # Search in all messages (including not loaded)
        matches = []
        for msg in self.current_conversation_messages:
            if find_term_lower in msg.clean_content.lower():
                matches.append(msg)

        self.find_matches = matches
        self.find_term = find_term

        if not matches:
            status_bar.update(f"'{find_term}' no encontrado en conversación | ESC=volver")
            return

        # Display filtered messages
        results_list.clear()

        conv_title = matches[0].normalized.metadata.get('conversation_title', 'Untitled')

        # Header
        header = Text()
        header.append(f"🔍 Búsqueda: '{find_term}' en {conv_title}\n", style="bold cyan")
        header.append(f"Encontrados: {len(matches)} de {len(self.current_conversation_messages)} mensajes\n", style="dim")
        header.append("─" * 80 + "\n", style="dim")

        results_list.append(ListItem(Static(header)))

        # Show matching messages
        for idx, msg in enumerate(matches):
            msg_text = Text()

            # Author
            if msg.normalized.author_role == "user":
                msg_text.append("👤 Usuario", style="bold green")
            else:
                msg_text.append("🤖 Assistant", style="bold blue")

            msg_text.append(f" [{msg.normalized.timestamp.strftime('%H:%M')}]", style="dim")

            # Highlight search term in content
            content = msg.clean_content
            content_lower = content.lower()

            # Simple highlight (could improve with regex)
            highlighted_content = ""
            last_pos = 0

            for match in re.finditer(re.escape(find_term_lower), content_lower):
                start, end = match.span()
                # Add text before match
                highlighted_content += content[last_pos:start]
                # Add highlighted match
                highlighted_content += f"[bold yellow on red]{content[start:end]}[/bold yellow on red]"
                last_pos = end

            # Add remaining text
            highlighted_content += content[last_pos:]

            msg_text.append(f"\n{highlighted_content[:500]}\n", style="")
            if len(content) > 500:
                msg_text.append("...\n", style="dim")

            results_list.append(ListItem(Static(msg_text)))

        status_bar.update(f"Encontrados {len(matches)} mensajes con '{find_term}' | n=siguiente, N=anterior | ESC=ver todos")

        # Stay in find results (don't exit find mode yet)
        # User can navigate with n/N
        self.find_current_index = 0

        # Focus results for navigation
        results_list.focus()

    def action_find_next(self):
        """Navigate to next find match (F3 or n)."""
        self._navigate_find_matches(1)

    def action_find_next_alias(self):
        """Alias for find_next (n key)."""
        if self.find_matches:  # Only if in find mode
            self._navigate_find_matches(1)

    def action_find_previous(self):
        """Navigate to previous find match (Shift+F3 or N)."""
        self._navigate_find_matches(-1)

    def action_find_previous_alias(self):
        """Alias for find_previous (N key)."""
        if self.find_matches:  # Only if in find mode
            self._navigate_find_matches(-1)

    def _navigate_find_matches(self, direction: int):
        """Navigate through find matches."""
        if not self.find_matches:
            status_bar = self.query_one("#status-bar", Static)
            status_bar.update("No hay búsqueda activa. Usa Ctrl+F para buscar en conversación.")
            return

        # Update index
        self.find_current_index += direction

        # Wrap around
        if self.find_current_index >= len(self.find_matches):
            self.find_current_index = 0
        elif self.find_current_index < 0:
            self.find_current_index = len(self.find_matches) - 1

        # Find the list item index (accounting for header)
        results_list = self.query_one("#results-list", ListView)

        # Jump to match index (+1 for header)
        try:
            results_list.index = self.find_current_index + 1
        except:
            pass

        # Update status
        status_bar = self.query_one("#status-bar", Static)
        status_bar.update(
            f"Match {self.find_current_index + 1}/{len(self.find_matches)} de '{self.find_term}' | "
            f"n=siguiente, N=anterior | ESC=ver todos"
        )


def run_interactive():
    """Launch interactive TUI."""
    app = DRecallApp()
    app.run()
