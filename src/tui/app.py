# src/tui/app.py
"""Main Textual TUI application."""

from textual.app import App, ComposeResult
from textual.containers import Container, Vertical, Horizontal
from textual.widgets import Header, Footer, Input, Static, ListView, ListItem, Label
from textual.binding import Binding
from textual import events
from rich.text import Text
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.config import ConfigLoader
from src.core.user_config import UserConfig
from src.core.registry import ComponentRegistry
from src.storage.sqlite_vector import SQLiteVectorStorage
from src.vectorizers.sentence_transformer import SentenceTransformerVectorizer
from src import __version__


class SearchResultItem(ListItem):
    """Custom list item for search results."""

    def __init__(self, result, rank: int):
        super().__init__()
        self.result = result
        self.rank = rank


class DRecallApp(App):
    """
    DRecall Interactive Terminal UI.

    Features:
    - Numbered search results [1-5]
    - Keyboard navigation (↑↓, Enter, ESC)
    - Lazy loading for efficiency
    - Slash commands (/connect, /files, @file)
    - State management (ESC returns without losing context)
    """

    CSS = """
    Screen {
        background: $surface;
    }

    #search-input {
        dock: top;
        height: 3;
        border: solid $primary;
        background: $panel;
    }

    #results-container {
        height: 1fr;
        border: solid $accent;
        background: $panel;
    }

    #status-bar {
        dock: bottom;
        height: 1;
        background: $primary;
        color: $text;
    }

    ListItem {
        padding: 1;
        border-bottom: solid $surface-lighten-1;
    }

    ListItem:hover {
        background: $accent 20%;
    }

    ListItem.--highlight {
        background: $accent 40%;
    }
    """

    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit", show=True),
        Binding("escape", "back", "Back", show=True),
        Binding("ctrl+r", "refresh", "Refresh"),
        ("1", "select_1", "Select [1]"),
        ("2", "select_2", "Select [2]"),
        ("3", "select_3", "Select [3]"),
        ("4", "select_4", "Select [4]"),
        ("5", "select_5", "Select [5]"),
    ]

    def __init__(self):
        super().__init__()
        self.user_config = UserConfig.load()
        self.system_config = ConfigLoader.load()
        self.registry = ComponentRegistry()

        # Initialize storage
        storage_config = self.system_config['components']['storage']['config']
        self.storage = SQLiteVectorStorage(config=storage_config)

        # Initialize vectorizer (lazy-loaded for searches)
        self.vectorizer = None

        # State
        self.current_results = []
        self.history_stack = []  # For ESC navigation

    def compose(self) -> ComposeResult:
        """Create UI layout."""
        # Get stats
        stats = self.storage.get_statistics()
        total_messages = stats.get('total_messages', 0)

        # Welcome message from config
        welcome = self.user_config.prompts.welcome.format(
            version=__version__,
            total_messages=total_messages
        )

        yield Header(show_clock=True)
        yield Container(
            Input(
                placeholder="Busca en tu memoria cognitiva... (o usa /comando)",
                id="search-input"
            ),
            Vertical(id="results-container"),
            Static(welcome, id="status-bar"),
        )
        yield Footer()

    def on_mount(self) -> None:
        """Initialize on app start."""
        self.query_one("#search-input").focus()

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle search query submission."""
        query = event.value.strip()

        if not query:
            return

        # Handle slash commands
        if query.startswith('/'):
            await self.handle_slash_command(query)
            return

        # Handle @file references
        if '@' in query:
            await self.handle_file_reference(query)
            return

        # Regular search
        await self.perform_search(query)

    async def perform_search(self, query: str):
        """Perform semantic search and display results."""
        status_bar = self.query_one("#status-bar", Static)
        results_container = self.query_one("#results-container", Vertical)

        # Show loading
        status_bar.update(self.user_config.prompts.loading)

        # Initialize vectorizer if needed
        if self.vectorizer is None:
            vectorizer_config = self.system_config['components']['vectorizer']['config']
            self.vectorizer = SentenceTransformerVectorizer(config=vectorizer_config)
            self.vectorizer.load()

        # Vectorize query
        query_vector = self.vectorizer.vectorize(query)

        # Search
        results = self.storage.search_by_vector(
            query_vector,
            limit=self.user_config.search.default_limit
        )

        # Filter by min similarity
        results = [
            r for r in results
            if r.score >= self.user_config.search.min_similarity
        ]

        self.current_results = results

        # Clear previous results
        results_container.remove_children()

        if not results:
            results_container.mount(
                Static(self.user_config.prompts.no_results)
            )
            status_bar.update(f"No results for: {query}")
            return

        # Display results
        status_bar.update(
            f"{self.user_config.prompts.search_intro} ({len(results)} resultados)"
        )

        for i, result in enumerate(results, 1):
            msg = result.message
            score = result.score

            # Format result item
            preview = msg.clean_content[:200]
            if len(msg.clean_content) > 200:
                preview += "..."

            # Build result text
            result_text = Text()
            result_text.append(f"[{i}] ", style="bold cyan")

            if self.user_config.search.show_scores:
                result_text.append(f"⭐ {score:.3f} ", style="yellow")

            result_text.append(f"- {msg.normalized.metadata.get('conversation_title', 'Untitled')}\n", style="bold")

            if self.user_config.ui.show_metadata:
                result_text.append(
                    f"    {msg.classification.intent} | {msg.normalized.timestamp.strftime('%Y-%m-%d %H:%M')}\n",
                    style="dim"
                )

            result_text.append(f"    {preview}", style="")

            results_container.mount(
                Static(result_text, classes=f"result-item-{i}")
            )

    async def handle_slash_command(self, command: str):
        """Handle slash commands."""
        status_bar = self.query_one("#status-bar", Static)

        if command == "/files":
            # List indexed files
            status_bar.update("Comando /files - Próximamente: listar archivos indexados")

        elif command.startswith("/connect"):
            parts = command.split()
            if len(parts) > 1:
                service = parts[1]
                status_bar.update(f"Comando /connect {service} - Próximamente: OAuth integration")
            else:
                status_bar.update("Uso: /connect <gemini|chatgpt|claude>")

        elif command == "/help":
            await self.show_help()

        else:
            status_bar.update(f"Comando desconocido: {command}. Usa /help para ver comandos disponibles.")

    async def handle_file_reference(self, query: str):
        """Handle @file references in queries."""
        status_bar = self.query_one("#status-bar", Static)

        # Extract @filename
        import re
        file_refs = re.findall(r'@(\S+)', query)

        if file_refs:
            filename = file_refs[0]
            status_bar.update(f"Búsqueda con referencia a archivo: @{filename} - Próximamente")
            # TODO: Search for messages that reference this file

    async def show_help(self):
        """Display help screen."""
        results_container = self.query_one("#results-container", Vertical)
        results_container.remove_children()

        help_text = Text()
        help_text.append("Comandos Disponibles:\n\n", style="bold cyan")
        help_text.append("Búsqueda:\n", style="bold")
        help_text.append("  <texto>               - Búsqueda semántica\n")
        help_text.append("  @archivo.pdf <texto>  - Buscar en contexto de archivo\n\n")
        help_text.append("Comandos Slash:\n", style="bold")
        help_text.append("  /files                - Listar archivos indexados\n")
        help_text.append("  /connect <servicio>   - Conectar a AI externa (gemini, chatgpt)\n")
        help_text.append("  /stats                - Ver estadísticas\n")
        help_text.append("  /help                 - Esta ayuda\n\n")
        help_text.append("Navegación:\n", style="bold")
        help_text.append("  1-5                   - Seleccionar resultado\n")
        help_text.append("  ↑↓                    - Navegar resultados\n")
        help_text.append("  Enter                 - Ver conversación completa\n")
        help_text.append("  ESC                   - Volver atrás\n")
        help_text.append("  Ctrl+C                - Salir\n")

        results_container.mount(Static(help_text))

    def action_back(self):
        """Go back (ESC key)."""
        if self.history_stack:
            # Restore previous state
            previous_state = self.history_stack.pop()
            # TODO: Restore results from state
            status_bar = self.query_one("#status-bar", Static)
            status_bar.update("Volviendo atrás...")
        else:
            # Clear results
            results_container = self.query_one("#results-container", Vertical)
            results_container.remove_children()
            status_bar = self.query_one("#status-bar", Static)
            status_bar.update("Listo para buscar")

    def action_select_1(self):
        """Select result [1]."""
        self._select_result(0)

    def action_select_2(self):
        """Select result [2]."""
        self._select_result(1)

    def action_select_3(self):
        """Select result [3]."""
        self._select_result(2)

    def action_select_4(self):
        """Select result [4]."""
        self._select_result(3)

    def action_select_5(self):
        """Select result [5]."""
        self._select_result(4)

    def _select_result(self, index: int):
        """Select a result by index."""
        if index < len(self.current_results):
            result = self.current_results[index]
            self.view_conversation(result)
        else:
            status_bar = self.query_one("#status-bar", Static)
            status_bar.update(f"Resultado [{index + 1}] no disponible")

    def view_conversation(self, result):
        """View full conversation for selected result."""
        results_container = self.query_one("#results-container", Vertical)
        status_bar = self.query_one("#status-bar", Static)

        # Save current state
        self.history_stack.append({'results': self.current_results})

        # Get full conversation
        conversation_id = result.message.normalized.conversation_id
        messages = self.storage.get_conversation_messages(conversation_id)

        # Clear and show conversation
        results_container.remove_children()

        conv_title = result.message.normalized.metadata.get('conversation_title', 'Untitled')

        header = Text()
        header.append(f"Conversación: {conv_title}\n", style="bold cyan")
        header.append(f"{len(messages)} mensajes | ", style="dim")
        header.append("ESC para volver\n", style="dim italic")
        header.append("─" * 80, style="dim")

        results_container.mount(Static(header))

        # Show messages (lazy load first N)
        preview_limit = min(20, len(messages))

        for msg in messages[:preview_limit]:
            msg_text = Text()

            # Author
            author_style = "bold green" if msg.normalized.author_role == "user" else "bold blue"
            msg_text.append(f"{msg.normalized.author_role}: ", style=author_style)

            # Timestamp
            msg_text.append(
                f"[{msg.normalized.timestamp.strftime('%Y-%m-%d %H:%M')}]\n",
                style="dim"
            )

            # Content
            content = msg.clean_content[:300]
            if len(msg.clean_content) > 300:
                content += "..."
            msg_text.append(content + "\n\n")

            results_container.mount(Static(msg_text))

        if len(messages) > preview_limit:
            results_container.mount(
                Static(f"[dim]... y {len(messages) - preview_limit} mensajes más (scroll para ver)[/dim]")
            )

        status_bar.update(f"Conversación: {conv_title} | Presiona ESC para volver")


def run_interactive():
    """Launch interactive TUI."""
    app = DRecallApp()
    app.run()
