"""
Textual-based Interactive TUI for Chat Recall.
"""

from __future__ import annotations

import asyncio
from typing import List, Optional

from rich.text import Text
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import (
    Header,
    Footer,
    Input,
    Static,
    DataTable,
    Markdown,
    Button,
    Label,
)
from textual.screen import ModalScreen
from textual import on

from src.core.config import ConfigLoader
from src.core.registry import ComponentRegistry
from src.storage.sqlite_vector import SQLiteVectorStorage
from src.contracts.vectorizer import IVectorizer
import re
from typing import Tuple

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


class MessageDetailScreen(ModalScreen):
    """Screen to show full conversation details."""

    CSS = """
    MessageDetailScreen {
        align: center middle;
    }

    #dialog {
        grid-size: 2;
        grid-gutter: 1 2;
        grid-rows: 1fr 3;
        padding: 0 1;
        width: 90%;
        height: 90%;
        border: thick $background 80%;
        background: $surface;
    }

    #content {
        column-span: 2;
        height: 1fr;
        width: 1fr;
        content-align: left top;
        overflow-y: auto;
    }

    .message-box {
        padding: 1;
        margin: 1;
        border: solid $primary;
        background: $surface-lighten-1;
        height: auto;
        min-height: 3;
    }

    .message-box Markdown {
        color: $text;
        background: $surface-lighten-1;
    }

    .message-box.highlight {
        border: double $accent;
        background: $primary-darken-2;
    }

    .message-box.highlight Markdown {
        background: $primary-darken-2;
    }

    .author {
        color: $secondary;
        text-style: bold;
        margin-bottom: 1;
    }

    .timestamp {
        color: $text-muted;
        text-align: right;
    }

    #buttons {
        column-span: 2;
        align: center bottom;
        height: auto;
        margin-top: 1;
    }
    """

    BINDINGS = [
        ("escape", "close_dialog", "Close"),
        ("up", "scroll_up", "Scroll Up"),
        ("down", "scroll_down", "Scroll Down"),
        ("pageup", "page_up", "Page Up"),
        ("pagedown", "page_down", "Page Down"),
    ]

    def __init__(self, conversation_id: str, messages: list, target_message_id: str):
        super().__init__()
        self.conversation_id = conversation_id
        self.messages = messages
        self.target_message_id = target_message_id

    def compose(self) -> ComposeResult:
        content_widgets = []
        
        for msg in self.messages:
            is_target = msg.normalized.id == self.target_message_id
            classes = "message-box"
            if is_target:
                classes += " highlight"
            
            header_text = f"{msg.normalized.author_role} | {msg.normalized.timestamp}"
            
            content_widgets.append(
                Container(
                    Label(header_text, classes="author"),
                    Markdown(msg.clean_content),
                    classes=classes,
                    id=f"msg-{msg.normalized.id}"
                )
            )

        yield Container(
            Vertical(
                Label(f"Conversation: {self.conversation_id}", classes="header"),
                Container(*content_widgets, id="content"),
                Horizontal(
                    Button("Close", variant="primary", id="close"),
                    id="buttons",
                ),
            ),
            id="dialog",
        )

    def on_mount(self):
        # Try to scroll to target
        try:
            target_widget = self.query_one(f"#msg-{self.target_message_id}")
            target_widget.scroll_visible(animate=False, top=True)
        except:
            pass

    def action_scroll_up(self):
        self.query_one("#content").scroll_up()

    def action_scroll_down(self):
        self.query_one("#content").scroll_down()
        
    def action_page_up(self):
        self.query_one("#content").scroll_page_up()

    def action_page_down(self):
        self.query_one("#content").scroll_page_down()

    @on(Button.Pressed, "#close")
    def close_dialog(self):
        self.dismiss()


class ChatRecallApp(App):
    """Textual TUI for Chat Recall."""

    CSS = """
    Screen {
        layout: vertical;
    }

    #search-bar {
        dock: top;
        height: 3;
        margin: 1 1;
    }

    #results-table {
        height: 1fr;
        border: solid green;
    }

    .header {
        dock: top;
        width: 100%;
        background: $primary;
        color: $text;
        padding: 1;
    }
    """

    BINDINGS = [
        ("d", "toggle_dark", "Toggle dark mode"),
        ("q", "quit", "Quit"),
        ("j", "cursor_down", "Down"),
        ("k", "cursor_up", "Up"),
    ]

    def __init__(self):
        super().__init__()
        self.storage: Optional[SQLiteVectorStorage] = None
        self.vectorizer: Optional[IVectorizer] = None
        self.registry = ComponentRegistry()
        self.results = []

    def on_mount(self) -> None:
        """Initialize resources."""
        try:
            cfg = ConfigLoader.load()
            storage_cfg = cfg["components"]["storage"]
            self.vectorizer_cfg = cfg["components"]["vectorizer"]

            self.storage = self.registry.create_instance(
                name="storage",
                class_path=storage_cfg["class"],
                config=storage_cfg.get("config", {}),
            )
            
            # Ensure database is initialized
            conn = self.storage._connect()
            table_exists = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='messages'"
            ).fetchone()
            if not table_exists:
                self.storage.initialize()

        except Exception as e:
            self.notify(f"Error initializing: {e}", severity="error")

    def compose(self) -> ComposeResult:
        yield Header()
        yield Container(
            Input(placeholder="Search query...", id="search-bar"),
            DataTable(id="results-table", cursor_type="row"),
        )
        yield Footer()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle search query."""
        raw_query = event.value
        if not raw_query:
            return

        self.run_worker(self.perform_search(raw_query), exclusive=True)

    async def perform_search(self, raw_query: str):
        """Execute search in background."""
        table = self.query_one(DataTable)
        table.clear(columns=True)
        # Add columns: Score, Topic (HCS), Context (Intent + Keywords), Content
        table.add_columns("Score", "Topic", "Context", "Content")
        
        # Parse topics
        query, topics = _parse_topic_query(raw_query)
        if topics:
            self.notify(f"Contexto: {', '.join(topics)}")
        
        try:
            # Lazy load vectorizer
            if not self.vectorizer:
                self.notify("Loading vectorizer...", title="System")
                self.vectorizer = self.registry.create_instance(
                    name="vectorizer",
                    class_path=self.vectorizer_cfg["class"],
                    config=self.vectorizer_cfg.get("config", {}),
                )
                self.vectorizer.load()

            # Vectorize query
            vec = self.vectorizer.vectorize(query)
            
            # Search
            self.results = self.storage.search_by_vector(vec, limit=20)
            
            if not self.results:
                self.notify("No results found", severity="warning")
                return

            for result in self.results:
                msg = result.message
                snippet = msg.clean_content[:100].replace("\n", " ") + "..."
                
                # Extract metadata
                segment_topic = msg.normalized.metadata.get('segment_topic', '-')
                keywords = ", ".join(msg.classification.topics[:3]) if msg.classification.topics else "-"
                intent = msg.classification.intent
                
                # Format Context column: Intent (dim) + Keywords
                context = Text.assemble(
                    (f"[{intent}]", "bold dim"),
                    " ",
                    (keywords, "italic")
                )
                
                table.add_row(
                    f"{result.score:.3f}",
                    segment_topic,
                    context,
                    snippet,
                    key=msg.normalized.id
                )
            
            table.focus()

        except Exception as e:
            self.notify(f"Search error: {e}", severity="error")

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """Show details when row selected."""
        row_key = event.row_key.value
        # Find result
        result = next((r for r in self.results if r.message.normalized.id == row_key), None)
        if result:
            msg = result.message
            conversation_id = msg.normalized.conversation_id
            
            # Fetch full conversation
            try:
                messages = self.storage.get_conversation_messages(conversation_id)
                self.push_screen(MessageDetailScreen(
                    conversation_id=conversation_id,
                    messages=messages,
                    target_message_id=msg.normalized.id
                ))
            except Exception as e:
                self.notify(f"Error loading conversation: {e}", severity="error")

    def action_cursor_down(self):
        self.query_one(DataTable).action_cursor_down()

    def action_cursor_up(self):
        self.query_one(DataTable).action_cursor_up()


def run_tui():
    app = ChatRecallApp()
    app.run()
