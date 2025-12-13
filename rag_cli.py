import click
from rich.console import Console
from rich.markdown import Markdown
from textual.app import App
from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.containers import Vertical
from textual.reactive import reactive
from textual.widgets import Button
from textual.widgets import Input
from textual.widgets import Static

from src.config import settings
from src.retrieval import CodeRAG

console = Console()


class AnswerContainer(Static):
    """Widget to display the answer with copy button."""

    answer = reactive("")

    def render(self) -> Markdown:
        return Markdown(self.answer)

    def compose(self) -> ComposeResult:
        yield Static(self.answer, id="answer-content")
        yield Button("Copy Answer", id="copy-button")


class RAGApp(App):
    """Interactive RAG query interface."""

    CSS = """
    Screen {
        background: $background;
    }

    #main-container {
        layout: horizontal;
        height: 1fr;
        background: $background;
    }

    #answer-panel {
        width: 1fr;
        height: 1fr;
        background: $surface;
        layout: vertical;
        border: solid $primary;
        margin: 1;
        position: relative;
    }

    #answer-content {
        height: 1fr;
        width: 1fr;
        margin: 1;
        overflow-y: auto;
        background: $surface;
    }

    #input-container {
        height: 50%;
        width: 1fr;
        layout: vertical;
        background: $surface;
        margin: 1;
        dock: bottom;
    }

    #query-input {
        height: auto;
        min-height: 3;
        margin: 1;
    }

    #copy-button {
        dock: bottom;
        margin: 1;
        offset-x: 1;
        background: $primary;
        color: $text;
        layer: above;
    }
    """

    BINDINGS = [
        ("ctrl+c", "quit", "Quit"),
    ]

    def __init__(self, rag_system: CodeRAG):
        super().__init__()
        self.rag_system = rag_system
        self.current_answer = ""

    def compose(self) -> ComposeResult:
        with Horizontal():
            # Answer panel on the left (full height)
            with Vertical(id="answer-panel"):
                yield Static("🤖 Answer", id="answer-header")
                yield Static(id="answer-content")
                yield Button("Copy Answer", id="copy-button")

            # Input at the bottom taking 50% height (on the right)
            with Vertical(id="input-container"):
                yield Input(placeholder="❓ Your question about the codebase...", id="query-input")

    def on_mount(self) -> None:
        # Set focus to input
        self.query_one("#query-input", Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle query submission."""
        if event.input.id == "query-input":
            query = event.value.strip()
            if not query:
                return

            # Clear input
            event.input.value = ""

            # Update answer content with loading message
            answer_content = self.query_one("#answer-content", Static)
            answer_content.update("🔍 Searching codebase...")

            # Run query in background
            self.call_later(self.process_query, query)

    async def process_query(self, query: str) -> None:
        """Process the query in the background and update UI."""
        try:
            # Get answer from RAG system
            answer = self.rag_system.query_codebase(query)

            # Update answer content
            answer_content = self.query_one("#answer-content", Static)
            answer_content.update(Markdown(answer))
            self.current_answer = answer

        except Exception as e:
            answer_content = self.query_one("#answer-content", Static)
            answer_content.update(f"Error: {e!s}")
            self.current_answer = f"Error: {e!s}"

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "copy-button":
            # Copy current answer to clipboard
            self.copy_to_clipboard(self.current_answer)

    def copy_to_clipboard(self, text: str) -> None:
        """Copy text to clipboard."""
        try:
            # Try to use pyperclip first
            import pyperclip

            pyperclip.copy(text)
            # Show notification
            self.notify("Copied to clipboard!", timeout=1.0)
        except ImportError:
            # Fallback: print message
            self.notify("pyperclip not installed - can't copy to clipboard", timeout=2.0, severity="warning")


@click.group()
def cli():
    """RAG Code Parser CLI"""


@cli.command()
@click.option("--project", "-p", required=True, help="Input JSON file with chunks")
def rag(project):
    """
    Interactive RAG query CLI using CodeRAG
    """
    settings.initialize_project(project)
    print("here")
    try:
        print("here")
        rag_system = CodeRAG()
        print("here")
        console.print("[bold green]CodeRAG Interactive CLI Ready![/bold green]")
        print("here")
    except Exception as e:
        print("here in error")
        console.print(f"[red]Error initializing RAG system: {e}[/red]")
        return

    app = RAGApp(rag_system)
    print("here after app definition")
    try:
        print("here in running app")
        app.run()
    except Exception as e:
        print("here in running app exception")
        console.print(f"[red]Error running application: {e}[/red]")


if __name__ == "__main__":
    cli()
