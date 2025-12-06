#!/usr/bin/env python3
"""
TUI (Text User Interface) Prototype for geminIndex

This file demonstrates a TUI approach with a menu-driven interface,
replacing the current Click CLI commands with an interactive terminal UI.
The structure includes project management, state tracking, and workflow guidance.
"""

from pathlib import Path

import click
from rich.console import Console
from rich.prompt import Confirm
from rich.prompt import IntPrompt
from rich.prompt import Prompt
from rich.table import Table
from tinydb import Query
from tinydb import TinyDB

console = Console()

# Define database path for project tracking
DB_PATH = Path("data/projects.json")

# Ensure data directory exists
Path("data").mkdir(exist_ok=True)


class ProjectManager:
    """Manages project state and tracking using TinyDB"""

    def __init__(self):
        self.db = TinyDB(DB_PATH)
        self.query = Query()

    def list_projects(self) -> list[dict]:
        """List all tracked projects"""
        return self.db.all()

    def get_project_by_path(self, project_path: str) -> dict | None:
        """Get a project by its path"""
        return self.db.get(self.query.project_path == project_path)

    def add_project(self, project_path: str, data_dir: str) -> int:
        """Add a new project to tracking"""
        project = {
            "project_path": project_path,
            "data_directory": data_dir,
            "indexed_at": None,
            "status": "created",
            "pipeline_state": {"parsed": False, "preprocessed": False, "embedded": False, "indexed": False},
        }
        return self.db.insert(project)

    def update_project_status(self, project_id: int, status: str):
        """Update project status"""
        self.db.update({"status": status}, self.query.id == project_id)

    def update_pipeline_state(self, project_id: int, step: str, completed: bool):
        """Update pipeline state for a project"""
        project = self.db.get(self.query.id == project_id)
        if project:
            pipeline_state = project.get("pipeline_state", {})
            pipeline_state[step] = completed
            self.db.update({"pipeline_state": pipeline_state}, self.query.id == project_id)


class TUIApp:
    """Main TUI Application Class"""

    def __init__(self):
        self.console = console
        self.project_manager = ProjectManager()
        self.current_project = None

    def display_menu(self):
        """Display the main menu"""
        table = Table(title="geminIndex - TUI Interface")
        table.add_column("Option", style="cyan", no_wrap=True)
        table.add_column("Description", style="magenta")

        table.add_row("1", "New Project - Index a new codebase")
        table.add_row("2", "Continue Project - Work with an existing project")
        table.add_row("3", "List Projects - View all indexed projects")
        table.add_row("4", "Pipeline Status - Check indexing progress")
        table.add_row("5", "Run Full Pipeline - Execute entire indexing process")
        table.add_row("6", "Query Project - Ask questions about your code")
        table.add_row("7", "Advanced RAG - Hybrid search with reranking")
        table.add_row("8", "Exit")

        self.console.print(table)

    def list_projects(self):
        """Display all tracked projects"""
        projects = self.project_manager.list_projects()

        if not projects:
            self.console.print("[yellow]No projects found. Start by indexing a new codebase![/yellow]")
            return

        table = Table(title="Tracked Projects")
        table.add_column("ID", style="cyan", no_wrap=True)
        table.add_column("Project Path", style="magenta")
        table.add_column("Data Directory", style="green")
        table.add_column("Status", style="yellow")
        table.add_column("Indexed At", style="blue")

        for project in projects:
            table.add_row(
                str(project.doc_id),
                project["project_path"],
                project["data_directory"],
                project["status"],
                project.get("indexed_at", "N/A"),
            )

        self.console.print(table)

    def select_project(self) -> dict | None:
        """Interactive project selection"""
        projects = self.project_manager.list_projects()

        if not projects:
            self.console.print("[yellow]No projects available. Create a new project first.[/yellow]")
            return None

        table = Table(title="Select a Project")
        table.add_column("ID", style="cyan", no_wrap=True)
        table.add_column("Project Path", style="magenta")
        table.add_column("Status", style="yellow")

        for project in projects:
            table.add_row(str(project.doc_id), project["project_path"], project["status"])

        self.console.print(table)

        while True:
            try:
                project_id = IntPrompt.ask("Enter project ID", choices=[str(p.doc_id) for p in projects])
                selected_project = self.db.get(self.query.id == project_id)
                if selected_project:
                    return selected_project
                self.console.print("[red]Invalid project ID. Please try again.[/red]")
            except ValueError:
                self.console.print("[red]Invalid input. Please enter a valid ID.[/red]")

    def create_project(self):
        """Create a new project"""
        self.console.print("[bold blue]Creating New Project[/bold blue]")

        # Get project path
        project_path = Prompt.ask("Enter path to project to index", default=".")
        project_path = Path(project_path).resolve()

        if not project_path.exists():
            self.console.print(f"[red]Project path does not exist: {project_path}[/red]")
            return

        # Check if project already exists
        existing_project = self.project_manager.get_project_by_path(str(project_path))
        if existing_project:
            self.console.print(f"[yellow]Project already exists: {project_path}[/yellow]")
            if Confirm.ask("Do you want to continue with this project?"):
                self.current_project = existing_project
                return
            return

        # Create data directory
        data_dir_name = f"data_{project_path.name}_{len(self.project_manager.list_projects())}"
        data_dir = Path("data") / data_dir_name
        data_dir.mkdir(exist_ok=True)

        # Add to project tracker
        project_id = self.project_manager.add_project(str(project_path), str(data_dir))
        self.current_project = self.project_manager.db.get(self.query.id == project_id)

        self.console.print(f"[green]Project created:[/green] {project_path}")
        self.console.print(f"[green]Data directory:[/green] {data_dir}")

    def run_ingestion_pipeline(self):
        """Run the ingestion pipeline"""
        if not self.current_project:
            self.console.print("[red]No project selected. Please select a project first.[/red]")
            return

        self.console.print(
            f"[bold blue]Running Ingestion Pipeline for:[/bold blue] {self.current_project['project_path']}"
        )

        # Implement ingestion logic here (without execution)
        # This is where parse_project would be called
        self.console.print("[cyan]Parsing project...[/cyan]")
        # parser = parse_project(self.current_project['project_path'])
        # Save results to project's data directory
        # Update pipeline state to mark as parsed

        self.project_manager.update_pipeline_state(self.current_project.doc_id, "parsed", True)
        self.console.print("[green]✅ Parsing complete![/green]")

    def run_preprocessing(self):
        """Run preprocessing pipeline"""
        if not self.current_project:
            self.console.print("[red]No project selected. Please select a project first.[/red]")
            return

        self.console.print(f"[bold blue]Running Preprocessing for:[/bold blue] {self.current_project['project_path']}")

        # Implement preprocessing logic here (without execution)
        # This is where ChunkPreprocessor would be used
        self.console.print("[cyan]Preprocessing chunks...[/cyan]")
        # preprocessor = ChunkPreprocessor()
        # processed = preprocessor.process(chunks)
        # Save results

        self.project_manager.update_pipeline_state(self.current_project.doc_id, "preprocessed", True)
        self.console.print("[green]✅ Preprocessing complete![/green]")

    def run_embedding(self):
        """Run embedding pipeline"""
        if not self.current_project:
            self.console.print("[red]No project selected. Please select a project first.[/red]")
            return

        self.console.print(f"[bold blue]Running Embedding for:[/bold blue] {self.current_project['project_path']}")

        # Implement embedding logic here (without execution)
        # This is where EmbeddingGenerator would be used
        self.console.print("[cyan]Generating embeddings...[/cyan]")
        # embedder = EmbeddingGenerator(config)
        # embedded_chunks = embedder.generate_all(chunks)
        # Save results

        self.project_manager.update_pipeline_state(self.current_project.doc_id, "embedded", True)
        self.console.print("[green]✅ Embedding complete![/green]")

    def run_indexing(self):
        """Run indexing pipeline"""
        if not self.current_project:
            self.console.print("[red]No project selected. Please select a project first.[/red]")
            return

        self.console.print(f"[bold blue]Running Indexing for:[/bold blue] {self.current_project['project_path']}")

        # Implement indexing logic here (without execution)
        # This is where QdrantIndexer would be used
        self.console.print("[cyan]Indexing in Qdrant...[/cyan]")
        # indexer = QdrantIndexer(config)
        # indexer.index_chunks(chunks)

        self.project_manager.update_pipeline_state(self.current_project.doc_id, "indexed", True)
        self.console.print("[green]✅ Indexing complete![/green]")

    def run_full_pipeline(self):
        """Run the complete pipeline"""
        if not self.current_project:
            self.console.print("[red]No project selected. Please select a project first.[/red]")
            return

        pipeline_steps = [
            ("parsed", self.run_ingestion_pipeline, "Parsing"),
            ("preprocessed", self.run_preprocessing, "Preprocessing"),
            ("embedded", self.run_embedding, "Embedding"),
            ("indexed", self.run_indexing, "Indexing"),
        ]

        for state_key, step_func, step_name in pipeline_steps:
            if not self.current_project["pipeline_state"].get(state_key, False):
                self.console.print(f"[yellow]Starting {step_name} step...[/yellow]")
                step_func()
            else:
                self.console.print(f"[blue]✓ {step_name} already completed[/blue]")

    def show_pipeline_status(self):
        """Show current pipeline status"""
        if not self.current_project:
            self.console.print("[red]No project selected. Please select a project first.[/red]")
            return

        state = self.current_project["pipeline_state"]
        table = Table(title=f"Pipeline Status: {self.current_project['project_path']}")
        table.add_column("Step", style="cyan")
        table.add_column("Status", style="magenta")

        steps = [
            ("Parse Project", "parsed"),
            ("Preprocess", "preprocessed"),
            ("Embed", "embedded"),
            ("Index", "indexed"),
        ]

        for step_name, step_key in steps:
            status = "✅ Completed" if state.get(step_key, False) else "⏳ Pending"
            table.add_row(step_name, status)

        self.console.print(table)

    def run_query_interface(self):
        """Run interactive query interface"""
        if not self.current_project:
            self.console.print("[red]No project selected. Please select a project first.[/red]")
            return

        self.console.print(f"[bold blue]Query Interface for:[/bold blue] {self.current_project['project_path']}")
        self.console.print("Type 'quit' or 'exit' to go back to the main menu")

        # Implement query interface here (without execution)
        # This is where CodeRAG_2 would be used
        """
        rag_system = CodeRAG_2(...)
        while True:
            query = input("Your question about the codebase: ")
            if not query or query.lower() in {"quit", "exit"}:
                break
            answer = rag_system.query_codebase(query)
            console.print(Markdown(answer))
        """

        self.console.print("[yellow]Query interface would run here...[/yellow]")

    def run_advanced_rag(self):
        """Run advanced RAG interface"""
        if not self.current_project:
            self.console.print("[red]No project selected. Please select a project first.[/red]")
            return

        self.console.print(f"[bold blue]Advanced RAG for:[/bold blue] {self.current_project['project_path']}")
        self.console.print("Type 'quit' or 'exit' to go back to the main menu")

        # Implement advanced RAG interface here (without execution)
        # This is where CompleteRetrievalSystem would be used
        """
        retrieval_system = CompleteRetrievalSystem(...)
        while True:
            query = input("Your question about the codebase: ")
            if not query or query.lower() in {"quit", "exit"}:
                break
            results = retrieval_system.retrieve(query=query, collection_name=collection_name, top_k=5)
            # Display results
        """

        self.console.print("[yellow]Advanced RAG interface would run here...[/yellow]")

    def run(self):
        """Main application loop"""
        self.console.print("[bold green]Welcome to geminIndex - TUI Mode![/bold green]")

        while True:
            self.display_menu()

            try:
                choice = Prompt.ask("Select option", choices=["1", "2", "3", "4", "5", "6", "7", "8"], default="1")

                if choice == "1":
                    # New Project
                    self.create_project()

                elif choice == "2":
                    # Continue Project
                    self.current_project = self.select_project()
                    if self.current_project:
                        self.console.print(f"[green]Selected project: {self.current_project['project_path']}[/green]")

                elif choice == "3":
                    # List Projects
                    self.list_projects()

                elif choice == "4":
                    # Pipeline Status
                    if not self.current_project:
                        self.current_project = self.select_project()
                    self.show_pipeline_status()

                elif choice == "5":
                    # Run Full Pipeline
                    if not self.current_project:
                        self.current_project = self.select_project()
                    if self.current_project:
                        self.run_full_pipeline()

                elif choice == "6":
                    # Query Project
                    if not self.current_project:
                        self.current_project = self.select_project()
                    if self.current_project:
                        self.run_query_interface()

                elif choice == "7":
                    # Advanced RAG
                    if not self.current_project:
                        self.current_project = self.select_project()
                    if self.current_project:
                        self.run_advanced_rag()

                elif choice == "8":
                    # Exit
                    self.console.print("[bold blue]Thank you for using geminIndex![/bold blue]")
                    break

            except KeyboardInterrupt:
                self.console.print("\n[red]Goodbye![/red]")
                break
            except Exception as e:
                self.console.print(f"[red]Error: {e}[/red]")


@click.command()
def main():
    """TUI Interface for geminIndex - Project Management and Pipelines"""
    app = TUIApp()
    app.run()


if __name__ == "__main__":
    main()
