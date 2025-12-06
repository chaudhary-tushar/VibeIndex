#!/usr/bin/env python3
"""
Test script to verify LLM connection using the LLMConfig ping method
"""

import asyncio
import sys

from rich.console import Console
from rich.table import Table

from src.config import LLMConfig

console = Console()


def test_llm_connection():
    """Test the LLM connection using the ping method"""
    console.print("[bold blue]Testing LLM Connection[/bold blue]")

    try:
        # Create LLMConfig instance (loads from settings/.env)
        llm_config = LLMConfig()

        console.print(f"[green]Model URL:[/green] {llm_config.model_url}")
        console.print(f"[green]Model Name:[/green] {llm_config.model_name}")

        console.print("\n[cyan]Testing connection...[/cyan]")

        # Test the connection using ping method
        is_connected = llm_config.ping()

        if is_connected:
            console.print("[bold green]✅ Connection successful![/bold green]")
            console.print("[green]LLM service is reachable and responding[/green]")
        else:
            console.print("[bold red]❌ Connection failed![/bold red]")
            console.print("[red]LLM service is not reachable or not responding properly[/red]")

        return is_connected

    except Exception as e:
        console.print(f"[bold red]❌ Error during connection test: {e}[/bold red]")
        return False


def show_config_details():
    """Display detailed configuration information"""
    console.print("\n[bold yellow]Configuration Details:[/bold yellow]")

    try:
        llm_config = LLMConfig()

        table = Table(title="LLM Configuration")
        table.add_column("Parameter", style="cyan", no_wrap=True)
        table.add_column("Value", style="magenta")

        table.add_row("Model URL", llm_config.model_url)
        table.add_row("Model Name", llm_config.model_name)
        table.add_row("Timeout", str(llm_config.timeout))
        table.add_row("Max Retries", str(llm_config.max_retries))

        console.print(table)

    except Exception as e:
        console.print(f"[red]Error loading configuration: {e}[/red]")


async def main():
    """Main function to run the tests"""
    console.print("[bold green]LLM Connection Test Utility[/bold green]")
    console.print("=" * 50)

    # Show configuration details
    show_config_details()

    # Test connection
    success = test_llm_connection()

    console.print("\n" + "=" * 50)
    if success:
        console.print("[bold green]Test completed successfully![/bold green]")
        return 0
    console.print("[bold red]Test failed![/bold red]")
    return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
