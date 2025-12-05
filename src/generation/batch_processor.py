"""
BatchProcessor_2 for sequential batch processing of prompts through LLMClient.
Adapted from old_code/batch_processor.py with enhancements for dirs, formats, etc.
Compatible with src/generation/ module.
"""

import csv
import json
import time
from pathlib import Path

from rich.console import Console
from rich.progress import track

from .generator import LLMClient


class BatchProcessorError(Exception):
    """Custom exception for BatchProcessor errors."""


class BatchProcessor:
    """Batch processor class for prompts."""

    def __init__(self, delay: float = 0.2):
        self.delay = delay
        self.console = Console()
        self.client = LLMClient()

    def load_prompts(self, input_path: str) -> list[str]:
        """Load prompts from file or directory of .txt files."""
        path = Path(input_path).resolve()
        if not path.exists():
            error_message = f"Input path not found: {input_path}"
            self.console.print(f"[red]❌ {error_message}[/red]")
            raise FileNotFoundError(error_message)

        prompts: list[str] = []
        if path.is_file() and path.suffix.lower() == ".txt":
            with Path(path).open("r", encoding="utf-8") as f:
                file_prompts = [line.strip() for line in f if line.strip()]
                prompts.extend(file_prompts)
                self.console.print(f"[green]✅ Loaded {len(file_prompts)} prompts from {path.name}[/green]")
        elif path.is_dir():
            txt_files = list(path.rglob("*.txt"))
            if not txt_files:
                self.console.print("[yellow]⚠️ No .txt files found in directory.[/yellow]")
                return prompts
            for txt_file in sorted(txt_files):
                with Path(txt_file).open("r", encoding="utf-8") as f:
                    file_prompts = [line.strip() for line in f if line.strip()]
                    prompts.extend(file_prompts)
                    self.console.print(f"[green]✅ Loaded {len(file_prompts)} from {txt_file.name}[/green]")
            self.console.print(f"[bold green]📊 Total: {len(prompts)} prompts[/bold green]")
        else:
            error_message = f"❌ Unsupported input: {input_path} (use .txt file or dir)"
            raise ValueError(error_message)
        return prompts

    def process_prompts(
        self,
        prompts: list[str],
        output_file: str | None = None,
        output_format: str = "jsonl",
    ) -> list[dict[str, str]]:
        """Process prompts sequentially, print results, optionally save."""
        if not prompts:
            self.console.print("[yellow]⚠️ No prompts to process.[/yellow]")
            return []

        results: list[dict[str, str]] = []
        for idx, prompt in enumerate(track(prompts, description="Processing prompts...")):
            self.console.print(f"\n[bold yellow]Prompt {idx + 1}/{len(prompts)}:[/bold yellow]")
            self.console.print(f"[italic cyan]{prompt}[/italic cyan]\n")

            try:
                content = self.client.generate(prompt).strip()
                self.console.print(f"[bold green]Response:[/bold green]\n{content}\n")
                result = {"prompt": prompt, "response": content}
            except Exception as e:
                error_message = f"Error processing prompt: {e!s}"
                self.console.print(f"[bold red]{error_message}[/bold red]\n")
                result = {"prompt": prompt, "response": f"ERROR: {e!s}"}
                # It's generally better to log the full exception for debugging
                # and then raise a more specific, user-friendly error.
                raise BatchProcessorError(error_message) from e

            results.append(result)

            # Incremental save for jsonl
            if output_file and output_format == "jsonl":
                self._append_jsonl(output_file, result)

            time.sleep(self.delay)

        # Batch save for other formats
        if output_file and output_format != "jsonl":
            self._save_batch(output_file, results, output_format)

        self.console.print(f"[bold green]✅ Completed {len(results)} prompts.[/bold green]")
        return results

    def _append_jsonl(self, output_file: str, result: dict[str, str]) -> None:
        """Append single result to JSONL file."""
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with Path(output_file).open("a", encoding="utf-8") as f:
            json.dump(result, f)
            f.write("\n")

    def _validate_format_and_save(self, output_file: str, results: list[dict[str, str]], fmt: str) -> None:
        """Helper to validate format and save batch results."""
        if fmt == "json":
            with Path(output_file).open("w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)
        elif fmt == "csv":
            with Path(output_file).open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["prompt", "response"])
                writer.writeheader()
                writer.writerows(results)
        else:
            error_message = f"Unsupported format '{fmt}'. Use 'jsonl', 'json', or 'csv'."
            raise ValueError(error_message)
        self.console.print(f"[green]💾 Saved to {output_file} ({fmt.upper()})[/green]")

    def _save_batch(self, output_file: str, results: list[dict[str, str]], fmt: str) -> None:
        """Save batch results in specified format."""
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        try:
            self._validate_format_and_save(output_file, results, fmt)
        except Exception as e:
            error_message = f"Save failed: {e!s}"
            self.console.print(f"[red]❌ {error_message}[/red]")
            raise BatchProcessorError(error_message) from e
