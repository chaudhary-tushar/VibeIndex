"""
BatchProcessor_2 for sequential batch processing of prompts through LLMClient.
Adapted from old_code/batch_processor.py with enhancements for dirs, formats, etc.
Compatible with src/generation/ module.
"""
import os
import json
import time
import csv
from pathlib import Path
from typing import List, Optional, Dict
from rich.console import Console
from rich.progress import track
from .generator import LLMClient


class BatchProcessor_2:
    """Batch processor class for prompts."""

    def __init__(self, delay: float = 0.2):
        self.delay = delay
        self.console = Console()
        self.client = LLMClient()

    def load_prompts(self, input_path: str) -> List[str]:
        """Load prompts from file or directory of .txt files."""
        path = Path(input_path).resolve()
        if not path.exists():
            self.console.print(f"[red]❌ Input path not found: {input_path}[/red]")
            raise FileNotFoundError(f"Input path not found: {input_path}")

        prompts: List[str] = []
        if path.is_file() and path.suffix.lower() == '.txt':
            with open(path, "r", encoding="utf-8") as f:
                file_prompts = [line.strip() for line in f if line.strip()]
                prompts.extend(file_prompts)
                self.console.print(f"[green]✅ Loaded {len(file_prompts)} prompts from {path.name}[/green]")
        elif path.is_dir():
            txt_files = list(path.rglob("*.txt"))
            if not txt_files:
                self.console.print("[yellow]⚠️ No .txt files found in directory.[/yellow]")
                return prompts
            for txt_file in sorted(txt_files):
                with open(txt_file, "r", encoding="utf-8") as f:
                    file_prompts = [line.strip() for line in f if line.strip()]
                    prompts.extend(file_prompts)
                    self.console.print(f"[green]✅ Loaded {len(file_prompts)} from {txt_file.name}[/green]")
            self.console.print(f"[bold green]📊 Total: {len(prompts)} prompts[/bold green]")
        else:
            raise ValueError(f"❌ Unsupported input: {input_path} (use .txt file or dir)")
        return prompts

    def process_prompts(
        self,
        prompts: List[str],
        output_file: Optional[str] = None,
        output_format: str = "jsonl",
    ) -> List[Dict[str, str]]:
        """Process prompts sequentially, print results, optionally save."""
        if not prompts:
            self.console.print("[yellow]⚠️ No prompts to process.[/yellow]")
            return []

        results: List[Dict[str, str]] = []
        for idx, prompt in enumerate(track(prompts, description="Processing prompts...")):
            self.console.print(f"\n[bold yellow]Prompt {idx + 1}/{len(prompts)}:[/bold yellow]")
            self.console.print(f"[italic cyan]{prompt}[/italic cyan]\n")

            try:
                content = self.client.generate(prompt).strip()
                self.console.print(f"[bold green]Response:[/bold green]\n{content}\n")
                result = {"prompt": prompt, "response": content}
            except Exception as e:
                content = f"ERROR: {str(e)}"
                self.console.print(f"[bold red]{content}[/bold red]\n")
                result = {"prompt": prompt, "response": content}

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

    def _append_jsonl(self, output_file: str, result: Dict[str, str]) -> None:
        """Append single result to JSONL file."""
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "a", encoding="utf-8") as f:
            json.dump(result, f)
            f.write("\n")

    def _save_batch(self, output_file: str, results: List[Dict[str, str]], fmt: str) -> None:
        """Save batch results in specified format."""
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        try:
            if fmt == "json":
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=2)
            elif fmt == "csv":
                with open(output_file, "w", encoding="utf-8", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=["prompt", "response"])
                    writer.writeheader()
                    writer.writerows(results)
            else:
                raise ValueError(f"Unsupported format '{fmt}'. Use 'jsonl', 'json', or 'csv'.")
            self.console.print(f"[green]💾 Saved to {output_file} ({fmt.upper()})[/green]")
        except Exception as e:
            self.console.print(f"[red]❌ Save failed: {e}[/red]")