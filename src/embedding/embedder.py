"""
Embedding generation using local models (Ollama/Docker) - Enhanced version
Includes advanced batch processing, quality validation, and error handling
"""

import json
import requests
import time
import math
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn
# Import consolidated configuration for backward compatibility
from src.config import EmbeddingConfig

console = Console()




class EmbeddingGenerator:
    """Enhanced embedding generation with batch processing and quality validation"""

    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.stats = {
            'success': 0,
            'failed': 0,
            'total_time': 0,
            'quality_checks_passed': 0,
            'quality_checks_failed': 0,
        }

    def validate_embedding_quality(self, embedding: List[float]) -> Tuple[bool, str]:
        """Validate embedding quality based on statistical properties"""
        if not embedding:
            return False, "Empty embedding"

        if len(embedding) != self.config.embedding_dim:
            return False, f"Wrong dimension: expected {self.config.embedding_dim}, got {len(embedding)}"

        # Check for NaN or infinite values
        for i, val in enumerate(embedding):
            if not (math.isfinite(val)):
                return False, f"Non-finite value at index {i}: {val}"

        # Check embedding magnitude (should be reasonable)
        magnitude = sum(x * x for x in embedding) ** 0.5
        if magnitude < 0.1:
            return False, f"Embedding magnitude too low: {magnitude}"
        if magnitude > 1000:
            return False, f"Embedding magnitude too high: {magnitude}"

        # Check if embedding is too uniform (low entropy/variance)
        import numpy as np
        arr = np.array(embedding)
        variance = np.var(arr)
        if variance < 0.0001:  # Very low variance indicates uniform values
            return False, f"Embedding variance too low: {variance}, suggesting uniform/uninformative values"

        return True, "Valid embedding"

    def generate_embedding_ollama(self, text: str) -> Optional[List[float]]:
        """Generate embedding using Ollama API with enhanced error handling"""
        url = f"{self.config.model_url}/embeddings"

        payload = {
            "model": self.config.model_name,
            "input": text
        }

        for attempt in range(self.config.max_retries):
            try:
                response = requests.post(
                    url,
                    json=payload,
                    timeout=self.config.timeout
                )

                if response.status_code == 200:
                    result = response.json()
                    embedding = result.get('data')[0].get('embedding')

                    # Validate embedding quality
                    is_valid, message = self.validate_embedding_quality(embedding)
                    if is_valid:
                        return embedding
                    else:
                        self.stats['quality_checks_failed'] += 1
                        console.print(f"[yellow]Quality check failed: {message}[/yellow]")
                        if attempt == self.config.max_retries - 1:
                            console.print(f"[red]Quality validation failed after {self.config.max_retries} attempts[/red]")
                else:
                    console.print(f"[yellow]Attempt {attempt + 1} failed: {response.status_code}[/yellow]")
                    try:
                        console.print(f"[red]Error body: {response.text}[/red]")
                    except Exception:
                        pass

            except requests.exceptions.RequestException as e:
                console.print(f"[yellow]Attempt {attempt + 1} failed: {e}[/yellow]")
                time.sleep(2 ** attempt)  # Exponential backoff

        return None

    def generate_batch(self, chunks: List[Dict]) -> List[Dict]:
        """Generate embeddings for a batch of chunks with enhanced error handling"""
        embedded_chunks = []

        console.print(f"[cyan]Processing batch of {len(chunks)} chunks...[/cyan]")

        for chunk in chunks:
            start_time = time.time()

            # Use enhanced text for embedding
            text = chunk.get('embedding_text', chunk['code'])

            # Validate input text
            if not text or len(text.strip()) == 0:
                self.stats['failed'] += 1
                console.print(f"[red]Empty text for chunk: {chunk.get('qualified_name', chunk.get('name', 'unknown'))}[/red]")
                continue

            embedding = self.generate_embedding_ollama(text)

            if embedding:
                # Additional quality validation
                is_valid, message = self.validate_embedding_quality(embedding)
                if is_valid:
                    chunk['embedding'] = embedding
                    chunk['embedding_model'] = self.config.model_name
                    chunk['embedding_timestamp'] = time.time()
                    chunk['embedding_quality'] = 'validated'
                    embedded_chunks.append(chunk)
                    self.stats['success'] += 1
                    self.stats['quality_checks_passed'] += 1
                else:
                    self.stats['failed'] += 1
                    self.stats['quality_checks_failed'] += 1
                    console.print(f"[red]Quality validation failed for: {chunk['qualified_name'] or chunk['name']} - {message}[/red]")
            else:
                self.stats['failed'] += 1
                console.print(f"[red]Failed to embed: {chunk['qualified_name'] or chunk['name']}[/red]")

            self.stats['total_time'] += time.time() - start_time

        return embedded_chunks

    def generate_all(self, chunks: List[Dict], parallel: bool = True) -> List[Dict]:
        """Generate embeddings for all chunks with enhanced reporting"""
        console.print(Panel.fit(
            f"[bold cyan]Starting Enhanced Embedding Generation[/bold cyan]",
            subtitle=f"Model: {self.config.model_name} | {len(chunks)} chunks",
            border_style="cyan"
        ))

        all_embedded = []

        # Split into batches
        batches = [
            chunks[i:i + self.config.batch_size]
            for i in range(0, len(chunks), self.config.batch_size)
        ]

        with Progress(
            SpinnerColumn(),
            *Progress.get_default_columns(),
            TimeElapsedColumn(),
            console=console
        ) as progress:

            if parallel and len(batches) > 1:
                # Parallel processing
                task = progress.add_task("Processing batches in parallel...", total=len(batches))

                with ThreadPoolExecutor(max_workers=4) as executor:
                    futures = [executor.submit(self.generate_batch, batch) for batch in batches]

                    for future in as_completed(futures):
                        batch_results = future.result()
                        all_embedded.extend(batch_results)
                        progress.advance(task)
            else:
                # Sequential processing with progress bar
                task = progress.add_task("Processing batches sequentially...", total=len(batches))

                for batch in batches:
                    batch_results = self.generate_batch(batch)
                    all_embedded.extend(batch_results)
                    progress.advance(task)

        # Enhanced statistics
        total_time = self.stats['total_time']
        avg_time = self.stats['total_time'] / self.stats['success'] if self.stats['success'] > 0 else 0
        success_rate = (self.stats['success'] / len(chunks)) * 100 if chunks else 0

        console.print(f"\n[green]✓ Embedding Generation Complete![/green]")
        console.print(f"  [bold]Success:[/bold] {self.stats['success']}/{len(chunks)} ({success_rate:.1f}%)")
        console.print(f"  [bold]Failed:[/bold] {self.stats['failed']}")
        console.print(f"  [bold]Quality Checks:[/bold] Passed: {self.stats['quality_checks_passed']}, Failed: {self.stats['quality_checks_failed']}")
        console.print(f"  [bold]Performance:[/bold] Avg: {avg_time:.3f}s per chunk, Total: {total_time:.1f}s")

        return all_embedded


# Preserve original EmbeddingGenerator with "_2" suffix for backward compatibility
class EmbeddingGenerator_2:
    """Original EmbeddingGenerator implementation (preserved for compatibility)"""

    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.stats = {
            'success': 0,
            'failed': 0,
            'total_time': 0,
        }

    def generate_embedding_ollama(self, text: str) -> Optional[List[float]]:
        """Generate embedding using Ollama API"""
        url = f"{self.config.model_url}/embeddings"

        payload = {
            "model": self.config.model_name,
            "input": text
        }

        for attempt in range(self.config.max_retries):
            try:
                response = requests.post(
                    url,
                    json=payload,
                    timeout=self.config.timeout
                )

                if response.status_code == 200:
                    result = response.json()
                    return result.get('data')[0].get('embedding')
                else:
                    console.print(f"[yellow]Attempt {attempt + 1} failed: {response.status_code}[/yellow]")
                    try:
                        console.print(f"[red]Error body: {response.text}[/red]")
                    except Exception:
                        pass

            except requests.exceptions.RequestException as e:
                console.print(f"[yellow]Attempt {attempt + 1} failed: {e}[/yellow]")
                time.sleep(2 ** attempt)  # Exponential backoff

        return None

    def generate_batch(self, chunks: List[Dict]) -> List[Dict]:
        """Generate embeddings for a batch of chunks"""
        embedded_chunks = []

        for chunk in chunks:
            start_time = time.time()

            # Use enhanced text for embedding
            text = chunk.get('embedding_text', chunk['code'])

            embedding = self.generate_embedding_ollama(text)

            if embedding:
                chunk['embedding'] = embedding
                chunk['embedding_model'] = self.config.model_name
                chunk['embedding_timestamp'] = time.time()
                embedded_chunks.append(chunk)
                self.stats['success'] += 1
            else:
                self.stats['failed'] += 1
                console.print(f"[red]Failed to embed: {chunk['qualified_name'] or chunk['name']}[/red]")

            self.stats['total_time'] += time.time() - start_time

        return embedded_chunks

    def generate_all(self, chunks: List[Dict], parallel: bool = True) -> List[Dict]:
        """Generate embeddings for all chunks"""
        console.print(f"[cyan]Generating embeddings for {len(chunks)} chunks...[/cyan]")
        console.print(f"Model: {self.config.model_name} @ {self.config.model_url}")

        all_embedded = []

        # Split into batches
        batches = [
            chunks[i:i + self.config.batch_size]
            for i in range(0, len(chunks), self.config.batch_size)
        ]

        if parallel and len(batches) > 1:
            # Parallel processing
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = [executor.submit(self.generate_batch, batch) for batch in batches]

                for future in tqdm(as_completed(futures), total=len(batches), desc="Embedding batches"):
                    all_embedded.extend(future.result())
        else:
            # Sequential processing with progress bar
            for batch in tqdm(batches, desc="Embedding batches"):
                all_embedded.extend(self.generate_batch(batch))

        # Stats
        avg_time = self.stats['total_time'] / self.stats['success'] if self.stats['success'] > 0 else 0
        console.print(f"[green]✓ Embedding complete![/green]")
        console.print(f"  - Success: {self.stats['success']}")
        console.print(f"  - Failed: {self.stats['failed']}")
        console.print(f"  - Avg time: {avg_time:.3f}s per chunk")

        return all_embedded