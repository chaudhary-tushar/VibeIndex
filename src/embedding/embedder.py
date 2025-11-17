"""
Embedding generation using local models (Ollama/Docker)
"""

import json
import requests
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from rich.console import Console

console = Console()


@dataclass
class EmbeddingConfig:
    """Configuration for embedding generation"""
    model_url: str = "http://localhost:12434/engines/llama.cpp/v1"  # Ollama/Docker endpoint
    model_name: str = "ai/embeddinggemma"  # or your embedding model
    embedding_dim: int = 768  # adjust based on model
    batch_size: int = 32
    max_retries: int = 3
    timeout: int = 30


class EmbeddingGenerator:
    """Generate embeddings using Docker-hosted embedding model"""

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