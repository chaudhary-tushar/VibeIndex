"""
Embedding Generation & Qdrant Indexing Pipeline
Processes parsed code chunks, generates embeddings, and indexes in Qdrant

Prerequisites:
pip install qdrant-client requests tqdm sentence-transformers

Docker setup:
1. docker run -p 11434:11434 --name embeddinggamma ollama/ollama
2. docker exec -it embeddinggamma ollama pull nomic-embed-text  # or your embedding model
3. docker run -p 6333:6333 -p 6334:6334 -v ./qdrant_storage:/qdrant/storage qdrant/qdrant
"""

import json
import requests
import hashlib
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from tqdm import tqdm
from qdrant_client import QdrantClient
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue,
    PayloadSchemaType, TextIndexParams, TextIndexType,
    CreateCollection, SparseVectorParams, SparseIndexParams
)
import sys

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


@dataclass
class QdrantConfig:
    """Configuration for Qdrant"""
    host: str = "localhost"
    port: int = 6333
    collection_prefix: str = "tipsy"
    distance_metric: Distance = Distance.COSINE
    enable_sparse_vectors: bool = True
    enable_payload_index: bool = True


class ChunkPreprocessor:
    """Enhanced preprocessing for code chunks before embedding"""

    def __init__(self):
        self.dedup_hashes = set()
        self.stats = {
            'total': 0,
            'duplicates': 0,
            'enhanced': 0,
            'too_large': 0,
        }

    def deduplicate(self, chunks: List[Dict]) -> List[Dict]:
        """Remove duplicate chunks based on code content"""
        unique_chunks = []

        for chunk in chunks:
            # Create hash from code content
            code_hash = hashlib.md5(chunk['code'].encode()).hexdigest()

            if code_hash not in self.dedup_hashes:
                self.dedup_hashes.add(code_hash)
                unique_chunks.append(chunk)
            else:
                self.stats['duplicates'] += 1

        console.print(f"[yellow]Removed {self.stats['duplicates']} duplicate chunks[/yellow]")
        return unique_chunks

    def enhance_chunk(self, chunk: Dict) -> Dict:
        """Enhance chunk with additional context for better embeddings"""
        enhanced = chunk.copy()
        # print(enhanced)

        # Build rich text representation for embedding
        parts = []

        # 1. Add contextual prefix
        context_prefix = f"# {chunk['language'].upper()} {chunk['type'].upper()}"
        if chunk.get('qualified_name'):
            context_prefix += f": {chunk['qualified_name']}"
        parts.append(context_prefix)
        if isinstance(chunk.get('context'), str):
            parts.append(f"html/css context: {chunk.get("context")}")
        else:
            # 2. Add file context
            if chunk.get('context', {}).get('file_hierarchy'):
                file_ctx = " > ".join(chunk['context']['file_hierarchy'])
                parts.append(f"# Location: {file_ctx}")

            # 3. Add domain context
            if chunk.get('context', {}).get('domain_context'):
                parts.append(f"# Purpose: {chunk['context']['domain_context']}")

        # 4. Add docstring if available
        if chunk.get('docstring'):
            parts.append(f'"""{chunk["docstring"]}"""')

        # 5. Add signature for functions
        if chunk.get('signature'):
            parts.append(chunk['signature'])

        # 6. Add decorators/metadata
        if chunk.get('metadata', {}).get('decorators'):
            parts.extend(chunk['metadata']['decorators'])

        # 7. Add the actual code
        parts.append(chunk['code'])

        # 8. Add dependencies as comments
        if chunk.get('dependencies'):
            deps = ", ".join(chunk['dependencies'][:5])  # Limit to 5
            parts.append(f"# Dependencies: {deps}")

        # 9. Add defined symbols
        if chunk.get('defines'):
            parts.append(f"# Defines: {', '.join(chunk['defines'])}")

        # Combine all parts
        enhanced['embedding_text'] = "\n".join(parts)
        enhanced['embedding_text_length'] = len(enhanced['embedding_text'])

        self.stats['enhanced'] += 1
        return enhanced

    def validate_chunk(self, chunk: Dict, max_tokens: int = 8192) -> bool:
        """Validate chunk is suitable for embedding"""
        # Rough token estimate (1 token ≈ 4 chars)
        estimated_tokens = len(chunk.get('embedding_text', chunk['code'])) // 4

        if estimated_tokens > max_tokens:
            self.stats['too_large'] += 1
            return False

        return True

    def process(self, chunks: List[Dict]) -> List[Dict]:
        """Full preprocessing pipeline"""
        self.stats['total'] = len(chunks)
        console.print(f"[cyan]Starting preprocessing of {len(chunks)} chunks...[/cyan]")

        # 1. Deduplicate
        chunks = self.deduplicate(chunks)

        # 2. Enhance each chunk
        enhanced_chunks = []
        for chunk in tqdm(chunks, desc="Enhancing chunks", unit="chunk"):
            enhanced = self.enhance_chunk(chunk)
            if self.validate_chunk(enhanced):
                enhanced_chunks.append(enhanced)

        console.print(f"[green]✓ Preprocessing complete: {len(enhanced_chunks)} chunks ready[/green]")
        console.print(f"  - Duplicates removed: {self.stats['duplicates']}")
        console.print(f"  - Too large (skipped): {self.stats['too_large']}")

        return enhanced_chunks


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
                # print(response)

                if response.status_code == 200:
                    result = response.json()
                    # print(result.get('data')[0].get('embedding'))
                    return result.get('data')[0].get('embedding')
                else:
                    console.print(f"[yellow]Attempt {attempt + 1} failed: {response.status_code}[/yellow]")
                    # 🔥 Print raw response body (often contains detailed error message)
                    try:
                        console.print(f"[red]Error body: {response.text}[/red]")
                        # print(payload)
                        # sys.exit()
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


class QdrantIndexer:
    """Index embedded chunks in Qdrant with optimized schema"""

    def __init__(self, config: QdrantConfig):
        self.config = config
        self.client = QdrantClient(host=config.host, port=config.port)
        self.collections = {}

    def create_collections(self, embedding_dim: int):
        """Create optimized Qdrant collections for different chunk types"""

        collection_configs = {
            f"{self.config.collection_prefix}_functions": "Individual functions and methods",
            f"{self.config.collection_prefix}_classes": "Class definitions",
            f"{self.config.collection_prefix}_modules": "Module and file-level code",
        }

        for collection_name, description in collection_configs.items():
            try:
                # Check if collection exists
                self.client.get_collection(collection_name)
                console.print(f"[yellow]Collection '{collection_name}' already exists[/yellow]")
            except:
                # Create collection
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=embedding_dim,
                        distance=self.config.distance_metric,
                        on_disk=False,  # Keep in memory for speed
                    ),
                    sparse_vectors_config={
                        # Sparse identifiers (variable/function/class names)
                        "symbols": SparseVectorParams(
                            index=SparseIndexParams(on_disk=False)
                        ),
                        # BM25-like text search (docstrings, comments)
                        "bm25": SparseVectorParams(
                            index=SparseIndexParams(on_disk=False)
                        ),
                        "text-sparse": SparseVectorParams(
                            index=SparseIndexParams(on_disk=False)
                        ),
                    }
                )

                # Create payload indexes for filtering
                if self.config.enable_payload_index:
                    # Index key fields
                    self.client.create_payload_index(
                        collection_name=collection_name,
                        field_name="language",
                        field_schema=PayloadSchemaType.KEYWORD
                    )
                    self.client.create_payload_index(
                        collection_name=collection_name,
                        field_name="type",
                        field_schema=PayloadSchemaType.KEYWORD
                    )
                    self.client.create_payload_index(
                        collection_name=collection_name,
                        field_name="file_path",
                        field_schema=PayloadSchemaType.KEYWORD
                    )
                    self.client.create_payload_index(
                        collection_name=collection_name,
                        field_name="qualified_name",
                        field_schema=PayloadSchemaType.TEXT,
                        # field_index_params=TextIndexParams(
                        #     type=TextIndexType.TEXT,
                        #     tokenizer="word",
                        #     min_token_len=2,
                        #     max_token_len=20,
                        # )
                    )

                console.print(f"[green]✓ Created collection: {collection_name}[/green]")
                console.print(f"  {description}")

            self.collections[collection_name] = True

    def _get_collection_for_chunk(self, chunk: Dict) -> str:
        """Determine which collection a chunk should go into"""
        chunk_type = chunk.get('type', 'unknown')

        if chunk_type in ['function', 'method']:
            return f"{self.config.collection_prefix}_functions"
        elif chunk_type == 'class':
            return f"{self.config.collection_prefix}_classes"
        else:
            return f"{self.config.collection_prefix}_modules"

    def _prepare_payload(self, chunk: Dict) -> Dict:
        """Prepare chunk payload for Qdrant (remove embedding, keep metadata)"""
        payload = chunk.copy()

        # Remove embedding (stored separately)
        payload.pop('embedding', None)
        payload.pop('embedding_text', None)  # Don't need full text in payload

        # Keep essential fields easily filterable
        essential = {
            'id': chunk['id'],
            'name': chunk['name'],
            'type': chunk['type'],
            'language': chunk['language'],
            'file_path': chunk['file_path'],
            'qualified_name': chunk.get('qualified_name', ''),
            'code': chunk['code'],  # Keep original code
            'start_line': chunk['start_line'],
            'end_line': chunk['end_line'],
            'complexity': chunk.get('complexity', 0),
            'docstring': chunk.get('docstring'),
            'signature': chunk.get('signature'),
            'dependencies': chunk.get('dependencies', []),
            'context': chunk.get('context', {}),
            'metadata': chunk.get('metadata', {}),
            'relationships': chunk.get('relationships', {}),
            'analysis': chunk.get('analysis', {}),
        }

        return essential

    def index_chunks(self, chunks: List[Dict], batch_size: int = 100):
        """Index chunks in Qdrant"""
        console.print(f"[cyan]Indexing {len(chunks)} chunks in Qdrant...[/cyan]")

        # Group chunks by collection
        by_collection = {}
        for chunk in chunks:
            collection = self._get_collection_for_chunk(chunk)
            if collection not in by_collection:
                by_collection[collection] = []
            by_collection[collection].append(chunk)

        # Index each collection
        total_indexed = 0
        for collection_name, coll_chunks in by_collection.items():
            console.print(f"  Indexing {len(coll_chunks)} chunks in {collection_name}...")

            # Prepare points
            points = []
            for chunk in coll_chunks:
                point = PointStruct(
                    id=int(chunk["id"], 16), # Use your chunk ID
                    vector=chunk['embedding'],
                    payload=self._prepare_payload(chunk)
                )
                points.append(point)

            # Batch upload
            for i in tqdm(range(0, len(points), batch_size), desc=f"Uploading to {collection_name}"):
                batch = points[i:i + batch_size]
                self.client.upsert(
                    collection_name=collection_name,
                    points=batch
                )
                total_indexed += len(batch)

        console.print(f"[green]✓ Indexed {total_indexed} chunks successfully![/green]")

    def test_search(self, query: str, limit: int = 5):
        """Test search across all collections"""
        console.print(Panel.fit(
            f"[bold cyan]Test Search: '{query}'[/bold cyan]",
            border_style="cyan"
        ))

        # Generate query embedding
        embedding_gen = EmbeddingGenerator(EmbeddingConfig())
        query_embedding = embedding_gen.generate_embedding_ollama(query)
        # print(query_embedding)

        if not query_embedding:
            console.print("[red]Failed to generate query embedding[/red]")
            return

        # Search each collection
        for collection_name in self.collections.keys():
            console.print(f"\n[yellow]Results from {collection_name}:[/yellow]")

            results = self.client.search(
                collection_name=collection_name,
                query_vector=query_embedding,
                limit=limit
            )

            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("Score", width=8)
            table.add_column("Name", width=30)
            table.add_column("Type", width=10)
            table.add_column("File", width=40)

            for result in results:
                table.add_row(
                    f"{result.score:.4f}",
                    result.payload.get('qualified_name', result.payload.get('name', 'N/A')),
                    result.payload.get('type', 'N/A'),
                    result.payload.get('file_path', 'N/A')
                )

            console.print(table)


