"""
Qdrant vector database integration for indexing and searching
"""

import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path
from tqdm import tqdm
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue,
    PayloadSchemaType, TextIndexParams, TextIndexType,
    CreateCollection, SparseVectorParams, SparseIndexParams
)
from .embedder import EmbeddingGenerator, EmbeddingConfig

console = Console()


@dataclass
class QdrantConfig:
    """Configuration for Qdrant"""
    host: str = "localhost"
    port: int = 6333
    collection_prefix: str = "tipsy"
    distance_metric: Distance = Distance.COSINE
    enable_sparse_vectors: bool = True
    enable_payload_index: bool = True


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
                    id=int(chunk["id"], 16),  # Use chunk ID
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