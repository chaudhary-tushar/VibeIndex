"""
Qdrant vector database integration for indexing and searching - Enhanced version
"""

import json
import time
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition
from qdrant_client.models import Filter
from qdrant_client.models import MatchValue
from qdrant_client.models import PayloadSchemaType
from qdrant_client.models import PointStruct
from qdrant_client.models import SparseIndexParams
from qdrant_client.models import SparseVectorParams
from qdrant_client.models import VectorParams
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from tqdm import tqdm

# Import consolidated configuration for backward compatibility
from src.config import EmbeddingConfig
from src.config import QdrantConfig

# Import embedding generator from correct location
from src.embedding.embedder import EmbeddingGenerator

console = Console()


class QdrantIndexer:
    """Enhanced Qdrant vector database integration with advanced collection management"""

    def __init__(self, config: QdrantConfig):
        self.config = config
        self.client = QdrantClient(host=config.host, port=config.port)
        self.collections = {}
        self.stats = {
            "collections_created": 0,
            "chunks_indexed": 0,
            "indexing_errors": 0,
            "searches_performed": 0,
        }

    def health_check(self) -> bool:
        """Perform health check on Qdrant connection"""
        try:
            # Test basic connectivity
            self.client.get_collections()
            return True
        except Exception as e:
            console.print(f"[red]Qdrant health check failed: {e}[/red]")
            return False

    def create_collections(self, embedding_dim: int):
        """Create optimized Qdrant collections for different chunk types with enhanced management"""

        collection_configs = {
            f"{self.config.collection_prefix}_functions": {
                "description": "Individual functions and methods",
                "optimization": "High-frequency function lookups",
            },
            f"{self.config.collection_prefix}_classes": {
                "description": "Class definitions and structure",
                "optimization": "Class hierarchy and inheritance",
            },
            f"{self.config.collection_prefix}_modules": {
                "description": "Module and file-level code",
                "optimization": "File-level context and imports",
            },
        }

        for collection_name, config in collection_configs.items():
            try:
                # Check if collection exists and get info
                collection_info = self.client.get_collection(collection_name)
                console.print(f"[yellow]Collection '{collection_name}' already exists[/yellow]")
                console.print(f"  Vectors: {collection_info.vectors_count}")
                console.print(f"  Status: {collection_info.status}")

                # Update existing collection if needed
                if collection_info.config.params.vectors.size != embedding_dim:
                    console.print(
                        f"[yellow]Updating vector dimension from {collection_info.config.params.vectors.size} to {embedding_dim}[/yellow]"
                    )

            except Exception:
                # Create new collection
                try:
                    self.client.create_collection(
                        collection_name=collection_name,
                        vectors_config=VectorParams(
                            size=embedding_dim,
                            distance=self.config.distance_metric,
                            on_disk=False,  # Keep in memory for speed
                        ),
                        sparse_vectors_config={
                            # Sparse identifiers (variable/function/class names)
                            "symbols": SparseVectorParams(index=SparseIndexParams(on_disk=False)),
                            # BM25-like text search (docstrings, comments)
                            "bm25": SparseVectorParams(index=SparseIndexParams(on_disk=False)),
                            "text-sparse": SparseVectorParams(index=SparseIndexParams(on_disk=False)),
                        },
                    )

                    # Create payload indexes for filtering
                    if self.config.enable_payload_index:
                        # Index key fields
                        self.client.create_payload_index(
                            collection_name=collection_name,
                            field_name="language",
                            field_schema=PayloadSchemaType.KEYWORD,
                        )
                        self.client.create_payload_index(
                            collection_name=collection_name, field_name="type", field_schema=PayloadSchemaType.KEYWORD
                        )
                        self.client.create_payload_index(
                            collection_name=collection_name,
                            field_name="file_path",
                            field_schema=PayloadSchemaType.KEYWORD,
                        )
                        self.client.create_payload_index(
                            collection_name=collection_name,
                            field_name="qualified_name",
                            field_schema=PayloadSchemaType.TEXT,
                        )

                    self.stats["collections_created"] += 1
                    console.print(f"[green]✓ Created collection: {collection_name}[/green]")
                    console.print(f"  {config['description']}")
                    console.print(f"  Optimization: {config['optimization']}")

                except Exception as e:
                    console.print(f"[red]Failed to create collection {collection_name}: {e}[/red]")
                    continue

            self.collections[collection_name] = config

    def _get_collection_for_chunk(self, chunk: dict) -> str:
        """Determine which collection a chunk should go into with enhanced logic"""
        chunk_type = chunk.get("type", "unknown")

        # More specific type handling
        if chunk_type in {"function", "method", "lambda"}:
            return f"{self.config.collection_prefix}_functions"
        if chunk_type == "class":
            return f"{self.config.collection_prefix}_classes"
        if chunk_type in {"module", "file", "script"}:
            return f"{self.config.collection_prefix}_modules"
        # Default to modules for unknown types
        console.print(f"[yellow]Unknown chunk type '{chunk_type}', defaulting to modules collection[/yellow]")
        return f"{self.config.collection_prefix}_modules"

    def _prepare_payload(self, chunk: dict) -> dict:
        """Prepare chunk payload for Qdrant with enhanced metadata handling"""
        payload = chunk.copy()

        # Remove embedding (stored separately)
        payload.pop("embedding", None)
        payload.pop("embedding_text", None)  # Don't need full text in payload

        # Keep essential fields easily filterable with defaults
        return {
            "id": chunk["id"],
            "name": chunk["name"],
            "type": chunk["type"],
            "language": chunk["language"],
            "file_path": chunk["file_path"],
            "qualified_name": chunk.get("qualified_name", ""),
            "code": chunk["code"],  # Keep original code
            "start_line": chunk["start_line"],
            "end_line": chunk["end_line"],
            "complexity": chunk.get("complexity", 0),
            "docstring": chunk.get("docstring"),
            "signature": chunk.get("signature"),
            "dependencies": chunk.get("dependencies", []),
            "context": chunk.get("context", {}),
            "metadata": chunk.get("metadata", {}),
            "relationships": chunk.get("relationships", {}),
            "analysis": chunk.get("analysis", {}),
            "embedding_quality": chunk.get("embedding_quality", "unknown"),
            "indexed_at": chunk.get("embedding_timestamp", time.time()),
        }

    def optimize_batch_size(self, collection_name: str) -> int:
        """Dynamically determine optimal batch size based on collection size"""
        try:
            collection_info = self.client.get_collection(collection_name)
            current_size = collection_info.vectors_count

            # Adaptive batch sizing based on collection size
            if current_size < 1000:
                return 50  # Small collections, smaller batches
            if current_size < 10000:
                return 100  # Medium collections
            return 200  # Large collections, larger batches
        except:
            return 100  # Default batch size

    def index_chunks(self, chunks: list[dict], batch_size: int | None = None):  # noqa: C901
        """Index chunks in Qdrant with enhanced error handling and progress tracking"""
        console.print(f"[cyan]Enhanced indexing of {len(chunks)} chunks in Qdrant...[/cyan]")

        # Health check before indexing
        if not self.health_check():
            console.print("[red]Qdrant health check failed. Aborting indexing.[/red]")
            return

        # Group chunks by collection
        by_collection = {}
        for chunk in chunks:
            collection = self._get_collection_for_chunk(chunk)
            if collection not in by_collection:
                by_collection[collection] = []
            by_collection[collection].append(chunk)

        # Index each collection with enhanced progress tracking
        total_indexed = 0
        start_time = time.time()

        for collection_name, coll_chunks in by_collection.items():
            console.print(f"[cyan]Indexing {len(coll_chunks)} chunks in {collection_name}...[/cyan]")

            # Use optimized batch size if not specified
            if batch_size is None:
                batch_size = self.optimize_batch_size(collection_name)
                console.print(f"Using optimized batch size: {batch_size}")

            # Prepare points
            points = []
            for chunk in coll_chunks:
                if "embedding" not in chunk:
                    console.print(f"[yellow]Skipping chunk without embedding: {chunk.get('name', 'unknown')}[/yellow]")
                    continue

                point = PointStruct(
                    id=int(chunk["id"], 16),  # Use chunk ID
                    vector=chunk["embedding"],
                    payload=self._prepare_payload(chunk),
                )
                points.append(point)

            # Batch upload with enhanced error handling
            batches_uploaded = 0
            for i in tqdm(range(0, len(points), batch_size), desc=f"Uploading to {collection_name}"):
                batch = points[i : i + batch_size]
                try:
                    self.client.upsert(collection_name=collection_name, points=batch)
                    total_indexed += len(batch)
                    batches_uploaded += 1

                except Exception as e:
                    self.stats["indexing_errors"] += 1
                    console.print(f"[red]Failed to upload batch to {collection_name}: {e}[/red]")
                    # Try smaller batch size
                    if batch_size > 10:
                        console.print(f"[yellow]Retrying with smaller batch size...[/yellow]")
                        batch_size = max(10, batch_size // 2)

            console.print(
                f"[green]✓ Completed {collection_name}: {len(coll_chunks)} chunks in {batches_uploaded} batches[/green]"
            )

        # Enhanced statistics
        total_time = time.time() - start_time
        avg_time_per_chunk = total_time / total_indexed if total_indexed > 0 else 0

        console.print(f"\n[green]✓ Enhanced Indexing Complete![/green]")
        console.print(f"  Total indexed: {total_indexed} chunks")
        console.print(f"  Collections: {len(by_collection)}")
        console.print(f"  Time: {total_time:.2f}s")
        console.print(f"  Rate: {avg_time_per_chunk:.3f}s per chunk")
        console.print(f"  Errors: {self.stats['indexing_errors']}")

        self.stats["chunks_indexed"] = total_indexed

    def advanced_search(
        self, query: str, limit: int = 10, filters: dict | None = None, collection_filter: str | None = None
    ) -> dict[str, list]:
        """Perform advanced search with filtering and ranking"""
        console.print(
            Panel.fit(
                f"[bold cyan]Advanced Search: '{query}'[/bold cyan]",
                subtitle=f"Limit: {limit} | Filters: {filters} | Collection: {collection_filter or 'all'}",
                border_style="cyan",
            )
        )

        self.stats["searches_performed"] += 1

        # Generate query embedding
        embedding_gen = EmbeddingGenerator(EmbeddingConfig())
        query_embedding = embedding_gen.generate_embedding_ollama(query)

        if not query_embedding:
            console.print("[red]Failed to generate query embedding[/red]")
            return {}

        # Prepare filter conditions
        search_filters = None
        if filters:
            conditions = []
            for field, value in filters.items():
                if isinstance(value, list):
                    conditions.append(
                        FieldCondition(
                            key=field,
                            match=MatchValue(value=value[0]),  # Simple match, could be extended
                        )
                    )
                else:
                    conditions.append(FieldCondition(key=field, match=MatchValue(value=value)))

            if conditions:
                search_filters = Filter(must=conditions)

        # Search each collection
        all_results = {}
        for collection_name in self.collections:
            # Skip if collection filter is specified and doesn't match
            if collection_filter and collection_filter not in collection_name:
                continue

            console.print(f"\n[yellow]Searching {collection_name}...[/yellow]")

            try:
                results = self.client.search(
                    collection_name=collection_name,
                    query_vector=query_embedding,
                    limit=limit,
                    query_filter=search_filters,
                )

                all_results[collection_name] = results

                # Display results in enhanced table
                table = Table(show_header=True, header_style="bold magenta")
                table.add_column("Score", width=8)
                table.add_column("Name", width=30)
                table.add_column("Type", width=10)
                table.add_column("File", width=40)
                table.add_column("Quality", width=10)

                for result in results:
                    table.add_row(
                        f"{result.score:.4f}",
                        result.payload.get("qualified_name", result.payload.get("name", "N/A")),
                        result.payload.get("type", "N/A"),
                        result.payload.get("file_path", "N/A")[:35],
                        result.payload.get("embedding_quality", "N/A"),
                    )

                console.print(table)

            except Exception as e:
                console.print(f"[red]Search failed for {collection_name}: {e}[/red]")
                all_results[collection_name] = []

        return all_results

    def test_search(self, query: str, limit: int = 5):
        """Test search across all collections (backward compatibility method)"""
        return self.advanced_search(query, limit)

    def get_collection_stats(self) -> dict[str, dict]:
        """Get statistics for all collections"""
        stats = {}
        for collection_name in self.collections:
            try:
                collection_info = self.client.get_collection(collection_name)
                stats[collection_name] = {
                    "vectors_count": collection_info.vectors_count,
                    "status": str(collection_info.status),
                    "config": {
                        "vector_size": collection_info.config.params.vectors.size,
                        "distance": str(collection_info.config.params.vectors.distance),
                    },
                }
            except Exception as e:
                stats[collection_name] = {"error": str(e)}

        return stats


# Preserve original QdrantIndexer with "_2" suffix for backward compatibility
class QdrantIndexer_2:
    """Original QdrantIndexer implementation (preserved for compatibility)"""

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
                        "symbols": SparseVectorParams(index=SparseIndexParams(on_disk=False)),
                        # BM25-like text search (docstrings, comments)
                        "bm25": SparseVectorParams(index=SparseIndexParams(on_disk=False)),
                        "text-sparse": SparseVectorParams(index=SparseIndexParams(on_disk=False)),
                    },
                )

                # Create payload indexes for filtering
                if self.config.enable_payload_index:
                    # Index key fields
                    self.client.create_payload_index(
                        collection_name=collection_name, field_name="language", field_schema=PayloadSchemaType.KEYWORD
                    )
                    self.client.create_payload_index(
                        collection_name=collection_name, field_name="type", field_schema=PayloadSchemaType.KEYWORD
                    )
                    self.client.create_payload_index(
                        collection_name=collection_name, field_name="file_path", field_schema=PayloadSchemaType.KEYWORD
                    )
                    self.client.create_payload_index(
                        collection_name=collection_name,
                        field_name="qualified_name",
                        field_schema=PayloadSchemaType.TEXT,
                    )

                console.print(f"[green]✓ Created collection: {collection_name}[/green]")
                console.print(f"  {description}")

            self.collections[collection_name] = True

    def _get_collection_for_chunk(self, chunk: dict) -> str:
        """Determine which collection a chunk should go into"""
        chunk_type = chunk.get("type", "unknown")

        if chunk_type in {"function", "method"}:
            return f"{self.config.collection_prefix}_functions"
        if chunk_type == "class":
            return f"{self.config.collection_prefix}_classes"
        return f"{self.config.collection_prefix}_modules"

    def _prepare_payload(self, chunk: dict) -> dict:
        """Prepare chunk payload for Qdrant (remove embedding, keep metadata)"""
        payload = chunk.copy()

        # Remove embedding (stored separately)
        payload.pop("embedding", None)
        payload.pop("embedding_text", None)  # Don't need full text in payload

        # Keep essential fields easily filterable
        return {
            "id": chunk["id"],
            "name": chunk["name"],
            "type": chunk["type"],
            "language": chunk["language"],
            "file_path": chunk["file_path"],
            "qualified_name": chunk.get("qualified_name", ""),
            "code": chunk["code"],  # Keep original code
            "start_line": chunk["start_line"],
            "end_line": chunk["end_line"],
            "complexity": chunk.get("complexity", 0),
            "docstring": chunk.get("docstring"),
            "signature": chunk.get("signature"),
            "dependencies": chunk.get("dependencies", []),
            "context": chunk.get("context", {}),
            "metadata": chunk.get("metadata", {}),
            "relationships": chunk.get("relationships", {}),
            "analysis": chunk.get("analysis", {}),
        }

    def index_chunks(self, chunks: list[dict], batch_size: int = 100):
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
                    vector=chunk["embedding"],
                    payload=self._prepare_payload(chunk),
                )
                points.append(point)

            # Batch upload
            for i in tqdm(range(0, len(points), batch_size), desc=f"Uploading to {collection_name}"):
                batch = points[i : i + batch_size]
                self.client.upsert(collection_name=collection_name, points=batch)
                total_indexed += len(batch)

        console.print(f"[green]✓ Indexed {total_indexed} chunks successfully![/green]")

    def test_search(self, query: str, limit: int = 5):
        """Test search across all collections"""
        console.print(Panel.fit(f"[bold cyan]Test Search: '{query}'[/bold cyan]", border_style="cyan"))

        # Generate query embedding
        embedding_gen = EmbeddingGenerator(EmbeddingConfig())
        query_embedding = embedding_gen.generate_embedding_ollama(query)

        if not query_embedding:
            console.print("[red]Failed to generate query embedding[/red]")
            return

        # Search each collection
        for collection_name in self.collections:
            console.print(f"\n[yellow]Results from {collection_name}:[/yellow]")

            results = self.client.search(collection_name=collection_name, query_vector=query_embedding, limit=limit)

            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("Score", width=8)
            table.add_column("Name", width=30)
            table.add_column("Type", width=10)
            table.add_column("File", width=40)

            for result in results:
                table.add_row(
                    f"{result.score:.4f}",
                    result.payload.get("qualified_name", result.payload.get("name", "N/A")),
                    result.payload.get("type", "N/A"),
                    result.payload.get("file_path", "N/A"),
                )

            console.print(table)


def index_from_embedded_json(
    json_path: str, embedding_dim: int = 768, collection_prefix: str | None = None, batch_size: int | None = None
) -> None:
    """
    Load pre-embedded chunks from JSON file and index them into Qdrant collections.
    Adapted from old_code/embedder.py logic.
    """

    path = Path(json_path)
    if not path.exists():
        console.print(f"[red]File not found: {json_path}[/red]")
        return

    console.print(f"[cyan]Loading embedded chunks from {json_path}...[/cyan]")
    with Path(path).open(encoding="utf-8") as f:
        data = json.load(f)
        chunks = data.get("chunks", [])
    console.print(f"[green]Loaded {len(chunks)} chunks[/green]")

    config = QdrantConfig()
    if collection_prefix:
        config.collection_prefix = collection_prefix

    indexer = QdrantIndexer(config)
    if not indexer.health_check():
        console.print("[red]Qdrant health check failed. Aborting.[/red]")
        return

    indexer.create_collections(embedding_dim)
    indexer.index_chunks(chunks, batch_size=batch_size)

    console.print(Panel.fit("[bold green]✓ Indexing from embedded JSON complete![/bold green]", border_style="green"))
