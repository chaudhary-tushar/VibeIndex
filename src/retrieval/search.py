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
from src.config.settings import settings

# Import embedding generator from correct location
from src.embedding.embedder import EmbeddingGenerator

console = Console()
SMALL_COLLECTION = 100
MEDIUM_COLLECTION = 1000
BATCH_LIMIT = 10


class QdrantIndexer:
    """Enhanced Qdrant vector database integration with advanced collection management"""

    def __init__(self):
        self.config = QdrantConfig()
        self.client = QdrantClient(host=self.config.host, port=self.config.port)
        self.collections = {}
        self.stats = {
            "collections_created": 0,
            "chunks_indexed": 0,
            "indexing_errors": 0,
            "searches_performed": 0,
            "search_time_total": 0.0,
            "search_query_count": 0,
            "average_search_time": 0.0,
            "indexing_time_total": 0.0,
            "average_indexing_time": 0.0,
            "query_success_rate": 0.0,
        }

    def health_check(self) -> bool:
        """Perform health check on Qdrant connection"""
        try:
            # Test basic connectivity
            self.client.get_collections()
        except Exception as e:
            console.print(f"[red]Qdrant health check failed: {e}[/red]")
            return False
        else:
            return True

    def collection_health_check(self, collection_name: str) -> bool:
        """Check health of a specific collection"""
        try:
            collection_info = self.client.get_collection(collection_name)
            if collection_info.status != "green":
                console.print(
                    f"[yellow]Warning: Collection '{collection_name}' has status: {collection_info.status}[/yellow]"
                )
                return False
            console.print(f"[green]✓ Collection '{collection_name}' is healthy[/green]")
        except Exception as e:
            console.print(f"[red]Health check failed for collection '{collection_name}': {e}[/red]")
            return False
        else:
            return True

    def overall_health_check(self) -> bool:
        """Check health of all collections"""
        console.print("[cyan]Performing overall health check...[/cyan]")
        healthy_collections = 0
        total_collections = len(self.collections)

        for collection_name in self.collections:
            if self.collection_health_check(collection_name):
                healthy_collections += 1

        if healthy_collections == total_collections and total_collections > 0:
            console.print(
                f"[green]✓ Overall health check passed: {healthy_collections}/{total_collections} collections healthy[/green]"
            )
            return True
        console.print(
            f"[yellow]Overall health check partially failed: {healthy_collections}/{total_collections} collections healthy[/yellow]"
        )
        return False

    def get_performance_metrics(self) -> dict:
        """Get comprehensive performance metrics for monitoring"""
        total_searches = self.stats["searches_performed"]
        total_queries = self.stats["search_query_count"]
        total_indexing = self.stats["chunks_indexed"]

        if total_searches > 0:
            avg_search_time = self.stats["search_time_total"] / total_searches
            self.stats["average_search_time"] = avg_search_time
        if total_indexing > 0:
            avg_indexing_time = self.stats["indexing_time_total"] / total_indexing
            self.stats["average_indexing_time"] = avg_indexing_time
        if total_queries > 0:
            success_count = total_searches  # Approximation
            self.stats["query_success_rate"] = success_count / total_queries

        # Add collection-specific metrics
        collection_sizes = {}
        for collection_name in self.collections:
            try:
                collection_info = self.client.get_collection(collection_name)
                collection_sizes[collection_name] = {
                    "vectors_count": collection_info.vectors_count,
                    "status": str(collection_info.status),
                    "segments_count": collection_info.segments_count,
                }
            except Exception:
                collection_sizes[collection_name] = {"error": "Could not retrieve metrics"}

        self.stats["collection_sizes"] = collection_sizes
        return self.stats

    def print_performance_report(self):
        """Print a formatted performance report"""
        metrics = self.get_performance_metrics()

        console.print("\n[yellow]════════ Performance Metrics Report ════════[/yellow]")
        console.print(f"[bold]Collections Created:[/bold] {metrics['collections_created']}")
        console.print(f"[bold]Chunks Indexed:[/bold] {metrics['chunks_indexed']}")
        console.print(f"[bold]Indexing Errors:[/bold] {metrics['indexing_errors']}")
        console.print(f"[bold]Searches Performed:[/bold] {metrics['searches_performed']}")
        console.print(f"[bold]Total Search Time:[/bold] {metrics['search_time_total']:.3f}s")
        console.print(f"[bold]Average Search Time:[/bold] {metrics['average_search_time']:.3f}s")
        console.print(f"[bold]Total Indexing Time:[/bold] {metrics['indexing_time_total']:.3f}s")
        console.print(f"[bold]Average Indexing Time:[/bold] {metrics['average_indexing_time']:.3f}s")
        console.print(f"[bold]Query Success Rate:[/bold] {metrics['query_success_rate']:.2%}")

        console.print("\n[bold]Collection Sizes:[/bold]")
        for collection_name, size_info in metrics["collection_sizes"].items():
            if "error" not in size_info:
                console.print(
                    f"  {collection_name}: {size_info['vectors_count']} vectors, {size_info['segments_count']} segments"
                )
            else:
                console.print(f"  {collection_name}: {size_info['error']}")

        console.print("[yellow]═══════════════════════════════════════════════[/yellow]\n")

    def create_collections(self):
        """Create optimized Qdrant collections for different chunk types with enhanced management"""
        embedding_dim = settings.embedding_dim
        collection_configs = {
            f"{self.config.collection_prefix}_functions": {
                "description": "Individual functions and methods",
                "optimization": "High-frequency function lookups",
                "hnsw_m": self.config.hnsw_m,  # Use default m
                "hnsw_ef_construct": self.config.hnsw_ef_construct,  # Use default ef_construct
                "hnsw_ef": self.config.hnsw_ef,  # Use default ef
                "quantization": None,  # No quantization for accuracy
                "optimizers": {
                    "deleted_threshold": self.config.deleted_threshold,
                    "vacuum_min_vector_number": self.config.vacuum_min_vector_number,
                    "default_segment_number": self.config.default_segment_number,
                    "max_segment_size": self.config.max_segment_size,
                    "memmap_threshold": self.config.memmap_threshold,
                    "indexing_threshold": self.config.indexing_threshold,
                },
                # Function-specific fields to index
                "indexed_fields": [
                    "language",
                    "type",
                    "file_path",
                    "qualified_name",
                    "docstring",
                    "signature",
                    "start_line",
                    "end_line",
                    "complexity",
                    "metadata.decorators",
                    "metadata.access_modifier",
                    "dependencies",
                    "context.file_hierarchy",
                    "context.domain_context",
                    "documentation.summary",
                ],
            },
            f"{self.config.collection_prefix}_classes": {
                "description": "Class definitions and structure",
                "optimization": "Class hierarchy and inheritance",
                "hnsw_m": self.config.hnsw_m,  # Use default m
                "hnsw_ef_construct": self.config.hnsw_ef_construct,  # Use default ef_construct
                "hnsw_ef": self.config.hnsw_ef,  # Use default ef
                "quantization": None,  # No quantization for accuracy
                "optimizers": {
                    "deleted_threshold": self.config.deleted_threshold,
                    "vacuum_min_vector_number": self.config.vacuum_min_vector_number,
                    "default_segment_number": self.config.default_segment_number,
                    "max_segment_size": self.config.max_segment_size,
                    "memmap_threshold": self.config.memmap_threshold,
                    "indexing_threshold": self.config.indexing_threshold,
                },
                # Class-specific fields to index
                "indexed_fields": [
                    "language",
                    "type",
                    "file_path",
                    "qualified_name",
                    "docstring",
                    "signature",
                    "start_line",
                    "end_line",
                    "complexity",
                    "metadata.decorators",
                    "metadata.access_modifier",
                    "dependencies",
                    "context.file_hierarchy",
                    "context.domain_context",
                    "relationships.children",
                    "relationships.parents",
                    "documentation.summary",
                ],
            },
            f"{self.config.collection_prefix}_modules": {
                "description": "Module and file-level code",
                "optimization": "File-level context and imports",
                "hnsw_m": self.config.hnsw_m,  # Use default m
                "hnsw_ef_construct": self.config.hnsw_ef_construct,  # Use default ef_construct
                "hnsw_ef": self.config.hnsw_ef,  # Use default ef
                "quantization": None,  # No quantization for accuracy
                "optimizers": {
                    "deleted_threshold": self.config.deleted_threshold,
                    "vacuum_min_vector_number": self.config.vacuum_min_vector_number,
                    "default_segment_number": self.config.default_segment_number,
                    "max_segment_size": self.config.max_segment_size,
                    "memmap_threshold": self.config.memmap_threshold,
                    "indexing_threshold": self.config.indexing_threshold,
                },
                # Module-specific fields to index
                "indexed_fields": [
                    "language",
                    "type",
                    "file_path",
                    "qualified_name",
                    "docstring",
                    "start_line",
                    "end_line",
                    "complexity",
                    "dependencies",
                    "relationships.imports",
                    "relationships.exports",
                    "context.file_hierarchy",
                    "context.domain_context",
                    "documentation.summary",
                ],
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
                    # Create collection with named vectors for semantic and keyword search
                    # Each collection has its specific configuration optimized for the data type
                    self.client.create_collection(
                        collection_name=collection_name,
                        vectors_config={
                            # Named dense vector for semantic similarity search
                            "semantic_dense": VectorParams(
                                size=embedding_dim,
                                distance=self.config.distance_metric,
                                on_disk=self.config.on_disk_vectors,
                                # Configure HNSW parameters for optimized search based on collection type
                                hnsw_config={
                                    "m": config["hnsw_m"],  # Number of edges per node in the index graph
                                    "ef_construct": config[
                                        "hnsw_ef_construct"
                                    ],  # Number of neighbours during index building
                                    "full_scan_threshold": self.config.full_scan_threshold,  # Use full scan for small collections
                                },
                                quantization_config=config["quantization"],  # Quantization setting for this collection
                            ),
                        },
                        sparse_vectors_config={
                            # Named sparse vector for keyword-based search (BM25)
                            "keyword_sparse": SparseVectorParams(
                                index=SparseIndexParams(
                                    on_disk=self.config.on_disk_sparse_vectors  # Configurable on-disk storage
                                )
                            ),
                        },
                        # Optimize performance with collection-specific configuration
                        optimizers_config=config["optimizers"],
                    )

                    # Create essential payload indexes for filtering with on-disk option where appropriate
                    if self.config.enable_payload_index:
                        from qdrant_client.models import IntegerIndexParams
                        from qdrant_client.models import KeywordIndexParams
                        from qdrant_client.models import TextIndexParams

                        # Index common fields for all collections
                        self.client.create_payload_index(
                            collection_name=collection_name,
                            field_name="language",
                            field_schema=KeywordIndexParams(
                                type="keyword",
                                is_tenant=False,  # Not used for multi-tenancy
                                on_disk=self.config.on_disk_vectors,  # Use on_disk setting from config
                            ),
                        )
                        self.client.create_payload_index(
                            collection_name=collection_name,
                            field_name="type",
                            field_schema=KeywordIndexParams(
                                type="keyword", is_tenant=False, on_disk=self.config.on_disk_vectors
                            ),
                        )
                        self.client.create_payload_index(
                            collection_name=collection_name,
                            field_name="file_path",
                            field_schema=KeywordIndexParams(
                                type="keyword", is_tenant=False, on_disk=self.config.on_disk_vectors
                            ),
                        )
                        self.client.create_payload_index(
                            collection_name=collection_name,
                            field_name="qualified_name",
                            field_schema=TextIndexParams(
                                type="text",
                                tokenizer="word",  # Tokenize by words for text search
                                min_token_len=2,  # Minimum token length
                                max_token_len=10,  # Maximum token length
                                on_disk=self.config.on_disk_vectors,  # Use on_disk setting from config
                            ),  # Full-text search on names
                        )
                        # Index numeric fields for range queries
                        self.client.create_payload_index(
                            collection_name=collection_name,
                            field_name="start_line",
                            field_schema=IntegerIndexParams(type="integer", on_disk=self.config.on_disk_vectors),
                        )
                        self.client.create_payload_index(
                            collection_name=collection_name,
                            field_name="end_line",
                            field_schema=IntegerIndexParams(type="integer", on_disk=self.config.on_disk_vectors),
                        )
                        self.client.create_payload_index(
                            collection_name=collection_name,
                            field_name="complexity",
                            field_schema=IntegerIndexParams(type="integer", on_disk=self.config.on_disk_vectors),
                        )
                        # Index quality field for quality-based filtering
                        self.client.create_payload_index(
                            collection_name=collection_name,
                            field_name="embedding_quality",
                            field_schema=KeywordIndexParams(
                                type="keyword", is_tenant=False, on_disk=self.config.on_disk_vectors
                            ),
                        )
                        # Index text content for full-text search on docstrings, code, and signatures
                        self.client.create_payload_index(
                            collection_name=collection_name,
                            field_name="docstring",
                            field_schema=TextIndexParams(
                                type="text",
                                tokenizer="word",  # Tokenize by words
                                min_token_len=2,  # Minimum token length
                                max_token_len=15,  # Maximum token length (allow longer tokens for code)
                                lowercase=True,  # Case-insensitive search
                                on_disk=self.config.on_disk_vectors,  # Use on_disk setting from config
                            ),
                        )
                        self.client.create_payload_index(
                            collection_name=collection_name,
                            field_name="signature",
                            field_schema=TextIndexParams(
                                type="text",
                                tokenizer="word",  # Tokenize by words in function signatures
                                min_token_len=1,  # For signatures, include shorter tokens
                                max_token_len=20,  # Longer max length for complex signatures
                                lowercase=True,  # Case-insensitive search
                                on_disk=self.config.on_disk_vectors,  # Use on_disk setting from config
                            ),
                        )
                        # Index code content with special configuration for code-specific search
                        self.client.create_payload_index(
                            collection_name=collection_name,
                            field_name="code",
                            field_schema=TextIndexParams(
                                type="text",
                                tokenizer="word",  # Tokenize by words for code
                                min_token_len=1,  # Include single-char tokens (e.g., 'i', 'j' in for loops)
                                max_token_len=50,  # Higher max length for longer identifiers
                                lowercase=False,  # Keep case for code search
                                on_disk=self.config.on_disk_vectors,  # Use on_disk setting from config
                            ),
                        )
                        # Index nested fields based on collection-specific needs
                        # File hierarchy in context
                        self.client.create_payload_index(
                            collection_name=collection_name,
                            field_name="context.file_hierarchy",
                            field_schema=TextIndexParams(
                                type="text",
                                tokenizer="word",
                                min_token_len=2,
                                max_token_len=20,
                                lowercase=True,
                                on_disk=self.config.on_disk_vectors,
                            ),
                        )
                        # Domain context in context
                        self.client.create_payload_index(
                            collection_name=collection_name,
                            field_name="context.domain_context",
                            field_schema=TextIndexParams(
                                type="text",
                                tokenizer="word",
                                min_token_len=2,
                                max_token_len=30,
                                lowercase=True,
                                on_disk=self.config.on_disk_vectors,
                            ),
                        )
                        # Access modifier in metadata
                        self.client.create_payload_index(
                            collection_name=collection_name,
                            field_name="metadata.access_modifier",
                            field_schema=KeywordIndexParams(
                                type="keyword", is_tenant=False, on_disk=self.config.on_disk_vectors
                            ),
                        )
                        # Decorators in metadata
                        self.client.create_payload_index(
                            collection_name=collection_name,
                            field_name="metadata.decorators",
                            field_schema=KeywordIndexParams(
                                type="keyword", is_tenant=False, on_disk=self.config.on_disk_vectors
                            ),
                        )
                        # Documentation fields
                        self.client.create_payload_index(
                            collection_name=collection_name,
                            field_name="documentation.summary",
                            field_schema=TextIndexParams(
                                type="text",
                                tokenizer="word",
                                min_token_len=2,
                                max_token_len=20,
                                lowercase=True,
                                on_disk=self.config.on_disk_vectors,
                            ),
                        )

                        # Add collection-specific indexes based on config
                        if collection_name.endswith("_classes"):
                            # Index class-specific fields
                            self.client.create_payload_index(
                                collection_name=collection_name,
                                field_name="relationships.children",
                                field_schema=KeywordIndexParams(
                                    type="keyword", is_tenant=False, on_disk=self.config.on_disk_vectors
                                ),
                            )
                            self.client.create_payload_index(
                                collection_name=collection_name,
                                field_name="relationships.parents",
                                field_schema=KeywordIndexParams(
                                    type="keyword", is_tenant=False, on_disk=self.config.on_disk_vectors
                                ),
                            )
                        elif collection_name.endswith("_modules"):
                            # Index module-specific fields
                            self.client.create_payload_index(
                                collection_name=collection_name,
                                field_name="relationships.imports",
                                field_schema=KeywordIndexParams(
                                    type="keyword", is_tenant=False, on_disk=self.config.on_disk_vectors
                                ),
                            )
                            self.client.create_payload_index(
                                collection_name=collection_name,
                                field_name="relationships.exports",
                                field_schema=KeywordIndexParams(
                                    type="keyword", is_tenant=False, on_disk=self.config.on_disk_vectors
                                ),
                            )
                        elif collection_name.endswith("_functions"):
                            # Index function-specific fields
                            self.client.create_payload_index(
                                collection_name=collection_name,
                                field_name="dependencies",
                                field_schema=KeywordIndexParams(
                                    type="keyword", is_tenant=False, on_disk=self.config.on_disk_vectors
                                ),
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
            if current_size < SMALL_COLLECTION:
                return 50  # Small collections, smaller batches
            if current_size < MEDIUM_COLLECTION:
                return 100  # Medium collections
        except Exception:
            return 100  # Default batch size
        return 200

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

                # Use chunk ID as string (UUID) - Qdrant supports both int and UUID string IDs
                point = PointStruct(
                    id=chunk["id"],  # Use chunk ID as string (UUID)
                    vector={"semantic_dense": chunk["embedding"]},
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
                    if batch_size > BATCH_LIMIT:
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

    def advanced_search(  # noqa: C901
        self,
        query: str,
        limit: int = 10,
        filters: dict | None = None,
        collection_filter: str | None = None,
        quality_threshold: str | None = None,
    ) -> dict[str, list]:
        """Perform advanced search with filtering, quality-based filtering and ranking"""
        console.print(
            Panel.fit(
                f"[bold cyan]Advanced Search: '{query}'[/bold cyan]",
                subtitle=f"Limit: {limit} | Filters: {filters} | Collection: {collection_filter or 'all'} | Quality: {quality_threshold or 'any'}",
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
        if filters or quality_threshold:
            conditions = []

            # Add filters for specified fields
            if filters:
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

            # Add quality-based filtering if specified
            if quality_threshold:
                conditions.append(FieldCondition(key="embedding_quality", match=MatchValue(value=quality_threshold)))

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
                    query_vector=("semantic_dense", query_embedding),
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

    def quality_based_search(self, query: str, limit: int = 10, min_quality: str = "validated") -> dict[str, list]:
        """
        Perform search with quality-based filtering to only return highly validated results
        """
        console.print(
            Panel.fit(
                f"[bold cyan]Quality-Based Search: '{query}'[/bold cyan]",
                subtitle=f"Limit: {limit} | Min Quality: {min_quality}",
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

        # Define quality-based filter
        quality_filters = Filter(must=[FieldCondition(key="embedding_quality", match=MatchValue(value=min_quality))])

        # Search each collection with quality filter
        all_results = {}
        for collection_name in self.collections:
            console.print(f"\n[yellow]Searching {collection_name} with quality filter...[/yellow]")

            try:
                results = self.client.search(
                    collection_name=collection_name,
                    query_vector=("semantic_dense", query_embedding),
                    limit=limit,
                    query_filter=quality_filters,
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
                console.print(f"[red]Quality-based search failed for {collection_name}: {e}[/red]")
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


def index_from_embedded_json(json_path: str, batch_size: int | None = None) -> None:
    """
    Load pre-embedded chunks from JSON file and index them into Qdrant collections.
    Adapted from old_code/embedder.py logic.
    """
    collection_prefix = settings.project_name
    embedding_dim = settings.embedding_dim
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
