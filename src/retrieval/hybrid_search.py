"""
Hybrid Search Implementation for Qdrant
Combines Dense Vectors (semantic) + Sparse Vectors (BM25 keyword)
Migrated from old_code
"""

import pathlib
import re
import time
from dataclasses import dataclass

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance
from qdrant_client.models import FieldCondition
from qdrant_client.models import Filter
from qdrant_client.models import FusionQuery
from qdrant_client.models import MatchValue
from qdrant_client.models import PointStruct
from qdrant_client.models import Prefetch
from qdrant_client.models import SparseIndexParams
from qdrant_client.models import SparseVector
from qdrant_client.models import SparseVectorParams
from qdrant_client.models import VectorParams
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn
from rich.progress import Progress
from rich.progress import SpinnerColumn
from rich.progress import TaskProgressColumn
from rich.progress import TextColumn
from rich.table import Table

from src.config.embedding_config import EmbeddingConfig
from src.embedding.embedder import EmbeddingGenerator

console = Console()


@dataclass
class HybridSearchConfig:
    """Configuration for hybrid search"""

    # Dense vector (semantic) weight
    dense_weight: float = 0.7

    # Sparse vector (keyword) weight
    sparse_weight: float = 0.3

    # BM25 parameters
    bm25_k1: float = 1.2  # Term frequency saturation
    bm25_b: float = 0.75  # Length normalization

    # Retrieval parameters
    top_k_dense: int = 50  # Retrieve more candidates for reranking
    top_k_sparse: int = 30
    final_top_k: int = 10

    # Score threshold
    min_score: float = 0.5


class BM25SparseEncoder:
    """
    Simple BM25 sparse vector encoder
    Converts text to sparse vectors for keyword matching
    """

    def __init__(self, k1: float = 1.2, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.vocab = {}
        self.idf = {}
        self.avg_doc_len = 0

    def tokenize(self, text: str) -> list[str]:
        """Simple tokenization"""
        # Convert to lowercase, split on non-alphanumeric
        text = text.lower()
        return re.findall(r"\b\w+\b", text)

    def build_vocab_from_collection(self, client: QdrantClient, collection_name: str):
        """Build vocabulary and IDF from existing collection"""
        console.print(f"[cyan]Building BM25 vocabulary from {collection_name}...[/cyan]")

        # Get total count
        collection_info = client.get_collection(collection_name)
        total_points = collection_info.points_count
        print(total_points)
        # Scroll through all documents
        offset = None
        doc_count = 0
        total_length = 0
        term_doc_freq = {}

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console,
        ) as progress:
            # Main document processing task
            doc_task = progress.add_task(f"[red]Processing documents...[/red]\n", total=total_points)

            while True:
                result = client.scroll(
                    collection_name=collection_name, limit=100, offset=offset, with_payload=True, with_vectors=False
                )

                points, next_offset = result

                if not points:
                    print("here", end="*")
                    break

                for point in points:
                    # Combine relevant text fields
                    text_parts = [
                        point.payload.get("code", ""),
                        point.payload.get("docstring", ""),
                        point.payload.get("qualified_name", ""),
                        point.payload.get("signature", ""),
                    ]
                    text = " ".join(filter(None, text_parts))

                    tokens = self.tokenize(text)
                    doc_count += 1
                    total_length += len(tokens)

                    # Track term document frequency
                    unique_tokens = set(tokens)
                    for token in unique_tokens:
                        term_doc_freq[token] = term_doc_freq.get(token, 0) + 1
                progress.update(doc_task, advance=len(points))
                if not next_offset:
                    print("here", end="*")
                    break
                offset = next_offset
                time.sleep(1)

            print("here")
            if term_doc_freq:
                idf_task = progress.add_task(
                    f"[green]Calculating IDF for {len(term_doc_freq)} terms...", total=len(term_doc_freq)
                )
                # Calculate IDF
                self.avg_doc_len = total_length / doc_count if doc_count > 0 else 0

                for term, df in term_doc_freq.items():
                    self.vocab[term] = len(self.vocab)
                    self.idf[term] = np.log((doc_count - df + 0.5) / (df + 0.5) + 1.0)
                    progress.update(idf_task, advance=1)

        console.print(f"[green]✓ Vocabulary built: {len(self.vocab)} terms from {doc_count} documents[/green]")

    def encode(self, text: str) -> SparseVector:
        """Encode text to sparse vector"""
        tokens = self.tokenize(text)
        doc_len = len(tokens)

        # Calculate term frequencies
        tf = {}
        for token in tokens:
            tf[token] = tf.get(token, 0) + 1

        # Calculate BM25 scores
        indices = []
        values = []

        for term, freq in tf.items():
            if term in self.vocab:
                # BM25 score
                idf = self.idf.get(term, 0)
                numerator = freq * (self.k1 + 1)
                denominator = freq + self.k1 * (1 - self.b + self.b * (doc_len / self.avg_doc_len))
                score = idf * (numerator / denominator)

                indices.append(self.vocab[term])
                values.append(float(score))

        return SparseVector(indices=indices, values=values)

    def build_vocab_from_texts(self, texts: list[str]):
        """Build vocabulary and IDF from raw texts (not from Qdrant)"""
        console.print(f"[cyan]Building BM25 vocabulary from {len(texts)} documents...[/cyan]")

        doc_count = len(texts)
        total_length = 0
        term_doc_freq = {}

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console,
        ) as progress:
            doc_task = progress.add_task("[red]Processing documents...[/red]", total=doc_count)

            for text in texts:
                tokens = self.tokenize(text)
                total_length += len(tokens)
                unique_tokens = set(tokens)
                for token in unique_tokens:
                    term_doc_freq[token] = term_doc_freq.get(token, 0) + 1
                progress.update(doc_task, advance=1)

            if term_doc_freq:
                idf_task = progress.add_task(
                    f"[green]Calculating IDF for {len(term_doc_freq)} terms...", total=len(term_doc_freq)
                )
                self.avg_doc_len = total_length / doc_count if doc_count > 0 else 0

                for term, df in term_doc_freq.items():
                    self.vocab[term] = len(self.vocab)
                    self.idf[term] = np.log((doc_count - df + 0.5) / (df + 0.5) + 1.0)
                    progress.update(idf_task, advance=1)

        console.print(f"[green]✓ Vocabulary built: {len(self.vocab)} terms from {doc_count} documents[/green]")


class HybridSearchEngine:
    """
    Advanced hybrid search combining dense and sparse vectors
    """

    def __init__(self, qdrant_client: QdrantClient, embedding_generator, config: HybridSearchConfig = None):
        self.client = qdrant_client
        self.embedding_gen = embedding_generator
        self.config = config or HybridSearchConfig()
        self.bm25_encoder = BM25SparseEncoder(k1=self.config.bm25_k1, b=self.config.bm25_b)

    def create_hybrid_collection(self, collection_name: str, dense_dim: int = 768):
        """Create a new collection with named dense + sparse vectors"""
        console.print(f"[cyan]Creating hybrid collection '{collection_name}'...[/cyan]")

        self.client.recreate_collection(
            collection_name=collection_name,
            vectors_config={"text-dense": VectorParams(size=dense_dim, distance=Distance.COSINE)},
            sparse_vectors_config={"text-sparse": SparseVectorParams(index=SparseIndexParams(on_disk=False))},
        )
        console.print(f"[green]✓ Hybrid collection created[/green]")

    def reindex_with_sparse_vectors(self, collection_name: str, chunks: list[dict]):
        """
        Add sparse vectors to existing points
        """
        console.print(f"[cyan]Adding sparse vectors to {len(chunks)} points...[/cyan]")

        from tqdm import tqdm

        batch_size = 100
        for i in tqdm(range(0, len(chunks), batch_size), desc="Reindexing"):
            batch = chunks[i : i + batch_size]
            points = []

            for chunk in batch:
                # Generate sparse vector
                text_parts = [
                    chunk.get("code", ""),
                    chunk.get("docstring", ""),
                    chunk.get("qualified_name", ""),
                ]
                text = " ".join(filter(None, text_parts))
                sparse_vector = self.bm25_encoder.encode(text)

                # Create point with both dense and sparse vectors
                point = PointStruct(
                    id=chunk["id"],  # Use chunk ID as string (UUID)
                    vector={
                        "text-dense": chunk["embedding"],  # Existing dense vector
                        "text-sparse": sparse_vector,
                    },
                    payload=self._prepare_payload(chunk),
                )
                points.append(point)

            # Upsert batch
            self.client.upsert(collection_name=collection_name, points=points)

        console.print(f"[green]✓ Reindexing complete[/green]")

    def _prepare_payload(self, chunk: dict) -> dict:
        """Prepare payload (remove embedding)"""
        payload = chunk.copy()
        payload.pop("embedding", None)
        payload.pop("embedding_text", None)
        return payload

    def hybrid_search(
        self, collection_name: str, query_text: str, filters: Filter | None = None, limit: int | None = None
    ) -> list[dict]:
        """
        Perform hybrid search (dense + sparse vectors)
        """
        limit = limit or self.config.final_top_k

        # 1. Generate dense query vector
        query_embedding = self.embedding_gen.generate_embedding(query_text)

        if not query_embedding:
            console.print("[red]Failed to generate query embedding[/red]")
            return []

        # 2. Generate sparse query vector
        query_sparse = self.bm25_encoder.encode(query_text)

        # 3. Perform hybrid search using RRF (Reciprocal Rank Fusion)
        try:
            results = self.client.query_points(
                collection_name=collection_name,
                prefetch=[
                    Prefetch(query=query_embedding, using="text-dense", limit=self.config.top_k_dense, filter=filters),
                    Prefetch(query=query_sparse, using="text-sparse", limit=self.config.top_k_sparse, filter=filters),
                ],
                query=FusionQuery(fusion="rrf"),  # Reciprocal Rank Fusion
                limit=limit,
            )

        except Exception as e:
            console.print(f"[yellow]Hybrid search not available, falling back to dense-only: {e}[/yellow]")
            # Fallback to dense-only search
            return self._dense_search_fallback(collection_name, query_embedding, filters, limit)
        else:
            return results.points

    def _dense_search_fallback(
        self, collection_name: str, query_vector: list[float], filters: Filter | None, limit: int
    ):
        """Fallback to dense-only search"""
        return self.client.search(
            collection_name=collection_name,
            query_vector=("text-dense", query_vector),
            query_filter=filters,
            limit=limit,
        )

    def compare_search_methods(self, collection_name: str, query: str, limit: int = 5):
        """
        Compare dense-only vs hybrid search results
        """
        console.print(
            Panel.fit(f"[bold cyan]Comparing Search Methods[/bold cyan]\nQuery: '{query}'", border_style="cyan")
        )

        # 1. Dense-only search
        query_embedding = self.embedding_gen.generate_embedding(query)

        console.print("\n[yellow]═══ Dense-Only Search (Vector Similarity) ═══[/yellow]")
        dense_results = self.client.search(
            collection_name=collection_name, query_vector=("text-dense", query_embedding), limit=limit
        )

        self._display_results(dense_results, "Dense")

        # 2. Hybrid search
        console.print("\n[yellow]═══ Hybrid Search (Vector + BM25 Keyword) ═══[/yellow]")
        hybrid_results = self.hybrid_search(collection_name, query, limit=limit)

        self._display_results(hybrid_results, "Hybrid")

        # 3. Analysis
        self._compare_results(dense_results, hybrid_results)

    def _display_results(self, results, method_name: str):
        """Display search results in a table"""
        table = Table(show_header=True, header_style="bold magenta", title=f"{method_name} Search Results")
        table.add_column("Rank", width=6)
        table.add_column("Score", width=8)
        table.add_column("Name", width=30)
        table.add_column("Type", width=10)
        table.add_column("File", width=40)

        for i, result in enumerate(results, 1):
            table.add_row(
                str(i),
                f"{result.score:.4f}",
                result.payload.get("qualified_name", result.payload.get("name", "N/A"))[:30],
                result.payload.get("type", "N/A"),
                result.payload.get("file_path", "N/A")[:40],
            )

        console.print(table)

    def _compare_results(self, dense_results, hybrid_results):
        """Compare and analyze differences"""
        console.print("\n[cyan]═══ Analysis ═══[/cyan]")

        dense_ids = [r.id for r in dense_results]
        hybrid_ids = [r.id for r in hybrid_results]

        overlap = len(set(dense_ids) & set(hybrid_ids))
        overlap_pct = (overlap / len(dense_ids)) * 100 if dense_ids else 0

        console.print(f"  • Overlap: {overlap}/{len(dense_ids)} results ({overlap_pct:.1f}%)")
        console.print(f"  • Unique to Dense: {len(set(dense_ids) - set(hybrid_ids))}")
        console.print(f"  • Unique to Hybrid: {len(set(hybrid_ids) - set(dense_ids))}")

        # Show what hybrid found that dense didn't
        if set(hybrid_ids) - set(dense_ids):
            console.print("\n[green]Hybrid search found additional relevant results:[/green]")
            for result in hybrid_results:
                if result.id not in dense_ids:
                    console.print(f"  + {result.payload.get('qualified_name', result.payload.get('name'))}")

    def advanced_filtered_search(  # noqa: PLR0913, PLR0917
        self,
        collection_name: str,
        query: str,
        language: str | None = None,
        chunk_type: str | None = None,
        min_complexity: str | None = None,
        file_pattern: str | None = None,
        limit: int = 10,
    ) -> list[dict]:
        """
        Advanced search with multiple filters
        """
        must_conditions = []

        if language:
            must_conditions.append(FieldCondition(key="language", match=MatchValue(value=language)))

        if chunk_type:
            must_conditions.append(FieldCondition(key="type", match=MatchValue(value=chunk_type)))

        if min_complexity:
            must_conditions.append(FieldCondition(key="complexity", range={"gte": min_complexity}))

        if file_pattern:
            must_conditions.append(FieldCondition(key="file_path", match=MatchValue(value=file_pattern)))

        filters = Filter(must=must_conditions) if must_conditions else None

        return self.hybrid_search(collection_name, query, filters=filters, limit=limit)


def setup_hybrid_collection(collection_name: str, chunks_path: str):
    """
    Function to setup hybrid collection for use in main.py CLI
    """
    client = QdrantClient(host="localhost", port=6333)
    embedding_gen = EmbeddingGenerator(EmbeddingConfig())

    hybrid_engine = HybridSearchEngine(
        qdrant_client=client, embedding_generator=embedding_gen, config=HybridSearchConfig(final_top_k=10)
    )

    # Load chunks
    import json

    with pathlib.Path(chunks_path).open("r", encoding="utf-8") as f:
        data = json.load(f)
        chunks = data.get("chunks", [])

    # Build BM25 vocab from raw texts
    console.print("[cyan]Building BM25 vocabulary from chunks...[/cyan]")
    all_texts = []
    for chunk in chunks:
        text_parts = [
            chunk.get("code", ""),
            chunk.get("docstring", ""),
            chunk.get("qualified_name", ""),
        ]
        text = " ".join(filter(None, text_parts))
        all_texts.append(text)
    hybrid_engine.bm25_encoder.build_vocab_from_texts(all_texts)

    # Create hybrid collection
    hybrid_engine.create_hybrid_collection(collection_name, dense_dim=768)

    # Reindex all chunks
    hybrid_engine.reindex_with_sparse_vectors(collection_name, chunks)

    console.print(
        Panel.fit(
            "[bold green]✓ Hybrid Search Enabled![/bold green]\n"
            "Your collection now supports both semantic and keyword search.",
            border_style="green",
        )
    )


def main():
    client = QdrantClient(host="localhost", port=6333)
    embedding_gen = EmbeddingGenerator(EmbeddingConfig())

    hybrid_engine = HybridSearchEngine(
        qdrant_client=client, embedding_generator=embedding_gen, config=HybridSearchConfig(final_top_k=10)
    )

    collection_name = input("\nEnter new collection name for hybrid search: ").strip()

    # Load chunks
    import json

    chunks_file = "parsed_chunks_tipsy_2_embedded.json"
    with pathlib.Path(chunks_file).open("r", encoding="utf-8") as f:
        data = json.load(f)
        chunks = data.get("chunks", [])

    # Build BM25 vocab from raw texts
    console.print("[cyan]Building BM25 vocabulary from chunks...[/cyan]")
    all_texts = []
    for chunk in chunks:
        text_parts = [
            chunk.get("code", ""),
            chunk.get("docstring", ""),
            chunk.get("qualified_name", ""),
        ]
        text = " ".join(filter(None, text_parts))
        all_texts.append(text)
    hybrid_engine.bm25_encoder.build_vocab_from_texts(all_texts)

    # Create hybrid collection
    hybrid_engine.create_hybrid_collection(collection_name, dense_dim=768)

    # Reindex all chunks
    hybrid_engine.reindex_with_sparse_vectors(collection_name, chunks)

    # Test
    test_query = input("\nEnter test query to compare search methods: ").strip()
    if test_query:
        hybrid_engine.compare_search_methods(collection_name, test_query, limit=5)

    console.print(
        Panel.fit(
            "[bold green]✓ Hybrid Search Enabled![/bold green]\n"
            "Your collection now supports both semantic and keyword search.",
            border_style="green",
        )
    )


if __name__ == "__main__":
    main()
