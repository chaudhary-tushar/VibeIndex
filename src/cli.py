import asyncio
import json
import os
from pathlib import Path

import click
from rich.console import Console
from rich.markdown import Markdown

from src.config import EmbeddingConfig
from src.config import QdrantConfig
from src.embedding.embedder import EmbeddingGenerator
from src.generation import BatchProcessor_2
from src.generation.context_builder import ContextEnricher
from src.generation.context_builder import get_summarized_chunks_ids
from src.generation.context_builder import stats_check
from src.preprocessing import parse_project
from src.preprocessing.chunk import ChunkPreprocessor
from src.retrieval import CodeRAG
from src.retrieval.hybrid_search import setup_hybrid_collection
from src.retrieval.search import index_from_embedded_json

console = Console()


@click.group()
def cli():
    """RAG Code Parser CLI"""


@cli.command()
@click.option("--path", "-p", default=".", help="Project path to ingest")
@click.option("--output", "-o", help="Output file for parsed chunks (JSON)")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
def ingest(path, output, verbose):
    """
    Run the data ingestion pipeline - parse code into chunks.
    """
    click.echo(f"Running code parsing pipeline for: {path}")

    try:
        # Parse the project
        parser = parse_project(path)

        if verbose:
            click.echo(f"Found {len(parser.chunks)} code chunks")
            click.echo(f"Statistics: {dict(parser.stats)}")

        # Save results if output specified
        if output:
            parser.save_results(output)
            click.echo(f"Results saved to: {output}")
        else:
            # Show summary
            click.echo("\nParsed chunks summary:")
            for i, chunk in enumerate(parser.chunks[:10]):  # Show first 10
                click.echo(f"  {i + 1}. {chunk.name} ({chunk.type}) - {chunk.file_path}")
            if len(parser.chunks) > 10:
                click.echo(f"  ... and {len(parser.chunks) - 10} more")

        click.echo("\n✅ Code parsing complete!")

    except Exception as e:
        click.echo(f"❌ Error during parsing: {e}", err=True)
        return 1


@cli.command()
@click.option("--input", "-i", required=True, help="Input JSON file with chunks")
@click.option("--output", "-o", help="Output file for preprocessed chunks (JSON)")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
def preprocess(input, output, verbose):
    """
    Preprocess code chunks (deduplication, enhancement).
    """
    click.echo(f"Preprocessing chunks from: {input}")

    try:
        # Load chunks
        with Path(input).open(encoding="utf-8") as f:
            data = json.load(f)

        # Handle both direct chunk arrays and wrapped formats
        if "chunks" in data:
            chunks = data["chunks"]
        else:
            chunks = data  # Assume it's directly the chunks array

        # Preprocess
        preprocessor = ChunkPreprocessor()
        processed_chunks = preprocessor.process(chunks)

        if verbose:
            click.echo(f"Processed {len(processed_chunks)} chunks")
            click.echo(f"Stats: {preprocessor.stats}")

        # Save results
        if output:
            output_data = {
                "project_path": data.get("project_path", ""),
                "total_chunks": len(processed_chunks),
                "statistics": data.get("statistics", {}),
                "chunks": processed_chunks,
            }
            Path(output).parent.mkdir(parents=True, exist_ok=True)
            with Path(output).open("w", encoding="utf-8") as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            click.echo(f"Results saved to: {output}")
        else:
            # Show summary
            click.echo(f"Processed {len(processed_chunks)} chunks successfully")
            click.echo(f"Duplicates removed: {preprocessor.stats['duplicates']}")
            click.echo(f"Too large (skipped): {preprocessor.stats['too_large']}")

        click.echo("\n✅ Chunk preprocessing complete!")

    except Exception as e:
        click.echo(f"❌ Error during preprocessing: {e}", err=True)
        return 1


@cli.command()
@click.option("--input", "-i", required=True, help="Input JSON file with chunks")
@click.option("--output", "-o", help="Output file for embedded chunks (JSON)")
@click.option("--model-url", default="http://localhost:12434/engines/llama.cpp/v1", help="Embedding model URL")
@click.option("--model-name", default="ai/embeddinggemma", help="Embedding model name")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
def embed(input, output, model_url, model_name, verbose):
    """
    Generate embeddings for code chunks.
    """
    click.echo(f"Generating embeddings for chunks from: {input}")
    click.echo(f"Model: {model_name} @ {model_url}")

    try:
        # Load chunks
        with Path(input).open(encoding="utf-8") as f:
            data = json.load(f)

        # Handle both direct chunk arrays and wrapped formats
        if "chunks" in data:
            chunks = data["chunks"]
        else:
            chunks = data  # Assume it's directly the chunks array

        # Generate embeddings
        config = EmbeddingConfig(model_url=model_url, model_name=model_name)
        embedder = EmbeddingGenerator(config)
        embedded_chunks = embedder.generate_all(chunks)

        if verbose:
            click.echo(f"Embedded {len(embedded_chunks)} chunks")
            click.echo(f"Success: {embedder.stats['success']}, Failed: {embedder.stats['failed']}")

        # Save results
        if output:
            output_data = {
                "project_path": data.get("project_path", ""),
                "total_chunks": len(embedded_chunks),
                "statistics": data.get("statistics", {}),
                "chunks": embedded_chunks,
            }
            Path(output).parent.mkdir(parents=True, exist_ok=True)
            with Path(output).open("w", encoding="utf-8") as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            click.echo(f"Results saved to: {output}")
        else:
            # Show summary
            click.echo(f"Embedded {len(embedded_chunks)} chunks successfully")
            click.echo(f"Success: {embedder.stats['success']}, Failed: {embedder.stats['failed']}")

        click.echo("\n✅ Embedding generation complete!")

    except Exception as e:
        click.echo(f"❌ Error during embedding: {e}", err=True)
        return 1


@cli.command()
@click.option("--input", "-i", required=True, help="Input JSON file with embedded chunks")
@click.option("--output", "-o", help="Output file for indexed chunks (JSON)")
@click.option("--host", default="localhost", help="Qdrant host")
@click.option("--port", default=6333, type=int, help="Qdrant port")
@click.option("--collection-prefix", default="tipsy", help="Collection prefix")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
def index(input, output, host, port, collection_prefix, verbose):
    """
    Index embedded chunks in Qdrant vector database.
    """
    click.echo(f"Indexing chunks from: {input}")
    click.echo(f"Qdrant: {host}:{port}, Prefix: {collection_prefix}")

    try:
        # Load chunks
        with Path(input).open(encoding="utf-8") as f:
            data = json.load(f)

        # Handle both direct chunk arrays and wrapped formats
        if "chunks" in data:
            chunks = data["chunks"]
        else:
            chunks = data  # Assume it's directly the chunks array

        # Index in Qdrant
        qdrant_config = QdrantConfig(host=host, port=port, collection_prefix=collection_prefix)
        indexer = QdrantIndexer(qdrant_config)

        # Create collections (assuming 768-dim embeddings, adjust as needed)
        indexer.create_collections(embedding_dim=768)

        # Index chunks
        indexer.index_chunks(chunks)

        if verbose:
            click.echo(f"Indexed {len(chunks)} chunks in collections: {list(indexer.collections.keys())}")

        click.echo("\n✅ Indexing complete!")

    except Exception as e:
        click.echo(f"❌ Error during indexing: {e}", err=True)
        return 1


@cli.command()
@click.option("--input", "-i", required=True, help="Input JSON file with pre-embedded chunks")
@click.option("--output", "-o", help="Output file for indexed chunks (JSON)")
@click.option("--embedding-dim", "-d", default=768, type=int, help="Embedding dimension")
@click.option("--collection-prefix", "-c", default="default", help="Collection prefix")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
def index_embedded(input, output, embedding_dim, collection_prefix, verbose):
    console.print(f"[bold blue]Indexing pre-embedded JSON from:[/bold blue] {input}")
    console.print(f"Embedding dim: {embedding_dim}, Prefix: {collection_prefix}")
    try:
        # Load chunks
        with Path(input).open(encoding="utf-8") as f:
            data = json.load(f)

        # Handle both direct chunk arrays and wrapped formats
        if "chunks" in data:
            chunks = data["chunks"]
        else:
            chunks = data  # Assume it's directly the chunks array

        # Index chunks
        indexed_chunks = index_from_embedded_json(chunks, embedding_dim, collection_prefix)

        if verbose:
            console.print(f"Indexed {len(indexed_chunks)} chunks")

        # Save results
        if output:
            output_data = {
                "project_path": data.get("project_path", ""),
                "total_chunks": len(indexed_chunks),
                "statistics": data.get("statistics", {}),
                "chunks": indexed_chunks,
            }
            Path(output).parent.mkdir(parents=True, exist_ok=True)
            with Path(output).open("w", encoding="utf-8") as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            console.print(f"Results saved to: {output}")
        else:
            console.print(f"Indexed {len(indexed_chunks)} chunks successfully")

        console.print("[bold green]✅ index_embedded complete![/bold green]")

    except Exception as e:
        console.print(f"[bold red]❌ Error: {e}[/bold red]")
        return 1


@cli.command()
@click.option("--collection-name", "-n", required=True, help="Name of the hybrid collection")
@click.option("--chunks-path", "-p", required=True, help="Path to chunks JSON file")
def hybrid_setup(collection_name, chunks_path):
    console.print(f"[bold blue]Setting up hybrid collection:[/bold blue] {collection_name}")
    console.print(f"Using chunks: {chunks_path}")
    try:
        setup_hybrid_collection(collection_name, chunks_path)
        console.print("[bold green]✅ hybrid_setup complete![/bold green]")
    except Exception as e:
        console.print(f"[bold red]❌ Error: {e}[/bold red]")
        return 1


@cli.command()
@click.option("--input", "-i", required=True, help="Input JSON file with chunks")
@click.option("--output", "-o", required=True, help="Output enriched JSON file")
@click.option("--symbol-index", "-s", help="Optional symbol index JSON file")
@click.option("--model", "-m", default="ai/llama3.2:latest", help="LLM model to use")
def enrich(input, output, symbol_index, model):
    """
    Enrich code chunks with AI-generated context summaries.
    """
    click.echo(f"Enriching chunks from {input} to {output}")

    try:
        # Load chunks
        with Path(input).open(encoding="utf-8") as f:
            data = json.load(f)

        # Handle both direct chunk arrays and wrapped formats
        if "chunks" in data:
            chunks = data["chunks"]
        else:
            chunks = data  # Assume it's directly the chunks array

        summarized_ids = set(get_summarized_chunks_ids())
        filtered_chunks = [item for item in chunks if item.get("id") not in summarized_ids]
        stats_check(len(summarized_ids), len(chunks), len(filtered_chunks))

        # Load symbol index (optional)
        symbol_index_data = None
        if symbol_index and Path(symbol_index).exists():
            with Path(symbol_index).open(encoding="utf-8") as f:
                symbol_index_data = json.load(f)

        # Set model if provided
        if model:
            os.environ["LLM_MODEL"] = model

        # Enrich
        enricher = ContextEnricher(chunks=filtered_chunks, symbol_index=symbol_index_data)
        enriched_chunks = asyncio.run(enricher.enrich())

        # Save with same structure as input
        output_data = {
            "project_path": data.get("project_path", ""),
            "total_chunks": len(enriched_chunks),
            "statistics": data.get("statistics", {}),
            "chunks": enriched_chunks,
        }

        Path(output).parent.mkdir(parents=True, exist_ok=True)
        with Path(output).open("w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        click.echo(f"✅ Enriched {len(enriched_chunks)} chunks. Saved to {output}")

    except Exception as e:
        click.echo(f"❌ Error during enrichment: {e}", err=True)
        return 1


@cli.command()
@click.option(
    "--input", "-i", required=True, help="Input .txt file or directory of .txt files with prompts (one per line)"
)
@click.option("--output", "-o", help="Output file (jsonl/json/csv)")
@click.option("--fmt", "-f", default="jsonl", type=click.Choice(["jsonl", "json", "csv"]), help="Output format")
@click.option("--delay", default=0.2, type=float, help="Delay (seconds) between requests")
@click.option("--model", "-m", help="Override LLM model name")
def batch(input_path, output, fmt, delay, model):
    """
    Batch process prompts through LLM.
    """
    click.echo(f"🔄 Batch processing prompts from: {input_path}")
    if model:
        os.environ["LLM_MODEL"] = model
    processor = BatchProcessor_2(delay=delay)
    prompts = processor.load_prompts(input_path)
    processor.process_prompts(prompts, output, fmt)
    click.echo("\n✅ Batch processing complete!")


@cli.command()
@click.option("--host", default="0.0.0.0", help="Host to bind to")
@click.option("--port", default=8000, type=int, help="Port to bind to")
@click.option("--reload", is_flag=True, help="Enable auto-reload for development")
def api(host, port, reload):
    """
    Run the FastAPI server for code parsing.
    """
    from src.app import main as app_main

    app_main(host, port, reload)


@cli.command()
@click.option("--qdrant-host", default="localhost", help="Qdrant host")
@click.option("--qdrant-port", default=6333, type=int, help="Qdrant port")
@click.option("--collection-prefix", default="tipsy", help="Collection prefix")
@click.option("--llm-url", default="http://localhost:12434/", help="LLM API URL")
@click.option("--llm-model", default="ai/llama3.2:latest", help="LLM model")
@click.option("--embedding-model", default="ai/embeddinggemma", help="Embedding model")
def rag(qdrant_host, qdrant_port, collection_prefix, llm_url, llm_model, embedding_model):
    """
    Interactive RAG query CLI using CodeRAG
    """
    from src.config import EmbeddingConfig
    from src.config import QdrantConfig

    embedding_config = EmbeddingConfig(
        model_url=f"{llm_url}engines/llama.cpp/v1", model_name=embedding_model, embedding_dim=768, batch_size=32
    )
    qdrant_config = QdrantConfig(host=qdrant_host, port=qdrant_port, collection_prefix=collection_prefix)

    rag_system = CodeRAG(
        embedding_config=embedding_config, qdrant_config=qdrant_config, llm_api_url=llm_url, llm_model_name=llm_model
    )

    console.print("[bold green]CodeRAG Interactive CLI Ready![/bold green]")
    while True:
        try:
            query = input("\n❓ Your question about the codebase: ").strip()
            if not query or query.lower() in {"quit", "exit"}:
                break

            console.print("[cyan]🔍 Searching codebase...[/cyan]")
            answer = rag_system.query_codebase(query)

            console.print("\n[bold green]🤖 Answer:[/bold green]")
            console.print(Markdown(answer))

        except KeyboardInterrupt:
            console.print("\n[red]Goodbye![/red]")
            break
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")


@cli.command()
@click.option("--qdrant-host", default="localhost", help="Qdrant host")
@click.option("--qdrant-port", default=6333, type=int, help="Qdrant port")
@click.option("--collection-name", default="default", help="Collection name to search in")
@click.option("--embedding-model", default="ai/embeddinggemma", help="Embedding model")
def advanced_rag(qdrant_host, qdrant_port, collection_name, embedding_model):
    """
    Advanced RAG query CLI following rag2.mermaid architecture with reranking
    """
    from qdrant_client import QdrantClient

    from src.config import EmbeddingConfig
    from src.config import QdrantConfig
    from src.retrieval.complete_retrieval_system import CompleteRetrievalSystem
    from src.retrieval.hybrid_search import HybridSearchConfig
    from src.retrieval.hybrid_search import HybridSearchEngine

    # Setup configurations
    embedding_config = EmbeddingConfig(
        model_url="http://localhost:12434/engines/llama.cpp/v1",
        model_name=embedding_model,
        embedding_dim=768,
        batch_size=32,
    )
    qdrant_config = QdrantConfig(  # noqa: F841
        host=qdrant_host, port=qdrant_port, collection_prefix=""
    )

    # Initialize components
    qdrant_client = QdrantClient(host=qdrant_host, port=qdrant_port)
    embedding_generator = EmbeddingGenerator(embedding_config)

    hybrid_search_config = HybridSearchConfig()
    hybrid_search_engine = HybridSearchEngine(
        qdrant_client=qdrant_client, embedding_generator=embedding_generator, config=hybrid_search_config
    )

    # Create the complete retrieval system following rag2.mermaid architecture
    retrieval_system = CompleteRetrievalSystem(
        hybrid_search_engine=hybrid_search_engine, embedding_generator=embedding_generator
    )

    console.print("[bold blue]Advanced RAG (rag2.mermaid) Interactive CLI Ready![/bold blue]")
    console.print("[bold]Features:[/bold] Hybrid Search + Initial Candidate Selection + Reranking + Quality Assurance")

    while True:
        try:
            query = input("\n❓ Your question about the codebase: ").strip()
            if not query or query.lower() in {"quit", "exit"}:
                break

            console.print("[cyan]🔍 Retrieving with advanced rag2.mermaid pipeline...[/cyan]")
            results = retrieval_system.retrieve(query=query, collection_name=collection_name, top_k=5)

            console.print(f"\n[bold green]Found {len(results)} relevant code snippets:[/bold green]")
            for i, result in enumerate(results, 1):
                console.print(f"\n[i]{i}. [bold]{result.get('qualified_name', result.get('name', 'Unknown'))}[/bold]")
                console.print(f"   Score: {result.get('score', 'N/A')}")
                console.print(f"   File: {result.get('file_path', 'N/A')}")
                console.print(f"   Preview: {result.get('code', '')[:100]}...")

        except KeyboardInterrupt:
            console.print("\n[red]Goodbye![/red]")
            break
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")


def main():
    cli()


if __name__ == "__main__":
    main()
