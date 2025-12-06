import asyncio
import json
import os
from pathlib import Path

import click
import uvicorn
from fastapi import FastAPI
from fastapi import HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from rich.console import Console
from rich.markdown import Markdown

from src.config import EmbeddingConfig
from src.config import QdrantConfig
from src.config import settings
from src.embedding.embedder import EmbeddingGenerator
from src.generation import BatchProcessor
from src.generation.context_builder import ContextEnricher
from src.generation.context_builder import get_summarized_chunks_ids
from src.generation.context_builder import stats_check

# Import our preprocessing modules
from src.preprocessing import parse_file
from src.preprocessing import parse_project
from src.preprocessing.preprocessor import ChunkPreprocessor
from src.retrieval import CodeRAG_2
from src.retrieval.hybrid_search import setup_hybrid_collection
from src.retrieval.search import QdrantIndexer
from src.retrieval.search import index_from_embedded_json

console = Console()
app = FastAPI(title="RAG Code Parser API", description="API for parsing code into chunks for RAG indexing")


@app.get("/")
def read_root():
    """Serve a simple HTML interface or redirect to docs"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>RAG Code Parser API</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .container { max-width: 800px; margin: 0 auto; }
            .endpoint { margin: 20px 0; padding: 10px; border: 1px solid #ddd; border-radius: 5px; }
            .method { font-weight: bold; color: #2E86AB; }
            input, button { padding: 8px; margin: 5px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>RAG Code Parser API</h1>
            <p>Use this API to parse code repositories into chunks for RAG indexing.</p>

            <div class="endpoint">
                <span class="method">POST</span> <code>/parse-project</code>
                <p>Parse an entire project directory</p>
                <p>Body: <code>{"project_path": "/path/to/project"}</code></p>
            </div>

            <div class="endpoint">
                <span class="method">GET</span> <code>/parse-file?file_path=/path/to/file.py</code>
                <p>Parse a single file</p>
            </div>

            <div class="endpoint">
                <span class="method">GET</span> <code>/docs</code>
                <p>View API documentation</p>
            </div>

            <h3>Quick Test</h3>
            <form onsubmit="parseCurrentProject()">
                <button type="submit">Parse Current Project (.)</button>
            </form>
            <div id="result"></div>
        </div>

        <script>
            function parseCurrentProject() {
                fetch('/parse-project', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ project_path: '.' })
                })
                .then(response => response.json())
                .then(data => {
                    document.getElementById('result').innerHTML =
                        '<pre>' + JSON.stringify(data, null, 2) + '</pre>';
                })
                .catch(error => {
                    document.getElementById('result').innerHTML =
                        '<p style="color: red;">Error: ' + error + '</p>';
                });
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@app.post("/parse-project")
async def api_parse_project(request: dict):
    """Parse an entire project directory"""
    project_path = request.get("project_path", ".")

    if not Path(project_path).exists():
        raise HTTPException(status_code=404, detail=f"Project path not found: {project_path}")

    try:
        # Parse the project
        parser = parse_project(project_path)

        # Return summary and chunks
        return {
            "project_path": str(parser.project_path),
            "total_chunks": len(parser.chunks),
            "statistics": dict(parser.stats),
            "chunks": [chunk.to_dict() for chunk in parser.chunks[:100]],  # Limit first 100 chunks
            "limited": len(parser.chunks) > 100,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Parsing error: {e!s}") from None


@app.get("/parse-file")
async def api_parse_file(file_path: str):
    """Parse a single file"""
    if not Path(file_path).exists():
        raise HTTPException(status_code=404, detail=f"File not found: {file_path}")

    try:
        chunks = parse_file(file_path)
        return {"file_path": file_path, "chunks": [chunk.to_dict() for chunk in chunks]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Parsing error: {e!s}") from None


@app.post("/preprocess-chunks")
async def api_preprocess_chunks(request: dict):
    """Preprocess code chunks (deduplication, enhancement)"""
    input_file = request.get("input_file")
    output_file = request.get("output_file")

    if not input_file or not Path(input_file).exists():
        raise HTTPException(status_code=400, detail="Valid input_file required")

    try:
        # Load chunks
        with Path(input_file).open(encoding="utf-8") as f:
            data = json.load(f)

        # Handle both direct chunk arrays and wrapped formats
        if "chunks" in data:
            chunks = data["chunks"]
        else:
            chunks = data  # Assume it's directly the chunks array

        # Preprocess
        preprocessor = ChunkPreprocessor()
        processed_chunks = preprocessor.process(chunks)

        # Save with same structure as input
        output_data = {
            "project_path": data.get("project_path", ""),
            "total_chunks": len(processed_chunks),
            "statistics": data.get("statistics", {}),
            "chunks": processed_chunks,
        }

        if output_file:
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            with Path(output_file).open("w", encoding="utf-8") as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)

        return {
            "message": f"Preprocessed {len(processed_chunks)} chunks",
            "output_file": output_file,
            "data": output_data if not output_file else None,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Preprocessing error: {e!s}") from None


@app.post("/embed-chunks")
async def api_embed_chunks(request: dict):
    """Generate embeddings for code chunks"""
    input_file = request.get("input_file")
    output_file = request.get("output_file")
    model_url = request.get("model_url", "http://localhost:12434/engines/llama.cpp/v1")
    model_name = request.get("model_name", "ai/embeddinggemma")

    if not input_file or not Path(input_file).exists():
        raise HTTPException(status_code=400, detail="Valid input_file required")

    try:
        # Load chunks
        with Path(input_file).open(encoding="utf-8") as f:
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

        # Save with same structure as input
        output_data = {
            "project_path": data.get("project_path", ""),
            "total_chunks": len(embedded_chunks),
            "statistics": data.get("statistics", {}),
            "chunks": embedded_chunks,
        }

        if output_file:
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            with Path(output_file).open("w", encoding="utf-8") as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)

        return {
            "message": f"Embedded {len(embedded_chunks)} chunks",
            "output_file": output_file,
            "data": output_data if not output_file else None,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding error: {e!s}")


@app.post("/index-chunks")
async def api_index_chunks(request: dict):
    """Index embedded chunks in Qdrant"""
    input_file = request.get("input_file")
    host = request.get("host", "localhost")
    port = request.get("port", 6333)
    collection_prefix = request.get("collection_prefix", "tipsy")

    if not input_file or not Path(input_file).exists():
        raise HTTPException(status_code=400, detail="Valid input_file required")

    try:
        # Load chunks
        with Path(input_file).open(encoding="utf-8") as f:
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

        return {
            "message": f"Indexed {len(chunks)} chunks in Qdrant",
            "host": host,
            "port": port,
            "collections": list(indexer.collections.keys()),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Indexing error: {e!s}")


@app.post("/enrich-chunks")
async def api_enrich_chunks(request: dict):
    """Enrich code chunks with AI-generated context"""
    input_file = request.get("input_file")
    output_file = request.get("output_file")
    symbol_index_file = request.get("symbol_index_file")

    if not input_file or not Path(input_file).exists():
        raise HTTPException(status_code=400, detail="Valid input_file required")

    try:
        # Load chunks
        with Path(input_file).open(encoding="utf-8") as f:
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
        symbol_index = None
        if symbol_index_file and Path(symbol_index_file).exists():
            with Path(symbol_index_file).open(encoding="utf-8") as f:
                symbol_index = json.load(f)

        # Enrich
        enricher = ContextEnricher(chunks=filtered_chunks, symbol_index=symbol_index)
        enriched_chunks = await enricher.enrich()

        # Save with same structure as input
        output_data = {
            "project_path": data.get("project_path", ""),
            "total_chunks": len(enriched_chunks),
            "statistics": data.get("statistics", {}),
            "chunks": enriched_chunks,
        }

        if output_file:
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            with Path(output_file).open("w", encoding="utf-8") as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)

        return {
            "message": f"Enriched {len(enriched_chunks)} chunks",
            "output_file": output_file,
            "data": output_data if not output_file else None,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Enrichment error: {e!s}")


@app.post("/batch-prompts")
async def api_batch_prompts(request: dict):
    """Batch process prompts through LLM"""
    input_path = request.get("input_path")
    output_file = request.get("output_file")
    output_format = request.get("output_format", "jsonl")
    delay = request.get("delay", 0.2)
    model = request.get("model")

    if not input_path or not Path(input_path).exists():
        raise HTTPException(status_code=400, detail="Valid input_path required")

    try:
        if model:
            os.environ["LLM_MODEL"] = model
        processor = BatchProcessor(delay=delay)
        prompts = processor.load_prompts(input_path)
        results = processor.process_prompts(prompts, output_file, output_format)
        return {"message": f"Processed {len(results)} prompts", "num_prompts": len(prompts), "output_file": output_file}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch processing error: {e!s}")


@app.post("/api/index-embedded")
async def api_index_embedded(request: dict):
    """Index pre-embedded JSON chunks"""
    path = request.get("path")
    if not path:
        raise HTTPException(status_code=400, detail='"path" is required')
    embedding_dim = request.get("embedding_dim", 768)
    collection_prefix = request.get("collection_prefix", "default")
    try:
        index_from_embedded_json(path, embedding_dim, collection_prefix)
        return {"status": "success", "message": "Pre-embedded JSON indexed successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Indexing error: {e!s}")


@app.post("/api/hybrid-setup")
async def api_hybrid_setup(request: dict):
    """Setup hybrid search collection"""
    collection_name = request.get("collection_name")
    chunks_path = request.get("chunks_path")
    if not collection_name or not chunks_path:
        raise HTTPException(status_code=400, detail='"collection_name" and "chunks_path" are required')
    try:
        setup_hybrid_collection(collection_name, chunks_path)
        return {"status": "success", "message": "Hybrid collection setup complete"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Hybrid setup error: {e!s}")


@click.group()
def cli():
    """RAG Code Parser CLI"""


@cli.command()
@click.option("--path", "-p", default=".", help="Project path to ingest")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
def ingest(path, verbose):
    """
    Run the data ingestion pipeline - parse code into chunks.
    """
    settings.project_path = Path(path).resolve()
    click.echo(f"Running code parsing pipeline for: {path}")

    try:
        # Parse the project
        parser = parse_project(path)

        if verbose:
            click.echo(f"Found {len(parser.chunks)} code chunks")
            click.echo(f"Statistics: {dict(parser.stats)}")

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
@click.option("--host", default="localhost", help="Qdrant host")
@click.option("--port", default=6333, type=int, help="Qdrant port")
@click.option("--collection-prefix", default="tipsy", help="Collection prefix")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
def index(input, host, port, collection_prefix, verbose):
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
@click.option("--path", "-p", required=True, help="Path to pre-embedded JSON file")
@click.option("--embedding-dim", "-d", default=768, type=int, help="Embedding dimension")
@click.option("--collection-prefix", "-c", default="default", help="Collection prefix")
def index_embedded(path, embedding_dim, collection_prefix):
    console.print(f"[bold blue]Indexing pre-embedded JSON from:[/bold blue] {path}")
    console.print(f"Embedding dim: {embedding_dim}, Prefix: {collection_prefix}")
    try:
        index_from_embedded_json(path, embedding_dim, collection_prefix)
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
    processor = BatchProcessor(delay=delay)
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
    click.echo(f"Starting RAG Code Parser API on {host}:{port}")

    # Mount static files if they exist
    if Path("static").exists():
        app.mount("/static", StaticFiles(directory="static"), name="static")
    uvicorn.run("main:app", host=host, port=port, reload=reload, log_level="info")


@cli.command()
@click.option("--qdrant-host", default="localhost", help="Qdrant host")
@click.option("--qdrant-port", default=6333, type=int, help="Qdrant port")
@click.option("--collection-prefix", default="tipsy", help="Collection prefix")
@click.option("--llm-url", default="http://localhost:12434/", help="LLM API URL")
@click.option("--llm-model", default="ai/llama3.2:latest", help="LLM model")
@click.option("--embedding-model", default="ai/embeddinggemma", help="Embedding model")
def rag(qdrant_host, qdrant_port, collection_prefix, llm_url, llm_model, embedding_model):
    """
    Interactive RAG query CLI using CodeRAG_2
    """
    from src.config import EmbeddingConfig
    from src.config import QdrantConfig

    embedding_config = EmbeddingConfig(
        model_url=f"{llm_url}engines/llama.cpp/v1", model_name=embedding_model, embedding_dim=768, batch_size=32
    )
    qdrant_config = QdrantConfig(host=qdrant_host, port=qdrant_port, collection_prefix=collection_prefix)

    rag_system = CodeRAG_2(
        embedding_config=embedding_config, qdrant_config=qdrant_config, llm_api_url=llm_url, llm_model_name=llm_model
    )

    console.print("[bold green]CodeRAG_2 Interactive CLI Ready![/bold green]")
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


if __name__ == "__main__":
    cli()
