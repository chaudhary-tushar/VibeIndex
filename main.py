import click
import uvicorn
from pathlib import Path
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
import json
import os
import asyncio

# Import our preprocessing modules
from src.preprocessing import parse_project, parse_file, CodeParser
from src.preprocessing.preprocessor import ChunkPreprocessor
from src.embedding.embedder import EmbeddingGenerator, EmbeddingConfig
from src.embedding.batch_processor import BatchProcessor
from src.retrieval.search import QdrantIndexer, QdrantConfig
from src.generation.context_builder import ContextEnricher, stats_check, get_summarized_chunks_ids

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

    if not os.path.exists(project_path):
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
            "limited": len(parser.chunks) > 100
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Parsing error: {str(e)}")

@app.get("/parse-file")
async def api_parse_file(file_path: str):
    """Parse a single file"""
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"File not found: {file_path}")

    try:
        chunks = parse_file(file_path)
        return {
            "file_path": file_path,
            "chunks": [chunk.to_dict() for chunk in chunks]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Parsing error: {str(e)}")

@app.post("/preprocess-chunks")
async def api_preprocess_chunks(request: dict):
    """Preprocess code chunks (deduplication, enhancement)"""
    input_file = request.get("input_file")
    output_file = request.get("output_file")

    if not input_file or not os.path.exists(input_file):
        raise HTTPException(status_code=400, detail="Valid input_file required")

    try:
        # Load chunks
        with open(input_file, "r", encoding="utf-8") as f:
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
            "chunks": processed_chunks
        }

        if output_file:
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)

        return {
            "message": f"Preprocessed {len(processed_chunks)} chunks",
            "output_file": output_file,
            "data": output_data if not output_file else None
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Preprocessing error: {str(e)}")

@app.post("/embed-chunks")
async def api_embed_chunks(request: dict):
    """Generate embeddings for code chunks"""
    input_file = request.get("input_file")
    output_file = request.get("output_file")
    model_url = request.get("model_url", "http://localhost:12434/engines/llama.cpp/v1")
    model_name = request.get("model_name", "ai/embeddinggemma")

    if not input_file or not os.path.exists(input_file):
        raise HTTPException(status_code=400, detail="Valid input_file required")

    try:
        # Load chunks
        with open(input_file, "r", encoding="utf-8") as f:
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
            "chunks": embedded_chunks
        }

        if output_file:
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)

        return {
            "message": f"Embedded {len(embedded_chunks)} chunks",
            "output_file": output_file,
            "data": output_data if not output_file else None
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding error: {str(e)}")

@app.post("/index-chunks")
async def api_index_chunks(request: dict):
    """Index embedded chunks in Qdrant"""
    input_file = request.get("input_file")
    host = request.get("host", "localhost")
    port = request.get("port", 6333)
    collection_prefix = request.get("collection_prefix", "tipsy")

    if not input_file or not os.path.exists(input_file):
        raise HTTPException(status_code=400, detail="Valid input_file required")

    try:
        # Load chunks
        with open(input_file, "r", encoding="utf-8") as f:
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
            "collections": list(indexer.collections.keys())
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Indexing error: {str(e)}")

@app.post("/enrich-chunks")
async def api_enrich_chunks(request: dict):
    """Enrich code chunks with AI-generated context"""
    input_file = request.get("input_file")
    output_file = request.get("output_file")
    symbol_index_file = request.get("symbol_index_file")

    if not input_file or not os.path.exists(input_file):
        raise HTTPException(status_code=400, detail="Valid input_file required")

    try:
        # Load chunks
        with open(input_file, "r", encoding="utf-8") as f:
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
        if symbol_index_file and os.path.exists(symbol_index_file):
            with open(symbol_index_file, "r", encoding="utf-8") as f:
                symbol_index = json.load(f)

        # Enrich
        enricher = ContextEnricher(chunks=filtered_chunks, symbol_index=symbol_index)
        enriched_chunks = await enricher.enrich()

        # Save with same structure as input
        output_data = {
            "project_path": data.get("project_path", ""),
            "total_chunks": len(enriched_chunks),
            "statistics": data.get("statistics", {}),
            "chunks": enriched_chunks
        }

        if output_file:
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)

        return {
            "message": f"Enriched {len(enriched_chunks)} chunks",
            "output_file": output_file,
            "data": output_data if not output_file else None
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Enrichment error: {str(e)}")

@click.group()
def cli():
    """RAG Code Parser CLI"""
    pass

@cli.command()
@click.option('--path', '-p', default='.', help='Project path to ingest')
@click.option('--output', '-o', help='Output file for parsed chunks (JSON)')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
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
                click.echo(f"  {i+1}. {chunk.name} ({chunk.type}) - {chunk.file_path}")
            if len(parser.chunks) > 10:
                click.echo(f"  ... and {len(parser.chunks) - 10} more")

        click.echo("\n✅ Code parsing complete!")

    except Exception as e:
        click.echo(f"❌ Error during parsing: {e}", err=True)
        return 1

@cli.command()
@click.option('--input', '-i', required=True, help='Input JSON file with chunks')
@click.option('--output', '-o', help='Output file for preprocessed chunks (JSON)')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
def preprocess(input, output, verbose):
    """
    Preprocess code chunks (deduplication, enhancement).
    """
    click.echo(f"Preprocessing chunks from: {input}")

    try:
        # Load chunks
        with open(input, "r", encoding="utf-8") as f:
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
                "chunks": processed_chunks
            }
            Path(output).parent.mkdir(parents=True, exist_ok=True)
            with open(output, "w", encoding="utf-8") as f:
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
@click.option('--input', '-i', required=True, help='Input JSON file with chunks')
@click.option('--output', '-o', help='Output file for embedded chunks (JSON)')
@click.option('--model-url', default="http://localhost:12434/engines/llama.cpp/v1", help='Embedding model URL')
@click.option('--model-name', default="ai/embeddinggemma", help='Embedding model name')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
def embed(input, output, model_url, model_name, verbose):
    """
    Generate embeddings for code chunks.
    """
    click.echo(f"Generating embeddings for chunks from: {input}")
    click.echo(f"Model: {model_name} @ {model_url}")

    try:
        # Load chunks
        with open(input, "r", encoding="utf-8") as f:
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
                "chunks": embedded_chunks
            }
            Path(output).parent.mkdir(parents=True, exist_ok=True)
            with open(output, "w", encoding="utf-8") as f:
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
@click.option('--input', '-i', required=True, help='Input JSON file with embedded chunks')
@click.option('--host', default="localhost", help='Qdrant host')
@click.option('--port', default=6333, type=int, help='Qdrant port')
@click.option('--collection-prefix', default="tipsy", help='Collection prefix')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
def index(input, host, port, collection_prefix, verbose):
    """
    Index embedded chunks in Qdrant vector database.
    """
    click.echo(f"Indexing chunks from: {input}")
    click.echo(f"Qdrant: {host}:{port}, Prefix: {collection_prefix}")

    try:
        # Load chunks
        with open(input, "r", encoding="utf-8") as f:
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
@click.option('--input', '-i', required=True, help='Input JSON file with chunks')
@click.option('--output', '-o', required=True, help='Output enriched JSON file')
@click.option('--symbol-index', '-s', help='Optional symbol index JSON file')
@click.option('--model', '-m', default="ai/llama3.2:latest", help='LLM model to use')
def enrich(input, output, symbol_index, model):
    """
    Enrich code chunks with AI-generated context summaries.
    """
    click.echo(f"Enriching chunks from {input} to {output}")

    try:
        # Load chunks
        with open(input, "r", encoding="utf-8") as f:
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
        if symbol_index and os.path.exists(symbol_index):
            with open(symbol_index, "r", encoding="utf-8") as f:
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
            "chunks": enriched_chunks
        }

        Path(output).parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        click.echo(f"✅ Enriched {len(enriched_chunks)} chunks. Saved to {output}")

    except Exception as e:
        click.echo(f"❌ Error during enrichment: {e}", err=True)
        return 1

@cli.command()
@click.option('--host', default='0.0.0.0', help='Host to bind to')
@click.option('--port', default=8000, type=int, help='Port to bind to')
@click.option('--reload', is_flag=True, help='Enable auto-reload for development')
def api(host, port, reload):
    """
    Run the FastAPI server for code parsing.
    """
    click.echo(f"Starting RAG Code Parser API on {host}:{port}")

    # Mount static files if they exist
    if os.path.exists('static'):
        app.mount("/static", StaticFiles(directory="static"), name="static")

    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )

if __name__ == '__main__':
    cli()
