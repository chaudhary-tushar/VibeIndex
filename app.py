import json
import os
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi import HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from src.config import EmbeddingConfig
from src.config import QdrantConfig
from src.embedding.embedder import EmbeddingGenerator
from src.generation import BatchProcessor_2
from src.generation.context_builder import ContextEnricher
from src.generation.context_builder import get_summarized_chunks_ids
from src.generation.context_builder import stats_check

# Import our preprocessing modules
from src.preprocessing import parse_file
from src.preprocessing import parse_project
from src.preprocessing.chunk import ChunkPreprocessor
from src.retrieval.hybrid_search import setup_hybrid_collection
from src.retrieval.search import QdrantIndexer
from src.retrieval.search import index_from_embedded_json

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
        processor = BatchProcessor_2(delay=delay)
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


def main(host="0.0.0.0", port=8000, reload=False):
    """
    Run the FastAPI server for code parsing.
    """
    print(f"Starting RAG Code Parser API on {host}:{port}")

    # Mount static files if they exist
    if Path("static").exists():
        app.mount("/static", StaticFiles(directory="static"), name="static")
    uvicorn.run(app, host=host, port=port, reload=reload, log_level="info")


if __name__ == "__main__":
    main()
