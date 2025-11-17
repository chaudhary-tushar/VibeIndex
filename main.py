import click
import uvicorn
from pathlib import Path
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
import json
import os

# Import our preprocessing modules
from src.preprocessing import parse_project, parse_file, CodeParser

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
