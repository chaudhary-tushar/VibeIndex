# QWEN.md - Project Context for geminIndex

## Project Overview

geminIndex is a Retrieval-Augmented Generation (RAG) system designed to index and search through code repositories and documentation. The project uses a modular architecture with distinct components for preprocessing, embedding, and retrieval. It leverages Qdrant as a vector database for efficient similarity search and FastAPI for the retrieval API.

The primary purpose of this system is to enable semantic search over codebases, documentation, and other structured data by converting them into searchable embeddings.

## Architecture

The project follows a modular design with the following main components:

- **Preprocessing Layer**: Handles parsing of various file types (code, markdown, PDF, CSV) into meaningful chunks
- **Embedding Layer**: Converts parsed content into vector embeddings using transformer models
- **Retrieval Layer**: Provides a FastAPI-based API for semantic search against the vector database
- **Storage**: Uses Qdrant as a vector database for storing embeddings and enabling similarity search

## Key Technologies

- Python 3.13+
- FastAPI
- Qdrant (vector database)
- Transformers (for embeddings)
- Pydantic (for settings and data validation)
- Click (for CLI commands)

## Project Structure

```
geminIndex/
├── config/                 # Configuration and settings
│   ├── settings.py         # Pydantic settings model
├── scripts/                # Script files
│   └── ingest.py           # Data ingestion pipeline
├── src/                    # Main source code
│   ├── preprocessing/      # File parsing and chunking
│   │   ├── parser.py       # Code and document parser
│   │   └── ...
│   ├── embedding/          # Embedding generation
│   │   └── embedder.py     # Embedding logic
│   └── retrieval/          # Search and retrieval API
│       ├── main.py         # FastAPI application
│       └── search.py       # Search implementation
├── data/                   # Data directory (empty in initial setup)
├── cli-tools/              # CLI tools
├── docker/                 # Docker configurations
├── tests/                  # Test files
├── main.py                 # Main CLI entry point
├── pyproject.toml          # Project dependencies and configuration
└── docker-compose.yml      # Docker services
```

## Building and Running

### Prerequisites
- Python 3.13+
- Docker and Docker Compose (for Qdrant)

### Setup

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   # Or if using uv (based on uv.lock file):
   uv sync
   ```

2. **Set up environment variables**:
   Create a `.env` file based on the expected variables in `config/settings.py`:
   ```
   QDRANT_HOST=localhost
   QDRANT_PORT=6333
   QDRANT_API_KEY=your_api_key
   EMBEDDING_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
   EMBEDDING_DIM=384
   GENERATION_MODEL_NAME=gpt-3.5-turbo
   ```

3. **Start Qdrant (using Docker)**:
   ```bash
   docker-compose up -d
   ```

### Running the Application

The application provides a CLI with two main commands:

1. **Ingest Data**:
   ```bash
   python main.py ingest
   ```
   This command runs the data ingestion pipeline which:
   - Reads files from a code repository
   - Parses them using the preprocessing module
   - Generates embeddings
   - Stores embeddings in Qdrant

2. **Run the Retrieval API**:
   ```bash
   python main.py api
   ```
   This starts the FastAPI server which provides a `/retrieve` endpoint for semantic search.

### Development Commands

- **Running tests**: `pytest`
- **Formatting code**: (Not specified in current project, would need to add in pyproject.toml)
- **Linting**: (Not specified in current project, would need to add in pyproject.toml)

## Development Conventions

1. **Code Structure**: Follow the modular architecture with clear separation between preprocessing, embedding, and retrieval layers
2. **Configuration**: Use Pydantic BaseSettings for configuration management
3. **Error Handling**: Implement proper error handling throughout the pipeline
4. **Logging**: Add appropriate logging for debugging and monitoring
5. **Documentation**: Document public APIs and complex functions

## Key Components Details

### Preprocessing Layer
- Supports multiple file types: Python code, Markdown, PDF, CSV
- Implements AST-based parsing for code files
- Chunks content into meaningful segments for better retrieval

### Embedding Layer
- Uses transformer models for generating vector embeddings
- Designed to work with HuggingFace models
- Creates dense vector representations of parsed content

### Retrieval Layer
- FastAPI-based REST API
- Implements hybrid search capabilities
- Exposes `/retrieve` endpoint for semantic search queries

## Environment Variables

The application expects the following environment variables (defined in `config/settings.py`):
- `QDRANT_HOST`: Host for Qdrant database
- `QDRANT_PORT`: Port for Qdrant database
- `QDRANT_API_KEY`: API key for Qdrant (if authentication enabled)
- `EMBEDDING_MODEL_NAME`: Name of the embedding model to use
- `EMBEDDING_DIM`: Dimension of generated embeddings
- `GENERATION_MODEL_NAME`: Name of the generation model (future use)

## Data Flow

1. **Ingestion**: Files are read, parsed into chunks, converted to embeddings, and stored in Qdrant
2. **Query**: User queries are converted to embeddings
3. **Retrieval**: Vector similarity search is performed against stored embeddings
4. **Response**: Relevant results are returned to the user

This project is currently in development with several TODOs marked in the code that need implementation.