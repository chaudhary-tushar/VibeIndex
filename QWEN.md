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
- **Configuration Layer**: Centralized configuration management for all external services with connectivity checks

## Key Technologies

- Python 3.13+
- FastAPI
- Qdrant (vector database)
- Transformers (for embeddings)
- Pydantic (for settings and data validation)
- Click (for CLI commands)
- Tree-sitter (for code parsing)
- LangChain (for LLM integration)

## Project Structure

```
geminIndex/
├── config/                 # Configuration and settings
│   ├── settings.py         # Pydantic settings model
│   ├── embedding_config.py # Embedding service configuration
│   ├── qdrant_config.py    # Qdrant service configuration
│   ├── llm_config.py       # LLM service configuration
│   ├── neo4j_config.py     # Neo4j service configuration
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
   EMBEDDING_MODEL_URL=http://localhost:12434/v1
   EMBEDDING_MODEL_NAME=ai/embeddinggemma
   EMBEDDING_DIM=768
   GENERATION_MODEL_URL=http://localhost:12434/v1
   GENERATION_MODEL_NAME=gpt-3.5-turbo
   QDRANT_HOST=localhost
   QDRANT_PORT=6333
   QDRANT_API_KEY=your_api_key
   NEO4J_URI=bolt://localhost:7687
   NEO4J_USERNAME=neo4j
   NEO4J_PASSWORD=your_password
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

- **Running tests**: `pytest tests/`
- **Formatting code**: `ruff format .`
- **Linting**: `ruff check .`
- **Additional CLI commands**:
  - `ingest`: Parse code into chunks
  - `preprocess`: Preprocess code chunks (deduplication, enhancement)
  - `embed`: Generate embeddings for code chunks
  - `index`: Index embedded chunks in Qdrant vector database
  - `api`: Run the FastAPI server for code parsing
  - `rag`: Interactive RAG query CLI
  - `advanced_rag`: Advanced RAG with reranking
  - `batch`: Batch process prompts through LLM
  - `enrich`: Enrich code chunks with AI-generated context summaries

## Development Conventions

1. **Code Structure**: Follow the modular architecture with clear separation between preprocessing, embedding, and retrieval layers
2. **Configuration**: Use Pydantic BaseSettings for configuration management, with external services configured via .env files
3. **Error Handling**: Implement proper error handling throughout the pipeline
4. **Logging**: Add appropriate logging for debugging and monitoring
5. **Documentation**: Document public APIs and complex functions

## Key Components Details

### Preprocessing Layer
- Supports multiple file types: Python code, Markdown, PDF, CSV
- Implements AST-based parsing for code files
- Chunks content into meaningful segments for better retrieval
- Uses Tree-sitter for parsing and libCST for Python-specific analysis

### Embedding Layer
- Uses transformer models for generating vector embeddings
- Designed to work with HuggingFace models
- Creates dense vector representations of parsed content
- Handles batch processing for efficiency

### Retrieval Layer
- FastAPI-based REST API
- Implements hybrid search capabilities
- Exposes `/retrieve` endpoint for semantic search queries
- Supports both keyword and vector similarity search

### Configuration Layer
- **Settings**: Centralized configuration using Pydantic BaseSettings
- **EmbeddingConfig**: Configuration for embedding services with connectivity check
- **QdrantConfig**: Configuration for Qdrant database with connectivity check
- **LLMConfig**: Configuration for LLM generation services with connectivity check
- **Neo4jConfig**: Configuration for Neo4j database with connectivity check

Each configuration class includes a `ping()` method to verify connectivity to the respective service before attempting to use it.

## Environment Variables

The application expects the following environment variables (defined in `config/settings.py`):
- `EMBEDDING_MODEL_URL`: API endpoint for the embedding model
- `EMBEDDING_MODEL_NAME`: Name of the embedding model to use
- `EMBEDDING_DIM`: Dimension of generated embeddings
- `GENERATION_MODEL_URL`: API endpoint for the generation model
- `GENERATION_MODEL_NAME`: Name of the generation model
- `QDRANT_HOST`: Host for Qdrant database
- `QDRANT_PORT`: Port for Qdrant database
- `QDRANT_API_KEY`: API key for Qdrant (if authentication enabled)
- `NEO4J_URI`: URI for Neo4j database (e.g., bolt://localhost:7687)
- `NEO4J_USERNAME`: Username for Neo4j database
- `NEO4J_PASSWORD`: Password for Neo4j database

## Data Flow

1. **Ingestion**: Files are read, parsed into chunks, converted to embeddings, and stored in Qdrant
2. **Query**: User queries are converted to embeddings
3. **Retrieval**: Vector similarity search is performed against stored embeddings
4. **Response**: Relevant results are returned to the user

## Testing

The project uses pytest for testing with a comprehensive test suite in the `tests/` directory:
- Unit tests for individual components
- Integration tests for pipeline components
- Mock external services (Qdrant, LLMs)
- Performance and accuracy tests

Run all tests with:
```bash
pytest tests/
```

## Debug Files

The project includes several `debug_*.py` files that are standalone scripts for manual testing and debugging specific functionality, rather than formal pytest tests. These include:
- `debug_meta.py`: Tests the MetadataExtractor functionality
- `debug_deps*.py`: Tests the DependencyMapper functionality with various Python code inputs

These files complement the formal test suite by providing an easy way to test specific components in isolation during development.
