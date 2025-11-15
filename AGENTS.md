# AGENTS.md

This file provides guidance to agents when working with code in this repository.

## Build/Lint/Test Commands

*   **Build:** N/A (Handled by `uvicorn` on startup)
*   **Lint:** `ruff check .`
*   **Test:** `pytest`
*   **Single Test:** `pytest tests/test_preprocessing.py::test_<function_name>`

## Code Style Guidelines

*   **Imports:** Follow standard Python import conventions.
*   **Formatting:** Use `ruff format .`
*   **Types:** Use type hints.
*   **Naming Conventions:** Follow standard Python naming conventions.
*   **Error Handling:** Use `try...except` blocks for error handling.

## Critical Patterns

- Main entry point is `main.py` with Click CLI (ingest/api commands)
- Preprocessing pipeline uses AST parsing for code chunking (see `src/preprocessing/README.md`)
- Docker Compose includes Qdrant and embedding model services (currently commented out)
- Configuration uses Pydantic `BaseSettings` with `.env` file loading
- Token counting utility `countokens.py` uses Hugging Face transformers
