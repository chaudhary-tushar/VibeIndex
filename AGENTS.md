# Agent Guidelines

## Build/Lint/Test Commands

*   **Build:** N/A (Likely handled by `uvicorn` on startup)
*   **Lint:**  `ruff check .` (Assumed based on common Python practices)
*   **Test:** `pytest`
*   **Single Test:** `pytest tests/test_preprocessing.py::test_<function_name>`

## Code Style Guidelines

*   **Imports:** Follow standard Python import conventions.
*   **Formatting:** Use `ruff format .` (Assumed based on common Python practices).
*   **Types:** Use type hints.
*   **Naming Conventions:** Follow standard Python naming conventions.
*   **Error Handling:** Use `try...except` blocks for error handling.
