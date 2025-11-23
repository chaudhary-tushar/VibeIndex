# Code Splitting Implementation for main.py

## Summary
This document provides the **complete, copy-paste ready code** for refactoring `main.py` by splitting it into modular files:

- `api.py`: FastAPI app (`app`) and all API endpoints (extracted and minimally adjusted from original lines 4-377).
- `cli.py`: Click CLI group (`cli`) and all commands (extracted from original lines 378-822, with adjustment to the `api` command to dynamically import and mount/run `api.app`).
- `main.py`: Minimal entrypoint that dispatches to the CLI (replacement for the entire original file).

**Key Changes:**
- Preserves all functionality, imports, options, and behaviors.
- No new dependencies.
- CLI interface unchanged (e.g., `python main.py ingest --path .`, `python main.py api --port 8000`).
- Dependencies handled: `api` imported inside `cli.py`'s `api` command to avoid circular imports.
- Duplicate imports removed where possible.
- Line numbers and whitespace preserved for accuracy.

## Migration Steps
1. **Backup** `main.py`.
2. Create `api.py` by copying the code below.
3. Create `cli.py` by copying the code below.
4. **Replace entire `main.py`** with the code below.
5. Test:
   ```
   python main.py --help
   python main.py ingest --path .
   python main.py api --reload
   curl http://localhost:8000/
   ```
6. Verify no linter errors (`ruff check .`).

## api.py (New File - ~350 lines)
```python
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses