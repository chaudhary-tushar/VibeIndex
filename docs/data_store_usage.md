# Data Store Usage Guide

The enhanced data storage system in geminIndex provides flexible saving capabilities for various data types with automatic file extension detection and organized directory structure.

## Features

- **Automatic file type detection**: The system automatically detects the appropriate file format based on the data type
- **Organized directory structure**: Files are organized by processing method and optionally by language
- **Multiple data type support**: Handles dictionaries, lists, LlamaIndex Documents, strings, and more
- **Flexible parameters**: Easy to use with method, language, and extension parameters

## Function Signature

```python
def save_data(data: Any, method: str, language: str = None, ext: str = None) -> Path:
```

## Parameters

- `data`: The data to save. Can be various types (dict, list, str, llama_index Document, etc.)
- `method`: The processing method name (e.g., 'collecting-reading', 'parsing', 'chunking')
- `language`: The programming language if applicable (optional)
- `ext`: Forced file extension. If not provided, auto-determined from data type

## File Type Detection Rules

- `dict` or `list` → JSON format
- `LlamaIndex Document` → JSON format
- Tree-sitter related objects → TXT format
- JSON-formatted string → JSON format
- Other data types → TXT format (default)

## Directory Structure

The data is stored in the `./data` directory with the following structure:
```
data/
└── <method>/
    └── [<language>/]
        └── <filename>.<ext>
```

## Usage Examples

### Basic Usage
```python
from src.config.data_store import save_data

# Save a dictionary (will be saved as JSON)
data = {"key": "value", "items": [1, 2, 3]}
file_path = save_data(data, method="testing", language="python")
# Creates: ./data/testing/python/python.json

# Save a string (will be saved as TXT)
text = "Hello, World!"
file_path = save_data(text, method="testing")
# Creates: ./data/testing/testing.txt
```

### Usage with LlamaIndex Documents
```python
# When working with LlamaIndex Documents, they will be saved in JSON format
from llama_index.core import Document

doc = Document(text="Sample document content")
file_path = save_data(doc, method="parsing", language="python")
# Creates: ./data/parsing/python/python.json
```

### Explicit Extension Override
```python
# Force saving data as a specific file type
data = "Some text data"
file_path = save_data(data, method="testing", ext="csv")
# Creates: ./data/testing/testing.csv
```

## Integration with Parser

The parser now uses the enhanced data store to save results during different processing stages:

- **Collection Phase**: Files discovered during the `discover_files()` method are saved to `./data/collecting-reading/`
- **Parsing Phase**: Parsing results can be saved to `./data/parsing/`
- **Chunking Phase**: Chunked code can be saved to `./data/chunking/`

## Benefits

1. **Process-Based Organization**: Different processing stages are kept in separate directories
2. **Language-Based Subdirectories**: Code in different languages gets organized into appropriate subdirectories
3. **Automatic Format Detection**: No need to specify the format - the system determines the best format automatically
4. **Flexible Usage**: Can be used from anywhere in the codebase with appropriate parameters