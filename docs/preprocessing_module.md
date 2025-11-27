# Preprocessing Module Knowledge Base

This document serves as a comprehensive knowledge base for the `src.preprocessing` module in the geminIndex project. It details all classes, functions, and their usage patterns to help you understand and maintain the codebase effectively.

## Module Overview

The preprocessing module handles parsing, analyzing, and preparing code for embedding in the RAG (Retrieval-Augmented Generation) system. It includes language-specific parsers, metadata extraction, dependency analysis, and chunk preprocessing.

## Directory Structure

```
src/preprocessing/
├── __init__.py          # Module exports and convenience functions
├── analyzer.py          # Code analysis and parsing logic for different languages
├── chunk.py             # Data structures for code chunks and preprocessing
├── dependency_mapper.py # Dependency analysis and mapping
├── language_config.py   # Language configuration and mappings
├── metadata_extractor.py # Metadata extraction from code chunks
├── parser.py            # Main parser orchestration using Tree-sitter
├── preprocessor.py      # Enhanced preprocessing for code chunks before embedding
└── README.md            # Module README
```

## Core Classes and Functions

### 1. CodeChunk (chunk.py)
**Safety Level:** ⚠️ Moderate to change - affects data structure across the module

**Purpose:** Represents a parsed code chunk with metadata.

**Fields:**
- `type`: Type of chunk (function, class, method, file, html_element, etc.)
- `name`: Name of the chunk
- `code`: The actual code content
- `file_path`: Path to the source file
- `language`: Programming language of the chunk
- `start_line`, `end_line`: Line numbers in the source file
- `id`: Unique identifier for the chunk
- `docstring`, `signature`, `complexity`: Additional metadata
- `dependencies`, `references`, `defines`: Relationship information
- `location`, `metadata`, `documentation`, `analysis`, `relationships`, `context`: Enhanced metadata structures

**Key Methods:**
- `to_dict()`: Convert to dictionary for JSON serialization

### 2. ChunkPreprocessor (chunk.py)
**Safety Level:** ✅ Safe to modify - preprocessing pipeline enhancements

**Purpose:** Enhanced preprocessing for code chunks before embedding.

**Key Methods:**
- `deduplicate(chunks: list[dict])`: Remove duplicate chunks based on code content
- `enhance_chunk(chunk: dict)`: Enhance chunk with additional context for better embeddings
- `validate_chunk(chunk: dict, max_tokens: int)`: Validate chunk suitability for embedding
- `process(chunks: list[dict])`: Full preprocessing pipeline

### 3. ChunkPreprocessor2 (chunk.py)
**Safety Level:** 🚫 Do not modify - preserved for backward compatibility

**Purpose:** Original ChunkPreprocessor implementation (preserved for compatibility).

### 4. CodeParser (parser.py)
**Safety Level:** ⚠️ Moderate to modify - core parsing logic

**Purpose:** Main parser using Tree-sitter for multiple languages.

**Key Methods:**
- `__init__(project_path: str)`: Initialize parser with project path
- `_load_gitignore()`: Load .gitignore patterns
- `_should_ignore(path: Path)`: Check if path should be ignored
- `_get_parser(language: str)`: Get or create parser for a language
- `discover_files()`: Discover all code files in project
- `_determine_language(file_path: Path)`: Determine language and read file content
- `parse_file(file_path: Path)`: Parse a single file
- `parse_project()`: Parse entire project
- `save_results(output_path: str)`: Save parsed chunks to JSON
- `visualize_results()`: Display visualization of results
- `save_symbol_index(output_path: str)`: Save symbol index to JSON

### 5. Analyzer (analyzer.py)
**Safety Level:** ✅ Safe to modify - analysis and parsing improvements

**Purpose:** Handles code analysis and parsing for different languages.

**Key Methods:**
- `extract_js_chunks(...)`: Extract functions and classes from JavaScript/TypeScript
- `extract_html_chunks(...)`: Extract meaningful chunks from HTML
- `extract_css_chunks(...)`: Extract chunks from CSS
- `extract_generic_chunks(...)`: Fallback chunking for unsupported languages
- `parse_python_file_libcst(file_path: Path)`: Parse Python file using libCST
- `find_called_symbols(...)`: Find symbols in code that are defined in the current project
- `_calculate_complexity(code: str)`: Calculate cyclomatic complexity
- `_extract_dependencies(...)`: Extract dependencies from code
- `add_location_metadata(chunk: CodeChunk, node)`: Add detailed location info
- `add_code_metadata(chunk: CodeChunk, node, code_bytes)`: Extract code-specific metadata
- `add_analysis_metadata(chunk: CodeChunk)`: Add analysis metadata
- `add_relationship_metadata(...)`: Add relationship metadata
- `add_context_metadata(...)`: Add contextual information
- `enhance_chunk_completely(...)`: Full enhancement pipeline for a chunk

### 6. DependencyMapper (dependency_mapper.py)
**Safety Level:** ✅ Safe to modify - dependency analysis improvements

**Purpose:** Handles dependency analysis and mapping between code elements.

**Key Methods:**
- `extract_dependencies(...)`: Extract import/dependency statements and referenced symbols
- `_extract_python_imports(code: str)`: Extract Python import statements
- `_extract_python_symbol_usage(...)`: Extract symbol usage in Python
- `_extract_js_imports(code: str)`: Extract JavaScript/TypeScript import statements
- `_extract_js_symbol_usage(...)`: Extract symbol usage in JavaScript
- `_extract_html_dependencies(code: str)`: Extract dependencies from HTML
- `_extract_css_dependencies(code: str)`: Extract dependencies from CSS
- `build_dependency_graph(...)`: Build a dependency graph from chunks
- `find_reverse_dependencies(chunk_id: str)`: Find all chunks that depend on the given chunk
- `get_dependency_hierarchy(chunk_id: str)`: Get dependency hierarchy for a chunk

### 7. LanguageConfig (language_config.py)
**Safety Level:** 🚫 Do not modify - critical configuration

**Purpose:** Configuration for different programming languages.

**Constants:**
- `LANGUAGE_MAP`: Extension to language mapping
- `LANGUAGES`: Tree-sitter language parsers
- `QUERIES`: Tree-sitter query patterns for extracting definitions
- `DEFAULT_IGNORE_PATTERNS`: Default patterns to ignore during file discovery

### 8. MetadataExtractor (metadata_extractor.py)
**Safety Level:** ✅ Safe to modify - metadata extraction improvements

**Purpose:** Handles extraction of semantic metadata from code.

**Key Methods:**
- `extract_docstring(...)`: Extract docstring from AST node
- `extract_signature(code: str)`: Extract function/method signature
- `extract_complexity(code: str)`: Calculate cyclomatic complexity
- `extract_tags_and_categories(chunk: CodeChunk)`: Generate tags based on code analysis
- `enhance_chunk_metadata(chunk: CodeChunk)`: Add additional metadata to a chunk

### 9. ChunkPreprocessor (preprocessor.py)
**Safety Level:** ⚠️ Moderate to modify - preprocessing algorithm changes

**Purpose:** Enhanced preprocessing for code chunks before embedding and indexing (duplicate of ChunkPreprocessor from chunk.py).

## Module Exports (__init__.py)

The module exports several key classes:
- `ChunkPreprocessor`: Enhanced preprocessing class
- `ChunkPreprocessor2`: Original preprocessing class for backward compatibility
- `CodeChunk`: Code chunk data structure
- `Analyzer`: Code analysis class
- `CodeParser`: Main parser class
- `DependencyMapper`: Dependency analysis class
- `LanguageConfig`: Language configuration
- `MetadataExtractor`: Metadata extraction class

## Convenience Functions

### parse_project(project_path: str) -> CodeParser
**Purpose:** Parse an entire project and return CodeParser instance with results.

### parse_file(file_path: str, project_path: str | None = None) -> list[CodeChunk]
**Purpose:** Parse a single file and return its chunks.

## Safety Guidelines for Future Changes

1. **High-Risk Changes (Avoid unless necessary):**
   - `LanguageConfig` class and its constants - changes could break parsing for multiple languages
   - Data structure fields in `CodeChunk` - affects serialization and downstream systems
   - Core parsing logic in `CodeParser` - could break the entire pipeline

2. **Medium-Risk Changes (Proceed with caution):**
   - Core preprocessing algorithms in `ChunkPreprocessor`
   - Main parsing methods in `CodeParser`
   - Core analysis methods in `Analyzer`

3. **Low-Risk Changes (Safest to modify):**
   - Analysis and enhancement methods in `Analyzer`
   - Dependency extraction methods in `DependencyMapper`
   - Metadata extraction in `MetadataExtractor`
   - Preprocessing utilities and helper functions

## Common Usage Patterns

### Using the Parser
```python
from src.preprocessing import CodeParser

# Parse an entire project
parser = CodeParser("/path/to/project")
parser.parse_project()
chunks = parser.chunks

# Parse a single file
from src.preprocessing import parse_file
chunks = parse_file("/path/to/file.py")
```

### Preprocessing Chunks
```python
from src.preprocessing.chunk import ChunkPreprocessor

preprocessor = ChunkPreprocessor()
processed_chunks = preprocessor.process(chunks)
```

### Dependency Analysis
```python
from src.preprocessing import DependencyMapper

mapper = DependencyMapper()
deps = mapper.extract_dependencies(code, language, symbol_index)
```

## Integration Points

- The module integrates with the `data_store` module to save collection results
- Connects with the embedding module through the `CodeChunk` data structure
- Works with Tree-sitter for parsing multiple languages
- Uses libCST for enhanced Python parsing
- Integrates with LlamaIndex for document handling