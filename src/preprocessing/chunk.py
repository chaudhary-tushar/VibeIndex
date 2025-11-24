"""
Code chunk data structures, models, and preprocessing pipeline
Enhanced with advanced features from mature implementations
"""

import hashlib
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rich.console import Console
from tqdm import tqdm

console = Console()


@dataclass
class CodeChunk:
    """Represents a parsed code chunk - Enhanced version (migrated from enhanced.py)"""

    # Core identification (required)
    type: str  # 'function', 'class', 'method', 'file', 'html_element', etc.
    name: str
    code: str
    file_path: str
    language: str
    start_line: int
    end_line: int

    # Basic attributes (with defaults)
    id: str = ""
    qualified_name: str | None = None
    docstring: str | None = None
    signature: str | None = None
    complexity: int = 0
    parent: str | None = None

    # Dependencies and references
    dependencies: list[str] = None  # External dependencies
    references: list[str] = None  # Symbols referenced in this chunk (NEW from enhanced.py)
    defines: list[str] = None  # Symbols defined in this chunk (NEW from enhanced.py)

    # Comprehensive metadata structures (NEW from enhanced.py)
    location: dict[str, Any] | None = None  # Detailed location info (start/end line, column)
    metadata: dict[str, Any] | None = None  # Code-specific metadata (decorators, access_modifier, etc.)
    documentation: dict[str, Any] | None = None  # Documentation and docstring info
    analysis: dict[str, Any] | None = None  # Code analysis (complexity, tokens, hash, etc.)
    relationships: dict[str, Any] | None = None  # Relationship info (imports, children, etc.)
    context: dict[str, Any] | None = None  # Context metadata (module, project, domain, hierarchy)

    def _init_optional_fields(self):
        """Initialize optional fields with defaults"""
        if self.location is None:
            self.location = {}
        if self.context is None:
            self.context = {}
        if self.metadata is None:
            self.metadata = {}
        if self.documentation is None:
            self.documentation = {}
        if self.analysis is None:
            self.analysis = {}
        if self.relationships is None:
            self.relationships = {}

    def __post_init__(self):
        # Initialize lists
        if self.dependencies is None:
            self.dependencies = []
        if self.references is None:
            self.references = []
        if self.defines is None:
            self.defines = []

        # Generate ID if not provided
        if not self.id:
            hash_input = f"{self.file_path}:{self.qualified_name or self.name}:{self.start_line}"
            self.id = hashlib.md5(hash_input.encode()).hexdigest()[:12]

        # Initialize optional fields
        self._init_optional_fields()

        # Generate qualified name if not provided
        if not self.qualified_name:
            self.qualified_name = self._generate_qualified_name()

    def _generate_qualified_name(self) -> str:
        """Generate fully qualified name (migrated from enhanced.py)"""
        file_stem = Path(self.file_path).stem
        if self.type in {"class", "function", "method"}:
            return f"{file_stem}.{self.name}"
        return self.name

    def to_dict(self):
        """Convert to dictionary for JSON serialization"""
        return asdict(self)


class ChunkPreprocessor:
    """Enhanced preprocessing for code chunks before embedding - Merged from mature implementation"""

    def __init__(self):
        self.dedup_hashes = set()
        self.stats = {
            "total": 0,
            "duplicates": 0,
            "enhanced": 0,
            "too_large": 0,
        }

    def deduplicate(self, chunks: list[dict]) -> list[dict]:
        """Remove duplicate chunks based on code content"""
        unique_chunks = []

        for chunk in chunks:
            # Create hash from code content
            code_hash = hashlib.md5(chunk["code"].encode()).hexdigest()

            if code_hash not in self.dedup_hashes:
                self.dedup_hashes.add(code_hash)
                unique_chunks.append(chunk)
            else:
                self.stats["duplicates"] += 1

        console.print(f"[yellow]Removed {self.stats['duplicates']} duplicate chunks[/yellow]")
        return unique_chunks

    def enhance_chunk(self, chunk: dict) -> dict:
        """Enhance chunk with additional context for better embeddings"""
        enhanced = chunk.copy()

        # Build rich text representation for embedding
        parts = []

        # 1. Add contextual prefix
        context_prefix = f"# {chunk['language'].upper()} {chunk['type'].upper()}"
        if chunk.get("qualified_name"):
            context_prefix += f": {chunk['qualified_name']}"
        parts.append(context_prefix)

        if isinstance(chunk.get("context"), str):
            parts.append(f"html/css context: {chunk.get('context')}")
        else:
            # 2. Add file context
            if chunk.get("context", {}).get("file_hierarchy"):
                file_ctx = " > ".join(chunk["context"]["file_hierarchy"])
                parts.append(f"# Location: {file_ctx}")

            # 3. Add domain context
            if chunk.get("context", {}).get("domain_context"):
                parts.append(f"# Purpose: {chunk['context']['domain_context']}")

        # 4. Add docstring if available
        if chunk.get("docstring"):
            parts.append(f'"""{chunk["docstring"]}"""')

        # 5. Add signature for functions
        if chunk.get("signature"):
            parts.append(chunk["signature"])

        # 6. Add decorators/metadata
        if chunk.get("metadata", {}).get("decorators"):
            parts.extend(chunk["metadata"]["decorators"])

        # 7. Add the actual code
        parts.append(chunk["code"])

        # 8. Add dependencies as comments
        if chunk.get("dependencies"):
            deps = ", ".join(chunk["dependencies"][:5])  # Limit to 5
            parts.append(f"# Dependencies: {deps}")

        # 9. Add defined symbols
        if chunk.get("defines"):
            parts.append(f"# Defines: {', '.join(chunk['defines'])}")

        # Combine all parts
        enhanced["embedding_text"] = "\n".join(parts)
        enhanced["embedding_text_length"] = len(enhanced["embedding_text"])

        self.stats["enhanced"] += 1
        return enhanced

    def validate_chunk(self, chunk: dict, max_tokens: int = 8192) -> bool:
        """Validate chunk is suitable for embedding"""
        # Rough token estimate (1 token ≈ 4 chars)
        estimated_tokens = len(chunk.get("embedding_text", chunk["code"])) // 4

        if estimated_tokens > max_tokens:
            self.stats["too_large"] += 1
            return False

        return True

    def process(self, chunks: list[dict]) -> list[dict]:
        """Full preprocessing pipeline"""
        self.stats["total"] = len(chunks)
        console.print(f"[cyan]Starting preprocessing of {len(chunks)} chunks...[/cyan]")

        # 1. Deduplicate
        chunks = self.deduplicate(chunks)

        # 2. Enhance each chunk
        enhanced_chunks = []
        for chunk in tqdm(chunks, desc="Enhancing chunks", unit="chunk"):
            enhanced = self.enhance_chunk(chunk)
            if self.validate_chunk(enhanced):
                enhanced_chunks.append(enhanced)

        console.print(f"[green]✓ Preprocessing complete: {len(enhanced_chunks)} chunks ready[/green]")
        console.print(f"  - Duplicates removed: {self.stats['duplicates']}")
        console.print(f"  - Too large (skipped): {self.stats['too_large']}")

        return enhanced_chunks


# Preserve original ChunkPreprocessor with "2" suffix for backward compatibility
class ChunkPreprocessor2:
    """Original ChunkPreprocessor implementation (preserved for compatibility)"""

    def __init__(self):
        self.dedup_hashes = set()
        self.stats = {
            "total": 0,
            "duplicates": 0,
            "enhanced": 0,
            "too_large": 0,
        }

    def deduplicate(self, chunks: list[dict]) -> list[dict]:
        """Remove duplicate chunks based on code content"""
        unique_chunks = []

        for chunk in chunks:
            # Create hash from code content
            code_hash = hashlib.md5(chunk["code"].encode()).hexdigest()

            if code_hash not in self.dedup_hashes:
                self.dedup_hashes.add(code_hash)
                unique_chunks.append(chunk)
            else:
                self.stats["duplicates"] += 1

        console.print(f"[yellow]Removed {self.stats['duplicates']} duplicate chunks[/yellow]")
        return unique_chunks

    def enhance_chunk(self, chunk: dict) -> dict:
        """Enhance chunk with additional context for better embeddings"""
        enhanced = chunk.copy()

        # Build rich text representation for embedding
        parts = []

        # 1. Add contextual prefix
        context_prefix = f"# {chunk['language'].upper()} {chunk['type'].upper()}"
        if chunk.get("qualified_name"):
            context_prefix += f": {chunk['qualified_name']}"
        parts.append(context_prefix)
        if isinstance(chunk.get("context"), str):
            parts.append(f"html/css context: {chunk.get('context')}")
        else:
            # 2. Add file context
            if chunk.get("context", {}).get("file_hierarchy"):
                file_ctx = " > ".join(chunk["context"]["file_hierarchy"])
                parts.append(f"# Location: {file_ctx}")

            # 3. Add domain context
            if chunk.get("context", {}).get("domain_context"):
                parts.append(f"# Purpose: {chunk['context']['domain_context']}")

        # 4. Add docstring if available
        if chunk.get("docstring"):
            parts.append(f'"""{chunk["docstring"]}"""')

        # 5. Add signature for functions
        if chunk.get("signature"):
            parts.append(chunk["signature"])

        # 6. Add decorators/metadata
        if chunk.get("metadata", {}).get("decorators"):
            parts.extend(chunk["metadata"]["decorators"])

        # 7. Add the actual code
        parts.append(chunk["code"])

        # 8. Add dependencies as comments
        if chunk.get("dependencies"):
            deps = ", ".join(chunk["dependencies"][:5])  # Limit to 5
            parts.append(f"# Dependencies: {deps}")

        # 9. Add defined symbols
        if chunk.get("defines"):
            parts.append(f"# Defines: {', '.join(chunk['defines'])}")

        # Combine all parts
        enhanced["embedding_text"] = "\n".join(parts)
        enhanced["embedding_text_length"] = len(enhanced["embedding_text"])

        self.stats["enhanced"] += 1
        return enhanced

    def validate_chunk(self, chunk: dict, max_tokens: int = 8192) -> bool:
        """Validate chunk is suitable for embedding"""
        # Rough token estimate (1 token ≈ 4 chars)
        estimated_tokens = len(chunk.get("embedding_text", chunk["code"])) // 4

        if estimated_tokens > max_tokens:
            self.stats["too_large"] += 1
            return False

        return True

    def process(self, chunks: list[dict]) -> list[dict]:
        """Full preprocessing pipeline"""
        self.stats["total"] = len(chunks)
        console.print(f"[cyan]Starting preprocessing of {len(chunks)} chunks...[/cyan]")

        # 1. Deduplicate
        chunks = self.deduplicate(chunks)

        # 2. Enhance each chunk
        enhanced_chunks = []
        for chunk in tqdm(chunks, desc="Enhancing chunks", unit="chunk"):
            enhanced = self.enhance_chunk(chunk)
            if self.validate_chunk(enhanced):
                enhanced_chunks.append(enhanced)

        console.print(f"[green]✓ Preprocessing complete: {len(enhanced_chunks)} chunks ready[/green]")
        console.print(f"  - Duplicates removed: {self.stats['duplicates']}")
        console.print(f"  - Too large (skipped): {self.stats['too_large']}")

        return enhanced_chunks
