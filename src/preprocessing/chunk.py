"""
Code chunk data structures, models, and preprocessing pipeline
Enhanced with advanced features from mature implementations
"""

import uuid
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rich.console import Console

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
    metadata: dict[str, Any] | None = None  # Code-specific metadata (decorators, access_modifier, etc.)
    documentation: dict[str, Any] | None = None  # Documentation and docstring info
    analysis: dict[str, Any] | None = None  # Code analysis (complexity, tokens, hash, etc.)
    relationships: dict[str, Any] | None = None  # Relationship info (imports, children, etc.)
    context: dict[str, Any] | None = None  # Context metadata (module, project, domain, hierarchy)

    def _init_optional_fields(self):
        """Initialize optional fields with defaults"""
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
            self.id = str(uuid.uuid4())

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
