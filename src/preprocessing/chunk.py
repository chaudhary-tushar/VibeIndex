"""
Code chunk data structures and models
"""

from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
import hashlib


@dataclass
class CodeChunk:
    """Represents a parsed code chunk - Enhanced version"""

    type: str  # 'function', 'class', 'method', 'file'
    name: str
    code: str
    file_path: str
    language: str
    start_line: int
    end_line: int
    id: str = ""
    qualified_name: Optional[str] = None
    docstring: Optional[str] = None
    signature: Optional[str] = None
    complexity: int = 0
    dependencies: List[str] = None
    parent: Optional[str] = None
    context: Optional[Dict[str, Any]] = None
    references: List[str] = None  # Symbols referenced in this chunk
    defines: List[str] = None     # Symbols defined in this chunk
    location: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    documentation: Optional[Dict[str, Any]] = None
    analysis: Optional[Dict[str, Any]] = None
    relationships: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []
        if self.references is None:
            self.references = []
        if self.defines is None:
            self.defines = []
        if not self.id:
            hash_input = f"{self.file_path}:{self.name}:{self.start_line}"
            self.id = hashlib.md5(hash_input.encode()).hexdigest()[:12]
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

    def to_dict(self):
        """Convert to dictionary for JSON serialization"""
        return asdict(self)
