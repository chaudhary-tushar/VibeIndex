"""
Code chunk data structures and models
"""

from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
import hashlib


@dataclass
class CodeChunk:
    """Represents a parsed code chunk"""

    type: str  # 'function', 'class', 'method', 'file'
    name: str
    code: str
    file_path: str
    language: str
    start_line: int
    end_line: int
    id: str = ""
    docstring: Optional[str] = None
    signature: Optional[str] = None
    complexity: int = 0
    dependencies: List[str] = None
    parent: Optional[str] = None
    context: Optional[Any] = None
    references: List[str] = None  # Symbols referenced in this chunk
    defines: List[str] = None     # Symbols defined in this chunk

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

    def to_dict(self):
        """Convert to dictionary for JSON serialization"""
        return asdict(self)
