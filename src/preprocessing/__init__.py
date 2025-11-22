"""
Preprocessing module for RAG indexing pipeline
"""

from .parser import CodeParser
from .analyzer import Analyzer
from .metadata_extractor import MetadataExtractor
from .dependency_mapper import DependencyMapper
from .chunk import CodeChunk
from .language_config import LanguageConfig

__all__ = [
    'CodeParser',
    'Analyzer',
    'MetadataExtractor',
    'DependencyMapper',
    'CodeChunk',
    'LanguageConfig'
]

# Convenience function for quick parsing
def parse_project(project_path: str) -> CodeParser:
    """
    Parse an entire project and return CodeParser instance with results.

    Args:
        project_path: Path to the project directory

    Returns:
        CodeParser instance with parsed chunks and metadata
    """
    parser = CodeParser(project_path)
    parser.parse_project()
    return parser

def parse_file(file_path: str, project_path: str = None) -> list[CodeChunk]:
    """
    Parse a single file and return its chunks.

    Args:
        file_path: Path to the file to parse
        project_path: Project root path (optional, defaults to file's parent)

    Returns:
        List of CodeChunk objects
    """
    from pathlib import Path

    if project_path is None:
        project_path = str(Path(file_path).parent)

    parser = CodeParser(project_path)
    return parser.parse_file(Path(file_path))

"""
Preprocessing module exports - Enhanced with advanced ChunkPreprocessor
"""

from .chunk import (
    CodeChunk,
    ChunkPreprocessor,  # Enhanced version
    ChunkPreprocessor_2  # Original version for backward compatibility
)

__all__ = [
    'CodeChunk',
    'ChunkPreprocessor',
    'ChunkPreprocessor_2'
]
