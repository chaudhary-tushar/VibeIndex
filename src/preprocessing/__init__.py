"""
Preprocessing module for RAG indexing pipeline
"""

from .analyzer import Analyzer
from .chunk import CodeChunk
from .dependency_mapper import DependencyMapper
from .language_config import LanguageConfig
from .metadata_extractor import MetadataExtractor
from .parser import CodeParser

__all__ = ["Analyzer", "CodeChunk", "CodeParser", "DependencyMapper", "LanguageConfig", "MetadataExtractor"]


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

from .chunk import ChunkPreprocessor  # Enhanced version
from .chunk import ChunkPreprocessor_2  # Original version for backward compatibility
from .chunk import CodeChunk

__all__ = ["ChunkPreprocessor", "ChunkPreprocessor_2", "CodeChunk"]
