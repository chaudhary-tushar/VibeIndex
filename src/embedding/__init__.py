"""
Embedding module exports - Enhanced with quality validation and batch processing
"""

from .embedder import EmbeddingGenerator  # Enhanced version with quality validation
from .embedder import EmbeddingGenerator2  # Original version for backward compatibility

__all__ = ["EmbeddingGenerator", "EmbeddingGenerator2"]
