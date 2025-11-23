"""
Configuration module for the application

This module provides consolidated configuration classes that combine settings
from multiple sources while maintaining backward compatibility.

Classes:
    EmbeddingConfig: Consolidated configuration for embedding generation
    QdrantConfig: Consolidated configuration for Qdrant vector database
"""

from .embedding_config import EmbeddingConfig
from .qdrant_config import QdrantConfig

__all__ = ["EmbeddingConfig", "QdrantConfig"]
