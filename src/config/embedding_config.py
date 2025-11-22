"""
Consolidated embedding configuration
Combines embedding-related settings from multiple sources
"""
from pydantic import Field, ConfigDict, model_validator
from pydantic_settings import BaseSettings
from typing import Optional, Any


class EmbeddingConfig(BaseSettings):
    """
    Consolidated configuration for embedding generation

    This class combines settings from:
    - src/embedding/embedder.py (EmbeddingConfig dataclass)
    - old_code/code_rag.py (EmbeddingConfig usage)
    - config/settings.py (embedding-related settings)
    """

    # Model configuration
    model_url: str = Field(
        default="http://localhost:12434/engines/llama.cpp/v1",
        description="API endpoint for embedding model (Ollama/Docker)"
    )
    model_name: str = Field(
        default="ai/embeddinggemma",
        description="Name of the embedding model"
    )

    # Embedding configuration
    embedding_dim: int = Field(
        default=768,
        description="Dimensionality of the embedding vectors"
    )

    # Processing configuration
    batch_size: int = Field(
        default=32,
        description="Number of texts to process in each batch"
    )
    max_retries: int = Field(
        default=3,
        description="Maximum number of retry attempts for failed embeddings"
    )
    timeout: int = Field(
        default=30,
        description="Timeout in seconds for embedding requests"
    )

    # Additional settings from old_code implementation
    embedding_model_name: str = Field(
        default="ai/embeddinggemma",
        alias="embedding_model_name",
        description="Alternative name for embedding model (from config/settings.py)"
    )

    model_config = ConfigDict(env_file=".env", case_sensitive=True, extra='ignore')

    @model_validator(mode='before')
    @classmethod
    def handle_backward_compatibility(cls, values: Any) -> Any:
        if isinstance(values, dict):
            if 'model_name' in values:
                values.setdefault('embedding_model_name', values['model_name'])
            elif 'embedding_model_name' in values:
                values.setdefault('model_name', values['embedding_model_name'])
        return values

    @property
    def effective_model_name(self) -> str:
        """Get the effective model name, preferring model_name over embedding_model_name"""
        return self.model_name or self.embedding_model_name

    def get_api_endpoint(self) -> str:
        """Get the full API endpoint for embeddings"""
        return f"{self.model_url}/embeddings"