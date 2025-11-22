"""
Consolidated Qdrant configuration
Combines Qdrant-related settings from multiple sources
"""
from pydantic import Field, ConfigDict
from pydantic_settings import BaseSettings
from qdrant_client.models import Distance


class QdrantConfig(BaseSettings):
    """
    Consolidated configuration for Qdrant vector database

    This class combines settings from:
    - src/retrieval/search.py (QdrantConfig dataclass)
    - old_code/code_rag.py (Qdrant connection configuration)
    - config/settings.py (qdrant-related settings)
    """

    # Connection configuration
    host: str = Field(
        default="localhost",
        description="Qdrant server host"
    )
    port: int = Field(
        default=6333,
        description="Qdrant server port"
    )
    qdrant_api_key: str = Field(
        default="",
        description="API key for Qdrant Cloud (optional)"
    )

    # Collection configuration
    collection_prefix: str = Field(
        default="tipsy",
        description="Prefix for collection names"
    )

    # Vector configuration
    distance_metric: Distance = Field(
        default=Distance.COSINE,
        description="Distance metric for vector similarity"
    )

    # Indexing configuration
    enable_sparse_vectors: bool = Field(
        default=True,
        description="Enable sparse vector indexing"
    )
    enable_payload_index: bool = Field(
        default=True,
        description="Enable payload field indexing for filtering"
    )

    # Batch processing
    batch_size: int = Field(
        default=100,
        description="Batch size for indexing operations"
    )

    # Memory and performance settings
    on_disk_vectors: bool = Field(
        default=False,
        description="Store vectors on disk instead of memory"
    )
    on_disk_sparse_vectors: bool = Field(
        default=False,
        description="Store sparse vectors on disk"
    )

    model_config = ConfigDict(env_file=".env", case_sensitive=True, extra='ignore')

    def get_collection_names(self) -> list:
        """Get list of all collection names based on prefix"""
        return [
            f"{self.collection_prefix}_functions",
            f"{self.collection_prefix}_classes",
            f"{self.collection_prefix}_modules"
        ]

    def get_connection_url(self) -> str:
        """Get Qdrant connection URL"""
        if self.qdrant_api_key:
            return f"https://{self.host}:{self.port}"
        else:
            return f"http://{self.host}:{self.port}"

    def get_client_config(self) -> dict:
        """Get configuration dictionary for QdrantClient"""
        config = {"host": self.host, "port": self.port}
        if self.qdrant_api_key:
            config["api_key"] = self.qdrant_api_key
        return config