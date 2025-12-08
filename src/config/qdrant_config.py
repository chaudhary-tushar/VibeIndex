"""
Qdrant configuration with connectivity check
"""

from pydantic import Field
from pydantic_settings import BaseSettings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance

from .settings import settings


class QdrantConfig(BaseSettings):
    """
    Configuration for Qdrant vector database
    """

    host: str = Field(default_factory=lambda: settings.qdrant_host, description="Qdrant server host")
    port: int = Field(default_factory=lambda: settings.qdrant_port, description="Qdrant server port")
    qdrant_api_key: str = Field(
        default_factory=lambda: settings.qdrant_api_key, description="API key for Qdrant Cloud (optional)"
    )
    collection_prefix: str = Field(
        default_factory=lambda: settings.project_name, description="Prefix for collection names"
    )

    # Vector configuration
    distance_metric: Distance = Field(default=Distance.COSINE, description="Distance metric for vector similarity")

    # Indexing configuration
    enable_sparse_vectors: bool = Field(default=True, description="Enable sparse vector indexing")
    enable_payload_index: bool = Field(default=True, description="Enable payload field indexing for filtering")

    # Batch processing
    batch_size: int = Field(default=100, description="Batch size for indexing operations")

    # Memory and performance settings
    on_disk_vectors: bool = Field(default=False, description="Store vectors on disk instead of memory")
    on_disk_sparse_vectors: bool = Field(default=False, description="Store sparse vectors on disk")

    def get_collection_names(self) -> list:
        """Get list of all collection names based on prefix"""
        return [
            f"{self.collection_prefix}_functions",
            f"{self.collection_prefix}_classes",
            f"{self.collection_prefix}_modules",
        ]

    def get_connection_url(self) -> str:
        """Get Qdrant connection URL"""
        if self.qdrant_api_key:
            return f"https://{self.host}:{self.port}"
        return f"http://{self.host}:{self.port}"

    def ping(self) -> bool:
        """
        Check if the Qdrant service is reachable by checking the service health
        """
        # Create a Qdrant client instance
        client = QdrantClient(
            host=self.host, port=self.port, api_key=self.qdrant_api_key if self.qdrant_api_key else None
        )

        try:
            # Try to get the cluster info to verify connection
            client.get_collections()
        except Exception:
            return False
        else:
            return True

    def get_client(self) -> QdrantClient:
        """
        Get a Qdrant client instance with the configured settings
        """
        return QdrantClient(
            host=self.host, port=self.port, api_key=self.qdrant_api_key if self.qdrant_api_key else None
        )
