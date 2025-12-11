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

    # HNSW index configuration parameters
    hnsw_m: int = Field(default=16, description="Number of edges per node in the index graph (m parameter)")
    hnsw_ef_construct: int = Field(default=100, description="Number of neighbours to consider during index building")
    hnsw_ef: int = Field(default=100, description="Number of candidates to consider during search (ef parameter)")
    full_scan_threshold: int = Field(
        default=10000, description="Threshold in KB below which full-scan is preferred over index-based search"
    )

    # Optimizer configuration
    deleted_threshold: float = Field(
        default=0.2, description="Threshold for ratio of deleted points that triggers cleanup"
    )
    vacuum_min_vector_number: int = Field(
        default=1000, description="Minimum number of vectors to trigger vacuum operation"
    )
    default_segment_number: int = Field(default=2, description="Number of segments for initial collection creation")
    max_segment_size: int = Field(default=50000, description="Maximum size of a single segment in kilobytes")
    memmap_threshold: int = Field(default=50000, description="Memory map segments larger than this threshold (in KB)")
    indexing_threshold: int = Field(default=20000, description="Number of vectors to keep in-memory before indexing")

    # New configuration options for the enhanced indexing features
    # Performance monitoring and health checks
    enable_performance_monitoring: bool = Field(
        default=True, description="Enable performance metrics collection and monitoring"
    )
    enable_collection_health_check: bool = Field(
        default=True, description="Enable automatic health checks for all collections"
    )
    collection_health_check_interval: int = Field(
        default=3600, description="Interval (in seconds) to perform collection health checks"
    )

    # Indexing strategies
    enable_adaptive_batch_sizing: bool = Field(
        default=True, description="Enable adaptive batch sizing based on collection size"
    )
    adaptive_batch_size_multiplier: int = Field(
        default=2, description="Multiplier for batch size based on collection characteristics"
    )
    max_batch_size: int = Field(default=500, description="Maximum batch size for uploading operations")

    # Search performance
    enable_optimized_prefetch: bool = Field(
        default=True, description="Enable optimized prefetch strategies for hybrid search"
    )
    default_prefetch_limit: int = Field(default=100, description="Default prefetch limit for search operations")

    # Quality-based filtering
    default_quality_threshold: str = Field(
        default="high", description="Default quality threshold for filtering results"
    )
    enable_quality_scoring: bool = Field(default=True, description="Enable quality scoring for indexed chunks")

    # Payload indexing configuration
    enable_nested_field_indexing: bool = Field(
        default=True, description="Enable indexing of nested fields in metadata and context"
    )
    enable_full_text_indexing: bool = Field(
        default=True, description="Enable full-text indexing for code and docstrings"
    )
    enable_custom_field_indexing: bool = Field(
        default=True, description="Enable indexing of custom fields defined in configuration"
    )

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
