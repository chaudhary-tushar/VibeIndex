"""
Embedding configuration with connectivity check
"""

import requests
from pydantic import Field
from pydantic_settings import BaseSettings

from .settings import settings

BAD_REQUEST = 400


class EmbeddingConfig(BaseSettings):
    """
    Configuration for embedding generation services
    """

    model_url: str = Field(
        default_factory=lambda: settings.embedding_model_url, description="URL of the Embedding model"
    )
    model_name: str = Field(
        default_factory=lambda: settings.embedding_model_name, description="Name of Embedding model"
    )
    embedding_dim: int = Field(
        default_factory=lambda: settings.embedding_dim, description="Dimension of the Embedding Model"
    )
    timeout: int = Field(default=30, description="Timeout in seconds for embedding requests")
    max_retries: int = Field(default=3, description="Number of retries for failed requests")
    batch_size: int = Field(default=16, description="Size of Batch to process embedding at once")

    def ping(self) -> bool:
        """
        Check if the embedding service is reachable by sending a test embedding request
        """
        try:
            # Make a simple test request to the embedding service
            test_url = f"{self.model_url}/embeddings"
            test_payload = {"model": self.model_name, "input": "test"}

            response = requests.post(test_url, json=test_payload, timeout=self.timeout)

            # Check if the response is successful
            success = response.status_code in {200, 201, BAD_REQUEST}

            # For more accurate check, we might want to validate specific error codes
            if response.status_code == BAD_REQUEST:
                # If it's a bad request, the service is reachable but there might be invalid parameters
                # Check for the presence of error details in the response to distinguish from service not found
                try:
                    response.json()
                except requests.exceptions.JSONDecodeError:
                    # If we can't parse the JSON, it might be a real connectivity issue
                    return False
                else:
                    # If we get a structured error response, the service is reachable
                    return True
            else:
                return success

        except requests.exceptions.RequestException:
            return False
