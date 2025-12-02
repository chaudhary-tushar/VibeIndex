"""
Embedding configuration with connectivity check
"""

import requests
from pydantic import ConfigDict
from pydantic import Field
from pydantic_settings import BaseSettings


class EmbeddingConfig(BaseSettings):
    """
    Configuration for embedding generation services
    """

    model_url: str = Field(..., description="API endpoint for embedding model (Ollama/Docker)")
    model_name: str = Field(..., description="Name of the embedding model")
    embedding_dim: int = Field(..., description="Dimensionality of the embedding vectors")
    timeout: int = Field(default=30, description="Timeout in seconds for embedding requests")
    max_retries: int = Field(default=3, description="Number of retries for failed requests")

    model_config = ConfigDict(env_file=".env", case_sensitive=True, extra="ignore")

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
            success = response.status_code in {200, 201, 400}  # 400 means it's reaching the service

            # For more accurate check, we might want to validate specific error codes
            if response.status_code == 400:
                # If it's a bad request, the service is reachable but there might be invalid parameters
                # Check for the presence of error details in the response to distinguish from service not found
                try:
                    response_json = response.json()
                    # If we get a structured error response, the service is reachable
                    return True
                except Exception:
                    # If we can't parse the JSON, it might be a real connectivity issue
                    return False

            return success
        except Exception:
            return False
