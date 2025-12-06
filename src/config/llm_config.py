"""
LLM configuration with connectivity check
"""

import requests
from pydantic import ConfigDict
from pydantic import Field
from pydantic_settings import BaseSettings

from .settings import settings

BAD_REQUEST = 400  # 400 means it's reaching the service but has invalid parameters


class LLMConfig(BaseSettings):
    """
    Configuration for LLM generation services
    """

    model_url: str = Field(
        default_factory=lambda: settings.generation_model_url,
        description="API endpoint for LLM generation model (Ollama/Docker)",
    )
    model_name: str = Field(
        default_factory=lambda: settings.generation_model_name, description="Name of the LLM generation model"
    )
    model_provider: str = Field(
        default_factory=lambda: settings.generation_model_provider, description="Provider of the LLM API"
    )
    timeout: int = Field(default=60, description="Timeout in seconds for LLM requests")
    max_retries: int = Field(default=3, description="Number of retries for failed requests")

    model_config = ConfigDict(env_file=".env", case_sensitive=True, extra="ignore")

    def ping(self) -> bool:
        """
        Check if the LLM service is reachable by sending a test chat completion request
        """
        try:
            # Make a simple test request to the LLM service
            test_url = f"{self.model_url}/chat/completions"
            test_payload = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 5,
                "temperature": 0,
            }

            response = requests.post(test_url, json=test_payload, timeout=self.timeout)

            # Check if the response is successful
            # Status 200 means success, 400 might mean it's reaching the service but has invalid parameters
            success = response.status_code in {200, 201, BAD_REQUEST}

            # For more accurate check, if it's a bad request, verify it's actually reaching the service
            if response.status_code == BAD_REQUEST:
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
