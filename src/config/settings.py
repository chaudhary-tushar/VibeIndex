from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Embedding Configuration
    embedding_model_url: str
    embedding_model_name: str
    embedding_dim: int

    # LLM Generation Configuration
    generation_model_url: str
    generation_model_name: str

    # Qdrant Configuration
    qdrant_host: str
    qdrant_port: int
    qdrant_api_key: str

    # Neo4j Configuration
    neo4j_uri: str
    neo4j_username: str
    neo4j_password: str

    # Project path provided during execution
    project_path: Path | None = None

    class Config:
        env_file = ".env"


settings = Settings()
