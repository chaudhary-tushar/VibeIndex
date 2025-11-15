from pydantic import BaseSettings

class Settings(BaseSettings):
    qdrant_host: str
    qdrant_port: int
    qdrant_api_key: str
    embedding_model_name: str
    embedding_dim: int
    generation_model_name: str

    class Config:
        env_file = ".env"

settings = Settings()


