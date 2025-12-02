import os

import pytest
from pydantic import ValidationError
from qdrant_client.models import Distance

from src.config.embedding_config import EmbeddingConfig
from src.config.qdrant_config import QdrantConfig

# Constants for test magic values
DEFAULT_EMBEDDING_DIM = 768
DEFAULT_BATCH_SIZE = 32
DEFAULT_MAX_RETRIES = 3
DEFAULT_TIMEOUT = 30
CUSTOM_EMBEDDING_DIM = 1024
CUSTOM_BATCH_SIZE = 64
CUSTOM_MAX_RETRIES = 5
CUSTOM_TIMEOUT = 60
DEFAULT_QDRANT_PORT = 6333
CUSTOM_QDRANT_PORT = 6334
DEFAULT_QDRANT_BATCH_SIZE = 100
CUSTOM_QDRANT_BATCH_SIZE = 50

# --- Tests for EmbeddingConfig ---


def test_embedding_config_defaults():
    config = EmbeddingConfig(_env_file=None)
    assert config.model_url == "http://localhost:12434/engines/llama.cpp/v1"
    assert config.model_name == "ai/embeddinggemma"
    assert config.embedding_dim == DEFAULT_EMBEDDING_DIM
    assert config.batch_size == DEFAULT_BATCH_SIZE
    assert config.max_retries == DEFAULT_MAX_RETRIES
    assert config.timeout == DEFAULT_TIMEOUT
    assert config.embedding_model_name == "ai/embeddinggemma"


def test_embedding_config_env_vars(monkeypatch):
    monkeypatch.setenv("model_url", "http://custom-host:8000")
    monkeypatch.setenv("model_name", "custom-model")
    monkeypatch.setenv("embedding_dim", "1024")
    monkeypatch.setenv("batch_size", "64")
    monkeypatch.setenv("max_retries", "5")
    monkeypatch.setenv("timeout", "60")

    config = EmbeddingConfig(_env_file=None)
    assert config.model_url == "http://custom-host:8000"
    assert config.model_name == "custom-model"
    assert config.embedding_dim == CUSTOM_EMBEDDING_DIM
    assert config.batch_size == CUSTOM_BATCH_SIZE
    assert config.max_retries == CUSTOM_MAX_RETRIES
    assert config.timeout == CUSTOM_TIMEOUT
    assert config.embedding_model_name == "custom-model"


def test_embedding_config_init_backward_compatibility_model_name_only():
    config = EmbeddingConfig(model_name="new-model", _env_file=None)
    assert config.model_name == "new-model"
    assert config.embedding_model_name == "new-model"


def test_embedding_config_init_backward_compatibility_embedding_model_name_only():
    config = EmbeddingConfig(embedding_model_name="old-model", _env_file=None)
    assert config.model_name == "old-model"
    assert config.embedding_model_name == "old-model"


def test_embedding_config_init_backward_compatibility_both_provided():
    config = EmbeddingConfig(model_name="new-model", embedding_model_name="old-model-ignored", _env_file=None)
    assert config.model_name == "new-model"
    assert config.embedding_model_name == "old-model-ignored"


def test_embedding_config_effective_model_name():
    config = EmbeddingConfig(model_name="primary-model", embedding_model_name="fallback-model", _env_file=None)
    assert config.effective_model_name == "primary-model"

    config = EmbeddingConfig(model_name="", embedding_model_name="fallback-model", _env_file=None)
    assert config.effective_model_name == "fallback-model"


def test_embedding_config_get_api_endpoint():
    config = EmbeddingConfig(model_url="http://test.com:1234", _env_file=None)
    assert config.get_api_endpoint() == "http://test.com:1234/embeddings"


# --- Tests for QdrantConfig ---


def test_qdrant_config_defaults():
    config = QdrantConfig(_env_file=None)
    assert config.host == "localhost"
    assert config.port == DEFAULT_QDRANT_PORT
    assert not config.qdrant_api_key
    assert config.collection_prefix == "tipsy"
    assert config.distance_metric == Distance.COSINE
    assert config.enable_sparse_vectors is True
    assert config.enable_payload_index is True
    assert config.batch_size == DEFAULT_QDRANT_BATCH_SIZE
    assert config.on_disk_vectors is False
    assert config.on_disk_sparse_vectors is False


def test_qdrant_config_env_vars(monkeypatch):
    monkeypatch.setenv("host", "qdrant.cloud")
    monkeypatch.setenv("port", "6334")
    monkeypatch.setenv("qdrant_api_key", "test-api-key")
    monkeypatch.setenv("collection_prefix", "my-app")
    monkeypatch.setenv("distance_metric", "Euclid")
    monkeypatch.setenv("enable_sparse_vectors", "false")
    monkeypatch.setenv("enable_payload_index", "false")
    monkeypatch.setenv("batch_size", "50")
    monkeypatch.setenv("on_disk_vectors", "true")
    monkeypatch.setenv("on_disk_sparse_vectors", "true")

    config = QdrantConfig(_env_file=None)
    assert config.host == "qdrant.cloud"
    assert config.port == CUSTOM_QDRANT_PORT
    assert config.qdrant_api_key == "test-api-key"
    assert config.collection_prefix == "my-app"
    assert config.distance_metric == Distance.EUCLID
    assert config.enable_sparse_vectors is False
    assert config.enable_payload_index is False
    assert config.batch_size == CUSTOM_QDRANT_BATCH_SIZE
    assert config.on_disk_vectors is True
    assert config.on_disk_sparse_vectors is True


def test_qdrant_config_get_collection_names():
    config = QdrantConfig(collection_prefix="my_project", _env_file=None)
    expected_names = ["my_project_functions", "my_project_classes", "my_project_modules"]
    assert config.get_collection_names() == expected_names


def test_qdrant_config_get_connection_url_no_api_key():
    config = QdrantConfig(host="test-host", port=1234, qdrant_api_key="", _env_file=None)
    assert config.get_connection_url() == "http://test-host:1234"


def test_qdrant_config_get_connection_url_with_api_key():
    config = QdrantConfig(host="cloud-host", port=5678, qdrant_api_key="some-key", _env_file=None)
    assert config.get_connection_url() == "https://cloud-host:5678"


def test_qdrant_config_get_client_config_no_api_key():
    config = QdrantConfig(host="local", port=1111, qdrant_api_key="", _env_file=None)
    assert config.get_client_config() == {"host": "local", "port": 1111}


def test_qdrant_config_get_client_config_with_api_key():
    config = QdrantConfig(host="remote", port=2222, qdrant_api_key="remote-key", _env_file=None)
    assert config.get_client_config() == {"host": "remote", "port": 2222, "api_key": "remote-key"}


def test_qdrant_config_distance_metric_invalid_value(monkeypatch):
    monkeypatch.setenv("distance_metric", "INVALID_METRIC")
    with pytest.raises(ValidationError):
        QdrantConfig(_env_file=None)
