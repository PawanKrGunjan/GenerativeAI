from Agent.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DATA_DIR,
    EMBEDDING_MODEL,
    MODEL_NAME,
    TOP_K,
)


def test_config_sane_values():
    assert MODEL_NAME in {"llama3.2:3b", "llama3.1:8b", "phi3:mini", "mistral"}
    assert EMBEDDING_MODEL == "nomic-embed-text:latest"
    assert 500 <= CHUNK_SIZE <= 1200
    assert 50 <= CHUNK_OVERLAP <= 300
    assert 3 <= TOP_K <= 10
    assert DATA_DIR.name == "data"
