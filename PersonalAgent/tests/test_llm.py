import numpy as np

from Agent.llm import embed_text, embeddings_model, llm


def test_llm_responds():
    response = llm.invoke("Say exactly: TEST_OK_42")
    assert "TEST_OK_42" in response.content


def test_embedding_dimension():
    text = "short sentence for testing embedding dimension"
    emb = embed_text(text)
    assert isinstance(emb, np.ndarray)
    assert emb.dtype == np.float32
    assert emb.shape[0] in {768, 1024, 2048}  # common nomic dimensions


def test_embeddings_model_loaded():
    assert embeddings_model is not None
    assert hasattr(embeddings_model, "embed_query")
