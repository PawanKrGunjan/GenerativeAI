"""
Agent/llm.py
LLM and embedding model initialization + fact extraction chain
"""

from typing import Any
import numpy as np
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_ollama import ChatOllama, OllamaEmbeddings

from .config import EMBEDDING_MODEL, MODEL_NAME, EMBEDDING_DIM  # add EMBEDDING_DIM
from .schemas import ExtractedFacts


def get_llm(temperature: float = 0.0, **kwargs: Any) -> ChatOllama:
    return ChatOllama(model=MODEL_NAME, temperature=temperature, **kwargs)


def get_embeddings() -> OllamaEmbeddings:
    return OllamaEmbeddings(model=EMBEDDING_MODEL)


llm: ChatOllama = get_llm(temperature=0.0)
embeddings_model: OllamaEmbeddings = get_embeddings()


def embed_text(text: str) -> np.ndarray:
    """
    Embed text using Ollama embeddings (nomic-embed-text).
    Returns numpy float32 vector of fixed dimension (EMBEDDING_DIM).
    """
    try:
        vec = embeddings_model.embed_query(text)  # returns list[float] [web:35]
        if len(vec) != EMBEDDING_DIM:
            raise ValueError(f"Expected {EMBEDDING_DIM} dims, got {len(vec)}")
        return np.asarray(vec, dtype=np.float32)
    except Exception as exc:
        raise RuntimeError(f"Embedding failed: {exc}") from exc


# extraction_prompt = ChatPromptTemplate.from_messages(
#     [
#         (
#             "system",
#             """Extract explicit facts as key-value pairs. Use confidence=1.0 for obvious facts.

# Examples (exact format):
# - "I am Rupam" → key="name", value="Rupam", confidence=1.0  
# - "I live in Patna" → key="location", value="Patna", confidence=1.0

# For "Hi, I am Rupam & You?": ONLY extract name="Rupam"
# Ignore "You?", punctuation.

# Output ONLY valid JSON array matching schema.""",
#         ),
#         ("human", "{message}"),
#     ]
# )

# extraction_chain = extraction_prompt | llm.with_structured_output(ExtractedFacts)
