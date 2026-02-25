"""
LLM and embedding model initialization + fact extraction chain
"""

from typing import Any

import numpy as np
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_ollama import ChatOllama, OllamaEmbeddings

from .config import EMBEDDING_MODEL, MODEL_NAME
from .schemas import ExtractedFacts

# ────────────────────────────────────────────────
# Global instances (lazy-loaded / singletons)
# ────────────────────────────────────────────────


def get_llm(temperature: float = 0.0, **kwargs: Any) -> ChatOllama:
    """Get or create the main chat model instance"""
    return ChatOllama(model=MODEL_NAME, temperature=temperature, **kwargs)


def get_embeddings() -> OllamaEmbeddings:
    """Get or create the embedding model instance"""
    return OllamaEmbeddings(model=EMBEDDING_MODEL)


# Create default instances (most code will use these)
llm: ChatOllama = get_llm(temperature=0.0)
embeddings_model: OllamaEmbeddings = get_embeddings()


# ────────────────────────────────────────────────
# Fact extraction chain
# ────────────────────────────────────────────────

extraction_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a strict fact extractor.
Rules:
- Extract ONLY explicit, clearly stated facts from the message.
- Do NOT infer, guess, interpret or hallucinate anything.
- Use very high confidence only when the fact is 100% literal.
- Return empty list if no clear facts are present.

Examples of keys: name, age, location, profession, salary, date_of_joining, company, etc.

Output format: JSON list of facts or empty list.""",
        ),
        ("human", "{message}"),
    ]
)

# We use .with_structured_output → requires model to support it well
extraction_chain: Runnable = extraction_prompt | llm.with_structured_output(
    ExtractedFacts
)


def embed_text(text: str) -> np.ndarray:
    """
    Embed a piece of text using the current embedding model.
    Returns a numpy float32 array (standard for vector DBs).
    """
    try:
        vec = embeddings_model.embed_query(text)
        return np.array(vec, dtype=np.float32)
    except Exception as exc:
        raise RuntimeError(f"Embedding failed: {exc}") from exc
