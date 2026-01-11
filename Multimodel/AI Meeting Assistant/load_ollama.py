from dataclasses import dataclass
from typing import Optional

from langchain_ollama import OllamaLLM, ChatOllama, OllamaEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


# -----------------------------
# Model Configuration
# -----------------------------

@dataclass(frozen=True)
class ModelRegistry:
    # LLMs
    TINY_LLAMA: str = "llama3.2:1b"
    FULL_LLAMA: str = "llama3.2:latest"
    QWEN_3B: str = "qwen2.5:3b"

    # Embeddings
    OLLAMA_EMBEDDING: str = "nomic-embed-text"


# -----------------------------
# Factory Class (Reusable)
# -----------------------------

class RAGFactory:
    """Central factory for LLMs, chat models, and embeddings."""

    @staticmethod
    def llm(
        model: str = ModelRegistry.TINY_LLAMA,
        temperature: float = 0.1,
        max_tokens: int = 300,
    ) -> OllamaLLM:
        return OllamaLLM(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    @staticmethod
    def chat(
        model: str = ModelRegistry.QWEN_3B,
        temperature: float = 0.5,
        num_ctx: int = 2048,
    ) -> ChatOllama:
        return ChatOllama(
            model=model,
            temperature=temperature,
            num_ctx=num_ctx,
        )

    @staticmethod
    def embeddings(
        model: str = ModelRegistry.OLLAMA_EMBEDDING,
    ) -> OllamaEmbeddings:
        return OllamaEmbeddings(model=model)


# -----------------------------
# Reusable Chains
# -----------------------------

class Chains:
    @staticmethod
    def simple_qa(chat_model: Optional[ChatOllama] = None):
        llm = chat_model or RAGFactory.chat()

        prompt = PromptTemplate(
            input_variables=["question"],
            template="Answer clearly and concisely:\n{question}",
        )

        return prompt | llm | StrOutputParser()
