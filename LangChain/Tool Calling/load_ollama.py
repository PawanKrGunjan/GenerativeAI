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
    MISTRAL: str = "mistral:latest"
    GRANITE: str = "granite4:350m"

    # Embeddings
    OLLAMA_EMBEDDING: str = "nomic-embed-text"


# -----------------------------
# Factory Class (Reusable)
# -----------------------------
class LargeLanguageModel:
    """Central factory for LLMs, chat models, and embeddings."""

    @staticmethod
    def llm(
        model: str = ModelRegistry.TINY_LLAMA,
        temperature: float = 0.1,
        max_tokens: int = 300,
    ) -> OllamaLLM:
        print(f"Loading {model} with Temperature :{temperature} & Max Tokens :{max_tokens}")
        return OllamaLLM(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    @staticmethod
    def chat(
        model: str = ModelRegistry.MISTRAL,
        temperature: float = 0.3,
        num_ctx: int = 500,
    ) -> ChatOllama:
        print(f"Loading {model} with Temperature :{temperature} & Max Charater Length :{num_ctx}")
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
        llm = chat_model or LargeLanguageModel.chat()

        prompt = PromptTemplate(
            input_variables=["question"],
            template="Answer clearly and concisely:\n{question}",
        )

        return prompt | llm | StrOutputParser()

# topic = "the life cycle of butterflies"
# prompt = PromptTemplate.from_template("""Write an engaging and educational story about {topic} for beginners. 
#         Use simple and clear language to explain basic concepts. 
#         Include interesting facts and keep it friendly and encouraging. 
#         The story should be around 200-300 words and end with a brief summary of what we learned. 
#         Make it perfect for someone just starting to learn about this topic.""")

# llm = LargeLanguageModel.llm()

# chain = prompt | llm | StrOutputParser()

# response = chain.invoke({"topic": topic})
# print(response)