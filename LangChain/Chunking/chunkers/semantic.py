# chunkers/semantic.py
from typing import List
from langchain_core.documents import Document


def chunk_semantic(
    text: str,
    breakpoint_threshold_type: str = "percentile",
    breakpoint_threshold_amount: float = 88
) -> List[Document]:
    """Semantic splitting using embeddings (requires OpenAI API key)."""
    try:
        from langchain_openai import OpenAIEmbeddings
        from langchain_text_splitters import SemanticChunker

        embeddings = OpenAIEmbeddings()
        chunker = SemanticChunker(
            embeddings,
            breakpoint_threshold_type=breakpoint_threshold_type,
            breakpoint_threshold_amount=breakpoint_threshold_amount,
        )
        return chunker.create_documents([text])
    except Exception as e:
        print(f"Semantic chunking failed: {e}")
        print("(most likely missing OPENAI_API_KEY)")
        return []


if __name__ == "__main__":
    text = "Long document text here... " * 20  # placeholder
    chunks = chunk_semantic(text)
    if chunks:
        for i, c in enumerate(chunks[:3], 1):
            print(f"Semantic Chunk {i} ({len(c.page_content)} chars):")
            print(c.page_content[:150] + "...\n")