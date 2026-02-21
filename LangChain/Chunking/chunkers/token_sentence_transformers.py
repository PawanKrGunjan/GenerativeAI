# chunkers/token_sentence_transformers.py
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import SentenceTransformersTokenTextSplitter


def chunk_sentence_transformers(
    text: str,
    model_name: str = "all-MiniLM-L6-v2",
    tokens_per_chunk: int = 180,
    chunk_overlap: int = 30
) -> List[Document]:
    """Token splitting using the exact tokenizer of a sentence-transformers model."""
    splitter = SentenceTransformersTokenTextSplitter(
        model_name=model_name,
        tokens_per_chunk=tokens_per_chunk,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_text(text)
    return [Document(page_content=c) for c in chunks]


if __name__ == "__main__":
    from utils.pdf_loader import load_pdf_documents, print_chunks, get_pdf_path

    pdf_path = get_pdf_path()
    docs = load_pdf_documents(pdf_path)
    sample_text = "\n\n".join(d.page_content for d in docs[:5])
    chunks = chunk_sentence_transformers(sample_text)
    print_chunks(chunks, "SentenceTransformersTokenTextSplitter")