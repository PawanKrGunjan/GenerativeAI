# chunkers/token_openai.py
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import TokenTextSplitter


def chunk_openai_tokens(
    docs: List[Document],
    chunk_size: int = 400,
    chunk_overlap: int = 80
) -> List[Document]:
    """Token-based splitting using OpenAI's cl100k_base tokenizer."""
    splitter = TokenTextSplitter(
        encoding_name="cl100k_base",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_documents(docs)


if __name__ == "__main__":
    from utils.pdf_loader import load_pdf_documents, print_chunks, get_pdf_path

    pdf_path = get_pdf_path()
    docs = load_pdf_documents(pdf_path)
    chunks = chunk_openai_tokens(docs)
    print_chunks(chunks, "OpenAI TokenTextSplitter")