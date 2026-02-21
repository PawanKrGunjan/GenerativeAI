# chunkers/recursive.py
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_recursive(
    docs: List[Document],
    chunk_size: int = 700,
    chunk_overlap: int = 140
) -> List[Document]:
    """
    Recommended default splitter.
    Tries to respect paragraphs → sentences → words.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
        add_start_index=True,
    )
    return splitter.split_documents(docs)


if __name__ == "__main__":
    from utils.pdf_loader import load_pdf_documents, print_chunks, get_pdf_path

    pdf_path = get_pdf_path()
    docs = load_pdf_documents(pdf_path)
    chunks = chunk_recursive(docs)
    print_chunks(chunks, "RecursiveCharacterTextSplitter")