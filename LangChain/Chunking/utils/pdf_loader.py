# utils/pdf_loader.py
import os
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader


def get_pdf_path(filename: str = "pawankrgunjan.pdf") -> str:
    """Return absolute path to PDF in current working directory."""
    return os.path.join(os.getcwd(), filename)


def load_pdf_documents(pdf_path: str) -> List[Document]:
    """Load pages from PDF using PyPDFLoader."""
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    return loader.load()


def print_chunks(chunks: List[Document], label: str = "Chunks", n: int = 3):
    """Pretty-print summary and preview of last N chunks."""
    print(f"\n{label}  ({len(chunks)} chunks total)")
    print("─" * 70)
    for i, chunk in enumerate(chunks[-n:], 1):
        preview = chunk.page_content.replace("\n", " ").strip()[:160]
        if len(preview) < len(chunk.page_content.replace("\n", " ").strip()):
            preview += "..."
        print(f"Chunk {i}   |   {len(chunk.page_content):4d} chars")
        print(f"   {preview}\n")