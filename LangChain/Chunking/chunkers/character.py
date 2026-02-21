# chunkers/character.py
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter


def chunk_simple_character(
    docs: List[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 50
) -> List[Document]:
    """Naive fixed-size character splitting — may cut mid-sentence."""
    splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separator="\n\n",
    )
    return splitter.split_documents(docs)


if __name__ == "__main__":
    from utils.pdf_loader import load_pdf_documents, print_chunks, get_pdf_path

    pdf_path = get_pdf_path()
    docs = load_pdf_documents(pdf_path)
    chunks = chunk_simple_character(docs)
    print_chunks(chunks, "CharacterTextSplitter (simple)")