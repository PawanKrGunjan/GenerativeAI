import pandas as pd
import hashlib
from pathlib import Path
from typing import Optional
from docx import Document
from langchain_community.document_loaders import CSVLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_experimental.text_splitter import SemanticChunker
from .llm import embeddings_model
from .config import CHUNK_OVERLAP, CHUNK_SIZE, DATA_DIR, DB_CONFIG, EMBEDDING_DIM
import logging


# text_splitter = SemanticChunker(embeddings_model, buffer_size=50, breakpoint_threshold_type='standard_deviation')

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    length_function=len,
    separators=["\n\n", "\n", " ", ""],
)


def compute_file_hash(filepath: Path | str, log: Optional[logging.Logger]) -> str:
    filepath = Path(filepath)
    hasher = hashlib.sha256()
    log.info(f"Opening {filepath}")
    with filepath.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_document_content(filepath: Path | str, log: Optional[logging.Logger]) -> str:
    filepath = Path(filepath)
    ext = filepath.suffix.lower()
    log.info(f"Reading {filepath}")

    try:
        if ext == ".pdf":
            loader = PyPDFLoader(str(filepath))
            docs = loader.load()
            return "\n".join(doc.page_content for doc in docs)

        elif ext == ".csv":
            loader = CSVLoader(str(filepath))
            docs = loader.load()
            return "\n".join(doc.page_content for doc in docs)

        elif ext in (".xlsx", ".xls"):
            df = pd.read_excel(filepath)
            return df.to_string(index=False)

        elif ext in (".txt", ".md"):
            return filepath.read_text(encoding="utf-8", errors="replace")

        elif ext == ".docx":
            doc = Document(filepath)
            return "\n".join(p.text for p in doc.paragraphs)

        else:
            log.warning("Unsupported: %s", ext)
            return ""

    except Exception as e:
        log.error("Load failed %s: %s", filepath.name, e)
        raise RuntimeError(f"Load failed {filepath.name}: {e}")