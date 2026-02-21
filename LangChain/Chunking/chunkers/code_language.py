# chunkers/code_language.py
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language


PYTHON_EXAMPLE = """
def calculate_sum(a: int, b: int) -> int:
    \"\"\"Add two numbers.\"\"\"
    return a + b

if __name__ == "__main__":
    print(calculate_sum(40, 2))
""".strip()


def chunk_code_by_language(
    code: str,
    language: Language,
    chunk_size: int = 120,
    chunk_overlap: int = 20
) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter.from_language(
        language=language,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.create_documents([code])


if __name__ == "__main__":
    chunks = chunk_code_by_language(PYTHON_EXAMPLE, Language.PYTHON)
    for i, chunk in enumerate(chunks, 1):
        print(f"Code Chunk {i}:\n{chunk.page_content}\n{'-'*60}")