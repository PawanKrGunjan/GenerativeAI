# chunkers/markdown_headers.py
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_core.documents import Document
from typing import List


def chunk_markdown_by_headers(md_content: str) -> List[Document]:
    """Split markdown respecting header hierarchy (preserves metadata)."""
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
        ("####", "Header 4"),
    ]
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False,
    )
    return splitter.split_text(md_content)


if __name__ == "__main__":
    # Example markdown
    example_md = """
# Introduction

This is intro text.

## Section One

Content of section one.

### Subsection

More details here.
    """.strip()
    chunks = chunk_markdown_by_headers(example_md)
    for i, chunk in enumerate(chunks, 1):
        print(f"--- Chunk {i} ---")
        print("Metadata:", chunk.metadata)
        print(chunk.page_content[:200])
        print()