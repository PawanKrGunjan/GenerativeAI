# 📚 Chunking with LangChain

> Practical guide to effective text chunking strategies using LangChain — from basics to advanced production techniques that significantly improve RAG performance.

## 🔹 Why Chunking Matters

Chunking = splitting large documents into smaller, semantically meaningful pieces.

Good chunking helps to:

- Fit within LLM context windows
- Create higher-quality embeddings
- Improve retrieval precision in vector search
- Preserve semantic coherence
- Reduce hallucinations in RAG pipelines

**Bad chunking** → fragmented context → incomplete or incorrect answers.

## 📊 Chunking Strategies Overview

| Strategy                        | Semantic Awareness | Speed     | Precision   | Context Quality | Typical Use Case                        | LangChain Class                              |
|---------------------------------|--------------------|-----------|-------------|------------------|-----------------------------------------|----------------------------------------------|
| Fixed Character                 | Low                | Very Fast | Low         | Poor            | Quick tests, uniform text               | `CharacterTextSplitter`                      |
| Recursive Character             | Medium             | Fast      | Medium      | Good            | Most general-purpose RAG                | `RecursiveCharacterTextSplitter` (⭐ default) |
| Token-based                     | Medium             | Fast      | Medium      | Good            | Token-sensitive models                  | `TokenTextSplitter`                          |
| Markdown / Header-aware         | High (structure)   | Fast      | High        | Good            | Docs, wikis, READMEs, notebooks         | `MarkdownHeaderTextSplitter`                 |
| Semantic (embedding-based)      | High               | Medium    | High        | Very Good       | Research papers, technical content      | `SemanticChunker`                            |
| Parent-Child (small-to-large)   | High               | Medium    | Very High   | Excellent       | Precision + rich context needed         | `ParentDocumentRetriever`                    |
| Proposition / Fact-level        | Very High          | Slow      | Highest     | Good            | Legal, medical, financial documents     | Custom + LLM                                 |
| Sliding Window (high overlap)   | Medium–High        | Fast      | Medium–High | Excellent       | Transcripts, conversations, logs        | `RecursiveCharacterTextSplitter` + large overlap |

## 🚀 Recommended Production Approaches

### 1. The Reliable Default (used in ~80% of cases)

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,           # 15–25% is usually ideal
    length_function=len,         # or use tiktoken for precise token count
    separators=["\n\n", "\n", ". ", " ", ""],
    add_start_index=True         # useful for source referencing / citation
)

# For documents with metadata (recommended)
chunks = splitter.split_documents(docs)
```

→ Tries to keep headings, paragraphs, sentences intact when possible.

### 2. Parent-Child Retrieval (Highly recommended for serious RAG)

Index **small, precise child chunks** → retrieve → return **larger parent context** to the LLM.

```python
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain_community.vectorstores import Chroma  # or FAISS, Qdrant, etc.
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

child_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)

vectorstore = Chroma(embedding_function=OpenAIEmbeddings())
store = InMemoryStore()

retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)

retriever.add_documents(your_documents)
```

→ Offers the best balance between retrieval precision and context richness.

### 3. Semantic Chunking (When meaning matters more than fixed size)

```python
from langchain_text_splitters import SemanticChunker
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()
chunker = SemanticChunker(
    embeddings,
    breakpoint_threshold_type="percentile",   # alternatives: "standard_deviation", "interquartile"
    breakpoint_threshold_amount=85            # tune between 80–95
)

chunks = chunker.split_text(long_text)
# or: chunker.create_documents([long_text])
```

Great for research papers, mixed-topic documents, and cases where topic shifts are important.

### 4. High-Overlap Sliding Window (Conversational / Sequential Content)

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=400,           # 40–50% overlap
)
```

Ideal for: meeting transcripts, podcasts, chat logs, streaming logs, lecture notes.

### 5. Structure-Aware Splitting (Markdown, structured PDFs)

```python
from langchain_text_splitters import MarkdownHeaderTextSplitter

markdown_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
        ("####", "Header 4"),
    ],
    strip_headers=False,   # keep headers in content
)

chunks = markdown_splitter.split_text(markdown_content)
```

→ Each chunk automatically carries section metadata.

## 📏 Quick Reference – Chunk Size Guidelines

| Content Type           | Child Chunk Size | Parent / Full Size | Overlap       | Recommended Strategy                  |
|------------------------|------------------|--------------------|---------------|----------------------------------------|
| General web/articles   | —                | 800–1500 tokens    | 150–300       | Recursive + overlap                    |
| Research / technical   | 300–600          | 1200–2500          | 20–30%        | Semantic + Parent-Child                |
| Legal / contracts      | 200–500          | Full section       | 150–400       | Parent-Child (+ proposition if needed) |
| Code                   | 400–800          | —                  | 50–150        | Recursive, respect code blocks         |
| Conversations / audio  | —                | 600–1200           | 300–600       | Sliding window 40–50%                  |

## 🏁 Final Recommendations

1. **Start simple**  
   `RecursiveCharacterTextSplitter` + 150–250 overlap + `split_documents()` (preserves metadata)

2. **Level up when needed**  
   Switch to **Parent-Document Retriever** (small children + larger parents)

3. **Need maximum semantic fidelity?**  
   Combine **SemanticChunker** + Parent-Child or explore proposition-based chunking

4. **Always remember**  
   - Preserve metadata  
   - Use overlap (almost never 0)  
   - Evaluate with real queries (recall, context sufficiency, answer quality)

**Chunking is often the single biggest lever** for improving RAG accuracy and reducing hallucinations.
