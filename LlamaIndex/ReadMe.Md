# LlamaIndex Quick Reference Notes  


This practical cheat-sheet covers **LlamaIndex** (formerly GPT Index), the leading data framework for building **RAG (Retrieval-Augmented Generation)** applications and **agentic systems** over private or domain-specific data.

LlamaIndex provides tools to ingest, index, query, and orchestrate data with LLMs. It excels at connecting LLMs to your documents, databases, APIs, and structured data while offering advanced routing, query engines, and multi-agent workflows.

**Key Highlights (2026 status):**
- Core focus: RAG pipelines, data connectors, advanced indexing (vector, knowledge graph, summary)
- Seamless integration with LangChain, LangGraph, CrewAI, etc.
- Supports 150+ data sources (PDFs, APIs, databases, Notion, Slack, etc.)
- Advanced features: Sub-question decomposition, multi-document agents, evaluation, tracing
- Production-ready: LlamaCloud, LlamaParse (PDF parsing), observability
- Main Repo: https://github.com/run-llama/llama_index
- Docs: https://docs.llamaindex.ai
- Install: `pip install llama-index`

---

### 1. Basic Setup (Python)

```python
# Install core + extras
pip install llama-index
pip install llama-index-embeddings-openai
pip install llama-index-llms-openai

import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

# Global settings (recommended)
os.environ["OPENAI_API_KEY"] = "your_key"

Settings.llm = OpenAI(model="gpt-4o-mini")
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

# Optional: Local models via Ollama
# pip install llama-index-llms-ollama llama-index-embeddings-ollama
# from llama_index.llms.ollama import Ollama
# from llama_index.embeddings.ollama import OllamaEmbedding
# Settings.llm = Ollama(model="llama3.2")
# Settings.embed_model = OllamaEmbedding(model_name="llama3.2")
```

---

### 2. Loading & Indexing Documents (Basic RAG)

```python
# Load documents (supports PDFs, TXT, Markdown, etc.)
documents = SimpleDirectoryReader("data/").load_data()  # Folder path

# Build vector index
index = VectorStoreIndex.from_documents(documents)

# Create query engine
query_engine = index.as_query_engine(similarity_top_k=5)

# Query
response = query_engine.query("What are the key trends in AI agents in 2026?")
print(response)
```

**When to use:** Quick RAG over a folder of documents.

---

### 3. Advanced Indexing Strategies

```python
from llama_index.core.indices import SummaryIndex, KnowledgeGraphIndex

# Summary Index (good for long documents)
summary_index = SummaryIndex.from_documents(documents)

# Knowledge Graph Index (extracts entities & relations)
kg_index = KnowledgeGraphIndex.from_documents(
    documents,
    kg_triplet_extract_fn=lambda x: ...,  # Custom extractor
    max_triplets_per_chunk=10
)

# Hybrid: List + Vector for routing
from llama_index.core import Indices
indices = [summary_index, vector_index]
query_engine = IndicesRouterQueryEngine.from_indices(indices)
```

**When to use:**
- **VectorStoreIndex** → Semantic search (most common)
- **SummaryIndex** → High-level overviews
- **KnowledgeGraphIndex** → Reasoning over relationships

---

### 4. Query Engines & Routing

```python
# Sub-question decomposition (for complex queries)
from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.core.tools import QueryEngineTool

tools = [
    QueryEngineTool.from_query_engine(vector_index.as_query_engine(), description="Search technical docs"),
    QueryEngineTool.from_query_engine(summary_index.as_query_engine(), description="Get summaries")
]
sub_query_engine = SubQuestionQueryEngine.from_tools(tools)

response = sub_query_engine.query("Compare LlamaIndex vs LangChain in 2026")
```

**Multi-document agents:**

```python
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool

def multiply(a: int, b: int) -> int:
    return a * b

tool = FunctionTool.from_defaults(fn=multiply)

agent = ReActAgent.from_tools([tool], llm=Settings.llm, verbose=True)
agent.chat("What is 7 * 8?")
```

---

### 5. Data Connectors & Structured Data

```python
from llama_index.readers.database import DatabaseReader

# Query SQL databases
reader = DatabaseReader(uri="postgresql://user:pass@localhost:5432/db")
docs = reader.load_data(query="SELECT * FROM employees")

# Notion, Google Docs, Slack, etc. — 150+ connectors
```

**LlamaParse** (for complex PDFs):

```python
# pip install llama-parse
from llama_parse import LlamaParse

parser = LlamaParse(result_type="markdown")
documents = parser.load_data("complex_invoice.pdf")
```

---

### 6. Advanced Features

- **LlamaCloud** → Managed indexing & retrieval (production)
- **Evaluation** → Faithfulness, relevance, correctness metrics
- **Tracing** → Integration with LangSmith, Phoenix, etc.
- **Workflows** → Event-driven pipelines (similar to LangGraph)
- **Multi-modal** → Image, audio indexing support

---

### 7. Best Practices Summary

| Goal                                | Recommended Approach                                      |
|-------------------------------------|-----------------------------------------------------------|
| Simple document RAG                 | SimpleDirectoryReader + VectorStoreIndex                  |
| Complex PDFs/tables                 | LlamaParse + structured extraction                        |
| Multi-source data                   | Multiple readers + router query engine                    |
| Complex reasoning queries           | SubQuestionQueryEngine or ReActAgent                      |
| Structured data (SQL, APIs)         | DatabaseReader or custom connectors                       |
| Production deployment               | LlamaCloud + observability + evaluation                    |

---

### 8. Recommended Models & Embeddings (2026)

| Type       | Model Example                     | Use Case                          |
|------------|-----------------------------------|-----------------------------------|
| LLM        | gpt-4o-mini, gpt-4o               | Best quality/cost                 |
| Local LLM  | llama3.2, granite-3.3             | Privacy/local                     |
| Embedding  | text-embedding-3-small/large      | Default choice                    |
| Open       | nomic-embed-text-v1.5             | Open-source alternative           |

---

**You now have a solid LlamaIndex foundation!**  
LlamaIndex is the go-to framework for **connecting LLMs to your data** — especially when building reliable, production RAG systems.

Explore starters: https://github.com/run-llama/llama_index/tree/main/examples  
Tutorials: https://docs.llamaindex.ai/en/stable/examples/


Happy indexing! 📚  


**Prepared By: Pawan Kumar Gunjan**  
**Date: January 09, 2026**