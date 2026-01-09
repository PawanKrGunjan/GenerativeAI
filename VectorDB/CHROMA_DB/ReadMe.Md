# ChromaDB Quick Reference Notes  
**Prepared for: Pawan Kumar Gunjan**  
**Date: January 09, 2026**

This is a clear, practical cheat-sheet for using **ChromaDB** in Python, based on the latest stable practices (2024–2026). It corrects outdated syntax and highlights the **recommended modern approach**.

---

### 1. Basic Setup

```python
import chromadb
from chromadb.utils import embedding_functions

# Embedding function (recommended model for speed & quality)
embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"   # Fast, lightweight, excellent for semantic search
)

# Client types
client = chromadb.Client()                    # In-memory only (lost on restart)
client = chromadb.PersistentClient(path="./chroma_db")  # Saves to disk — use this!
```

---

### 2. Creating / Loading a Collection (Correct Ways)

#### Recommended & Safest Method (Use This!)

```python
collection = client.get_or_create_collection(
    name="employee_collection",
    embedding_function=embedding_function,
    metadata={
        "description": "Employee semantic search",
        "hnsw:space": "cosine"   # Best for text/semantic similarity
    }
)
```

**Why this is best:**
- Won’t crash if collection already exists
- Automatically uses **cosine similarity** (perfect for sentence transformers)
- Clean and future-proof

#### Alternative: Explicit get + create (still valid)

```python
try:
    collection = client.get_collection("employee_collection")
except:
    collection = client.create_collection(
        name="employee_collection",
        embedding_function=embedding_function,
        metadata={"hnsw:space": "cosine"}
    )
```

#### Avoid This (Outdated / Invalid)

```python
# WRONG — configuration= parameter does NOT exist!
configuration={"hnsw": {"space": "cosine"}, "embedding_function": ef}
```

**Correct place for distance metric:** in `metadata` → `"hnsw:space": "cosine"`

Supported distance types:
- `"cosine"` → Best for text & natural language (use 99% of the time)
- `"l2"` → Euclidean distance
- `"ip"` → Inner product

---

### 3. Adding Data

```python
collection.add(
    ids=["emp001", "emp002"],                    # Must be unique strings
    documents=["Software Engineer with Python skills...", "Data Scientist..."],
    metadatas=[
        {"name": "Pawan", "department": "AI/ML", "location": "Noida", "salary": 1200000},
        {"name": "Ravi", "department": "Engineering", "location": "Bengaluru", "salary": 2200000}
    ]
)
```

`.add()` automatically **upserts** — if ID exists, it updates.

---

### 4. Querying (Similarity Search)

```python
results = collection.query(
    query_texts=["Python developer in Noida"],
    n_results=5,
    include=["metadatas", "documents", "distances"]  # What to return
)

# Access results
for i in range(len(results["ids"][0])):
    print(f"Rank {i+1}")
    print(f"ID: {results['ids'][0][i]}")
    print(f"Name: {results['metadatas'][0][i]['name']}")
    print(f"Similarity: {1 - results['distances'][0][i]:.3f}")  # Convert distance → similarity
    print(f"Text: {results['documents'][0][i]}")
    print("---")
```

---

### 5. Filtering (Metadata + Document Content)

#### Metadata Filters (Exact / Structured)

```python
# Basic
collection.query(..., where={"department": "AI/ML"})

# Operators
where={
    "$and": [
        {"salary": {"$gte": 1500000}},
        {"location": {"$in": ["Noida", "Bengaluru"]}},
        {"experience": {"$gt": 5}}
    ]
}

# Available operators
"$eq", "$ne", "$gt", "$gte", "$lt", "$lte", "$in", "$nin"
```

#### Document Content Filters (Text inside documents)

```python
where_document={"$contains": "Python"}
where_document={"$not_contains": "Java"}
where_document={
    "$or": [
        {"$contains": "Machine Learning"},
        {"$contains": "Deep Learning"}
    ]
}
```

#### Combine Both

```python
results = collection.query(
    query_texts=["AI engineer"],
    n_results=10,
    where={"department": "AI/ML"},           # metadata filter
    where_document={"$contains": "LLM"}      # text filter
)
```

---

### 6. Retrieving All Data (No Search)

```python
all_data = collection.get()                    # Everything
filtered = collection.get(where={"location": "Noida"})
```

---

### 7. Best Practices Summary

| Goal                                | Recommended Code                                      |
|-------------------------------------|--------------------------------------------------------|
| Persistent storage                  | `PersistentClient(path="./chroma_db")`                |
| Safe collection handling            | `get_or_create_collection()`                          |
| Text/semantic search                | `"hnsw:space": "cosine"` + SentenceTransformer        |
| Adding/updating data                | Use `.add()` with unique `ids`                        |
| Filtering by exact fields           | Use `where=` with metadata                            |
| Filtering by keywords in text       | Use `where_document={"$contains": "..."}`             |
| Export all data                     | `collection.get(include=["metadatas", "documents", "ids"])` |

---

### 8. Common Models (Embedding Functions)

| Model                     | Size   | Speed | Quality | Use When                     |
|---------------------------|--------|-------|---------|------------------------------|
| `all-MiniLM-L6-v2`        | Small  | Fast  | Good    | Most projects (recommended)  |
| `all-mpnet-base-v2`       | Medium | Medium| Better  | Need higher accuracy         |
| `multi-qa-MiniLM-L6-cos-v1`| Small  | Fast  | Great for Q&A/search         |

---

