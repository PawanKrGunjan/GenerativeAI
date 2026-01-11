import os
import time
import re
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from sklearn.datasets import fetch_20newsgroups
from pprint import pprint

# Suppress unnecessary warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 3 = ERROR only (even stricter than 2)
import warnings
warnings.filterwarnings('ignore')

# ──────────────────────────────────────────────────────────────
# 1. Load the 20 Newsgroups dataset
# ──────────────────────────────────────────────────────────────
print("Loading 20 Newsgroups dataset...")
newsgroups = fetch_20newsgroups(subset='all')
documents = newsgroups.data

print(f"→ Loaded {len(documents):,} documents\n")

# Optional: Show a few sample posts
for i in range(3):
    print(f"Sample post {i+1}:")
    print("-" * 80)
    print(documents[i][:600] + "..." if len(documents[i]) > 600 else documents[i])
    print("\n")

# ──────────────────────────────────────────────────────────────
# 2. Preprocessing function
# ──────────────────────────────────────────────────────────────
def preprocess_text(text: str) -> str:
    """Minimal preprocessing - keep as much meaning as possible for embeddings"""
    # Remove email headers
    text = re.sub(r'^From:.*\n?', '', text, flags=re.MULTILINE)
    # Remove email addresses
    text = re.sub(r'\S*@\S*\s?', '', text)
    # Keep only letters and spaces (you can make this lighter if needed)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Lowercase & normalize whitespace
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text

print("Preprocessing documents...")
start = time.time()
processed_documents = [preprocess_text(doc) for doc in documents]
print(f"→ Preprocessing done in {time.time() - start:.1f} seconds\n")

# Show example
sample_idx = 0
print("Original sample:")
print(documents[sample_idx][:500] + "...")
print("\nPreprocessed sample:")
print(processed_documents[sample_idx])
print("-" * 80)

# ──────────────────────────────────────────────────────────────
# 3. Load embedding model (modern & strong choice in 2026)
# ──────────────────────────────────────────────────────────────
print("Loading embedding model...")
model = SentenceTransformer('BAAI/bge-small-en-v1.5', device='cpu')
# Alternatives you might want to compare later:
# model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')     # fastest
# model = SentenceTransformer('all-mpnet-base-v2', device='cpu')    # very strong

dimension = model.get_sentence_embedding_dimension()
print(f"→ Model: {model._first_module().auto_model.config._name_or_path}")
print(f"→ Embedding dimension: {dimension}\n")

# ──────────────────────────────────────────────────────────────
# 4. Generate embeddings (with normalization for cosine)
# ──────────────────────────────────────────────────────────────
print("Generating embeddings...")
start = time.time()

embeddings = model.encode(
    processed_documents,
    batch_size=32,
    show_progress_bar=True,
    normalize_embeddings=True,     # Very important for cosine similarity
    convert_to_numpy=True
)

print(f"→ Embeddings created in {time.time() - start:.1f} seconds")
print(f"→ Shape: {embeddings.shape}\n")

# ──────────────────────────────────────────────────────────────
# 5. Build FAISS index (cosine similarity via Inner Product)
# ──────────────────────────────────────────────────────────────
print("Building FAISS index...")
index = faiss.IndexFlatIP(dimension)           # Inner Product = cosine when normalized
index.add(embeddings.astype('float32'))

print(f"→ Index ready with {index.ntotal:,} vectors (cosine similarity)\n")

# ──────────────────────────────────────────────────────────────
# 6. Search function
# ──────────────────────────────────────────────────────────────
def semantic_search(query_text: str, k: int = 5):
    """Search for top-k most similar documents using cosine similarity"""
    preprocessed = preprocess_text(query_text)
    query_embedding = model.encode(
        [preprocessed],
        normalize_embeddings=True,
        convert_to_numpy=True
    )
    
    # scores: cosine similarity (higher = more similar)
    scores, indices = index.search(query_embedding.astype('float32'), k)
    return scores, indices

# ──────────────────────────────────────────────────────────────
# 7. Example search & display
# ──────────────────────────────────────────────────────────────
query = "motorcycle"

print(f"\nPerforming semantic search for:  '{query}'")
print("=" * 70)

scores, indices = semantic_search(query, k=5)

print("Top 5 most similar documents (cosine similarity):\n")
for rank, idx in enumerate(indices[0], 1):
    if idx == -1:
        continue
    score = scores[0][rank - 1]
    print(f"Rank {rank}  |  Score: {score:.4f}")
    print("-" * 70)
    # Show beginning of original document
    print(documents[idx][:800] + "..." if len(documents[idx]) > 800 else documents[idx])
    print("\n")

print("Search completed!")