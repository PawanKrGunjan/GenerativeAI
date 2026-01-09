import math
import numpy as np
import scipy
import torch
from sentence_transformers import SentenceTransformer

"""
COMPREHENSIVE SIMILARITY SEARCH NOTES
========================================
This notebook covers ALL major similarity metrics for vector embeddings:
1. L2/Euclidean Distance ✅
2. Cosine Similarity ✅ 
3. Dot Product ✅
4. Manhattan Distance
5. Optimized implementations
6. Retrieval pipeline
========================================
"""

# Example documents for semantic similarity testing
documents = [
    'The weather is lovely today.',
    r"It's so sunny outside!",           # Raw string prevents escape warning
    'He drove to the stadium.',         # Low similarity
    'She walked to the park.'           # Low similarity  
]

# Load pre-trained sentence transformer model (384-dim embeddings)
model = SentenceTransformer('paraphrase-MiniLM-L6-v2', device='cpu')
embeddings = model.encode(documents)

print(f"Embedding Shape: {embeddings.shape}")  # (4, 384)
print("Sample embedding (first 5 dims):", embeddings[0, :5])

## 1. L2 (EUCLIDEAN) DISTANCE
# $$ \text{L2}(a,b) = \sqrt{\sum_{i=1}^n (a_i - b_i)^2} $$
def euclidean_distance_fn(vector1, vector2):
    """
    Manual L2 (Euclidean) distance between two vectors
    PURPOSE: Verify against optimized implementations
    """
    squared_sum = sum((x - y) ** 2 for x, y in zip(vector1, vector2))
    return math.sqrt(squared_sum)

print(f"L2(doc0,doc1): {euclidean_distance_fn(embeddings[0], embeddings[1]):.3f}")

# Full distance matrix - COMPLETE IMPLEMENTATION (FIXED)
l2_dist = np.zeros([len(documents), len(documents)])
for i in range(embeddings.shape[0]):
    for j in range(embeddings.shape[0]):
        l2_dist[i,j] = euclidean_distance_fn(embeddings[i], embeddings[j])

# OPTIMIZED upper triangle only (symmetric matrix)
l2_dist1 = np.zeros([len(documents), len(documents)])
for i in range(embeddings.shape[0]):
    for j in range(embeddings.shape[0]):
        if j > i:  # Upper triangle calculation
            l2_dist1[i,j] = euclidean_distance_fn(embeddings[i], embeddings[j])
        elif i > j:  # Mirror symmetry
            l2_dist1[i,j] = l2_dist1[j,i]

# SCIKIT-LEARN OPTIMIZED (VECTORIZED)
l2_dist_scipy = scipy.spatial.distance.cdist(embeddings, embeddings, 'euclidean')
print("✅ All L2 implementations match:", np.allclose(l2_dist, l2_dist_scipy))

print("\nL2 Distance Matrix:")
print(l2_dist.round(3))
print(f"l2_dist[0,1]: {l2_dist[0,1]:.3f}")      # FIXED: Use correct variable name
print(f"l2_dist[1,0]: {l2_dist[1,0]:.3f}")


## 2. DOT PRODUCT (RAW SIMILARITY)
def dot_product_fn(vector1, vector2):
    """Raw dot product: measures magnitude + direction"""
    return sum(x * y for x, y in zip(vector1, vector2))

# Manual vs Matrix multiplication
dot_product_manual = np.empty([len(documents), len(documents)])
for i in range(embeddings.shape[0]):
    for j in range(embeddings.shape[0]):
        dot_product_manual[i,j] = dot_product_fn(embeddings[i], embeddings[j])

dot_product_operator = embeddings @ embeddings.T  # VECTORIZED
print("\n✅ Dot product matches:", np.allclose(dot_product_manual, dot_product_operator))

## 3. COSINE SIMILARITY (MOST IMPORTANT FOR TEXT)
"""
$$ \text{cossim}(a, b) = \frac{a \cdot b}{||a|| \ ||b||} = \frac{a}{||a||} \cdot \frac{b}{||b||} $$
KEY: Works with normalized vectors (unit length)
"""

# L2 Norms (vector magnitudes)
l2_norms = np.sqrt(np.sum(embeddings**2, axis=1)).reshape(-1, 1)
normalized_embeddings_manual = embeddings / l2_norms

# TORCH NORMALIZATION (production ready)
normalized_embeddings_torch = torch.nn.functional.normalize(
    torch.from_numpy(embeddings)
).numpy()

print("\n✅ Normalization matches:", np.allclose(normalized_embeddings_manual, normalized_embeddings_torch))

# Cosine similarity = dot product of normalized vectors
cosine_similarity_manual = normalized_embeddings_manual @ normalized_embeddings_manual.T
print("\nCosine Similarity Matrix:")
print(cosine_similarity_manual.round(3))

## 4. ADDITIONAL METRICS
print("\n" + "="*60)
print("BONUS METRICS - COMPLETE COVERAGE")
print("="*60)

# MANHATTAN DISTANCE (L1)
manhattan_dist = scipy.spatial.distance.cdist(embeddings, embeddings, 'cityblock')
print("Manhattan Distance:\n", manhattan_dist.round(3))

# CORRELATION DISTANCE
corr_dist = scipy.spatial.distance.cdist(embeddings, embeddings, 'correlation')
print("Correlation Distance:\n", corr_dist.round(3))

## 5. PRODUCTION RETRIEVAL PIPELINE
print("\n" + "="*60)
print("COMPLETE SEARCH PIPELINE - PRODUCTION READY")
print("="*60)

def search_documents(query, documents, model, embeddings=None, top_k=2):
    """
    COMPLETE similarity search pipeline
    INPUT: raw query text
    OUTPUT: top_k most similar documents with scores
    
    PARAMS:
    - query: search query string
    - documents: list of document strings  
    - model: trained SentenceTransformer
    - embeddings: pre-computed doc embeddings (optional, faster)
    - top_k: return top K results
    """
    # 1. Embed query if embeddings not provided
    if embeddings is None:
        doc_embeddings = model.encode(documents)
    else:
        doc_embeddings = embeddings
    
    # 2. Normalize everything
    normalized_docs = torch.nn.functional.normalize(
        torch.from_numpy(doc_embeddings)
    ).numpy()
    normalized_query = torch.nn.functional.normalize(
        torch.from_numpy(model.encode([query]))
    ).numpy()
    
    # 3. Compute similarities (dot product = cosine similarity for normalized vectors)
    similarities = normalized_docs @ normalized_query.T
    
    # 4. Get top_k results
    top_indices = np.argsort(similarities.flatten())[::-1][:top_k]
    
    results = []
    for rank, idx in enumerate(top_indices, 1):
        results.append({
            'document': documents[idx],
            'similarity': similarities[idx, 0],
            'rank': rank
        })
    
    return results

# TEST PIPELINE
query = "Is the weather nice today?"
results = search_documents(query, documents, model, normalized_embeddings_torch)

print(f"\nQUERY: '{query}'")
print("-" * 50)
for result in results:
    print(f"Rank {result['rank']}: {result['document']}")
    print(f"  Similarity: {result['similarity']:.3f}\n")
