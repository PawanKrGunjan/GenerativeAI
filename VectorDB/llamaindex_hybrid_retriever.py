import warnings
from typing import List, Dict

warnings.filterwarnings('ignore')

# ─── LlamaIndex core imports ─────────────────────────────────────────────────────
from llama_index.core import (
    Document,
    Settings,
    VectorStoreIndex,
    DocumentSummaryIndex,
    KeywordTableIndex,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.schema import NodeWithScore

# Correct BM25 import
from llama_index.retrievers.bm25 import BM25Retriever

# Local embeddings (fast & good quality)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Ollama LLM
from llama_index.llms.ollama import Ollama

# ─── 1. Global configuration – MUST be set BEFORE any index creation ─────────────
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5",
    device="cpu",                    # change to "cuda" if you have GPU
    model_kwargs={"trust_remote_code": False},
)

# Global LLM – used only when necessary (most indexes don't need it)
def create_ollama_llm(max_tokens: int = 300) -> Ollama:
    """Create Ollama LLM instance with strict settings for reproducibility."""
    return Ollama(
        model="llama3.2:latest",
        request_timeout=120.0,
        temperature=0.0,
        max_tokens=max_tokens,
        context_window=2000,          # llama3.2 supports larger context
        format="json",                # Helps when you need structured output
    )
Settings.llm = create_ollama_llm()

print("✅ Global settings (embeddings + tiny LLM) configured successfully!")

# ─── Sample educational content ──────────────────────────────────────────────────
SAMPLE_DOCUMENTS = [
    "Artificial intelligence includes a branch called machine learning, which develops systems capable of improving performance through experience with data.",
    "Deep learning relies on multi-layered neural architectures to discover intricate patterns within large datasets.",
    "Natural language processing gives machines the ability to comprehend, analyze, and produce human language in meaningful ways.",
    "Computer vision technology enables devices to process and make sense of visual data from images and videos.",
    "In reinforcement learning, intelligent agents improve their behavior by receiving positive or negative feedback from their environment.",
    "Supervised learning trains models using input-output pairs, teaching them to predict outputs for new inputs.",
    "Unsupervised learning identifies underlying structures and relationships in data without any guidance from labeled examples.",
    "Transfer learning applies knowledge gained from one problem to accelerate progress on a different but related task.",
    "Generative artificial intelligence systems can produce original content such as written text, artwork, software code, and audio.",
    "Large-scale language models are developed by training on enormous text collections, enabling them to create remarkably human-like responses."
]

# ─── Advanced Retrievers Lab ─────────────────────────────────────────────────────
class AdvancedRetrieversLab:
    """Central class that manages different indexes for retrieval experiments."""

    def __init__(self, documents: List[Document]):
        print("🚀 Initializing Advanced Retrievers Lab...")
        self.documents = documents

        # Node splitting (balanced size for good retrieval)
        splitter = SentenceSplitter(chunk_size=512, chunk_overlap=80)
        self.nodes = splitter.get_nodes_from_documents(self.documents)

        print(f"📄 Loaded {len(self.documents)} documents")
        print(f"🔢 Created {len(self.nodes)} nodes")

        print("Building indexes...")

        # Vector Index – semantic search (fast with bge-small)
        self.vector_index = VectorStoreIndex.from_documents(self.documents)

        # Document Summary Index – uses tiny LLM to generate summaries
        self.document_summary_index = DocumentSummaryIndex.from_documents(
            self.documents,
            llm=create_ollama_llm(),
            transformations=[SentenceSplitter(chunk_size=1024, chunk_overlap=200)],
            show_progress=True,
        )

        # Keyword Index – exact term matching
        self.keyword_index = KeywordTableIndex.from_documents(self.documents)

        print("✅ All indexes successfully created!\n")


# ─── Optimized Hybrid Retriever ──────────────────────────────────────────────────
def hybrid_retrieve(
    vector_retriever: VectorIndexRetriever,
    bm25_retriever: BM25Retriever,
    query: str,
    vector_weight: float = 0.65,
    bm25_weight: float = 0.35,
    top_k: int = 5,
    candidate_k: int = 12
) -> List[NodeWithScore]:
    """
    Hybrid search: combines dense vector + sparse BM25 with normalized weighted fusion.
    Uses content string for deduplication (node ids differ between indexes).
    """
    # Fetch more candidates for better fusion
    vector_results = vector_retriever.retrieve(query)
    bm25_results = bm25_retriever.retrieve(query)

    # ─── Helper: Min-Max normalization per retriever ─────────────────────────────
    def normalize(results: List[NodeWithScore]) -> Dict[str, float]:
        if not results:
            return {}
        scores = [r.score for r in results if r.score is not None]
        if not scores:
            return {}
        min_s, max_s = min(scores), max(scores)
        if max_s == min_s:
            return {r.node.get_content(metadata_mode="none").strip(): 1.0 for r in results}
        return {
            r.node.get_content(metadata_mode="none").strip(): (r.score - min_s) / (max_s - min_s)
            for r in results if r.score is not None
        }

    vec_norm = normalize(vector_results)
    bm25_norm = normalize(bm25_results)

    # ─── Fusion: content-based best-score selection ──────────────────────────────
    fused: Dict[str, tuple[float, NodeWithScore]] = {}

    for text, score in vec_norm.items():
        hybrid = score * vector_weight
        # Find original node (first match)
        node = next((r.node for r in vector_results if r.node.get_content(metadata_mode="none").strip() == text), None)
        if node:
            fused[text] = (hybrid, NodeWithScore(node=node, score=hybrid))

    for text, score in bm25_norm.items():
        hybrid = score * bm25_weight
        node = next((r.node for r in bm25_results if r.node.get_content(metadata_mode="none").strip() == text), None)
        if node:
            if text not in fused or hybrid > fused[text][0]:
                fused[text] = (hybrid, NodeWithScore(node=node, score=hybrid))

    # Sort & return top results
    ranked = [ns for _, ns in fused.values()]
    ranked.sort(key=lambda x: x.score if x.score is not None else 0, reverse=True)
    return ranked[:top_k]


# ─── Execution ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Prepare documents
    docs = [Document(text=txt) for txt in SAMPLE_DOCUMENTS]

    # Build lab (indexes)
    lab = AdvancedRetrieversLab(documents=docs)

    # Create retrievers
    vector_retriever = VectorIndexRetriever(
        index=lab.vector_index,
        similarity_top_k=12
    )

    try:
        bm25_retriever = BM25Retriever.from_defaults(
            nodes=lab.nodes,
            similarity_top_k=12
        )
    except Exception as e:
        print(f"⚠️ BM25Retriever failed: {e}")
        bm25_retriever = vector_retriever

    # Test queries
    test_queries = [
        "What is machine learning?",
        "neural networks deep learning",
        "supervised learning techniques"
    ]

    print("\n" + "═"*80)
    print("HYBRID RETRIEVER DEMONSTRATION")
    print("═"*80)

    for query in test_queries:
        print(f"\nQuery: {query}")
        print("─"*70)

        results = hybrid_retrieve(
            vector_retriever=vector_retriever,
            bm25_retriever=bm25_retriever,
            query=query,
            vector_weight=0.65,
            bm25_weight=0.35,
            top_k=4
        )

        for i, res in enumerate(results, 1):
            preview = res.node.get_content(metadata_mode="none")[:140].replace("\n", " ").strip()
            score_str = f"{res.score:.3f}" if res.score is not None else "N/A"
            print(f"{i}. [{score_str}] {preview}...")
        print()