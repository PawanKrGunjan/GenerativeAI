"""
Production-Grade RAG Pipeline – Optimized Version (January 2026)
- Memory-safe (uses tiny LLM for summaries)
- Clean structure, type hints, logging
- Flexible configuration
- Robust error handling
"""

import logging
from typing import List, Dict, Optional, Tuple
from pathlib import Path

from llama_index.core import (
    Document,
    Settings,
    VectorStoreIndex,
    DocumentSummaryIndex,
    KeywordTableIndex,
    get_response_synthesizer,
    QueryBundle,
    Response,
    StorageContext,
    load_index_from_storage,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.schema import NodeWithScore
from llama_index.core.evaluation import SemanticSimilarityEvaluator
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# ─── Configuration ───────────────────────────────────────────────────────────────
PERSIST_DIR = Path("./storage")
USE_TINY_LLM = True  # Set to False only if you have 16+ GB RAM

# ─── Logging ─────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("ProductionRAG")

class RAGConfig:
    EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
    TINY_LLM_MODEL = "llama3.2:1b"          # ~2–3.5 GB
    FULL_LLM_MODEL = "llama3.2:latest"      # only if 16+ GB RAM

    @staticmethod
    def get_llm(tiny: bool = True) -> Ollama:
        model = RAGConfig.TINY_LLM_MODEL if tiny else RAGConfig.FULL_LLM_MODEL
        return Ollama(model=model, temperature=0.1, max_tokens=300)

# Usage – at the top of your file:
Settings.embed_model = HuggingFaceEmbedding(RAGConfig.EMBEDDING_MODEL, device = 'cpu')
Settings.llm = RAGConfig.get_llm(tiny=True)  # always safe default


# ─── Data ────────────────────────────────────────────────────────────────────────
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

# ─── Lab / Index Builder ─────────────────────────────────────────────────────────
class AdvancedRetrieversLab:
    """Manages index creation with persistence support."""

    def __init__(self, documents: List[Document], persist_dir: Path = PERSIST_DIR):
        self.documents = documents
        self.persist_dir = persist_dir
        self.nodes = SentenceSplitter(chunk_size=512, chunk_overlap=80).get_nodes_from_documents(documents)

        self.vector_index = self._load_or_build_vector_index()
        self.keyword_index = KeywordTableIndex.from_documents(documents)

        # Summary index – use tiny LLM explicitly
        summary_llm = RAGConfig.get_llm(tiny=True)
        self.document_summary_index = DocumentSummaryIndex.from_documents(
            documents,
            llm=summary_llm,
            transformations=[SentenceSplitter(chunk_size=1024, chunk_overlap=200)],
            show_progress=True,
        )

        # Optional: persist everything
        self.persist()

    def _load_or_build_vector_index(self) -> VectorStoreIndex:
        if self.persist_dir.exists():
            try:
                storage_context = StorageContext.from_defaults(persist_dir=self.persist_dir)
                return load_index_from_storage(storage_context)
            except Exception as e:
                logger.warning(f"Failed to load persisted index: {e}. Rebuilding...")
        
        logger.info("Building new vector index...")
        return VectorStoreIndex.from_documents(self.documents)

    def persist(self):
        """Save indexes for faster startup next time."""
        self.vector_index.storage_context.persist(persist_dir=self.persist_dir)
        logger.info(f"Indexes persisted to {self.persist_dir}")

# ─── Production RAG Pipeline ─────────────────────────────────────────────────────
class ProductionRAGPipeline:
    def __init__(
        self,
        vector_index: VectorStoreIndex,
        nodes: List[NodeWithScore],
        llm: Ollama,
        top_k: int = 6,
        hybrid_weight_vector: float = 0.65,
    ):
        self.vector_index = vector_index
        self.nodes = nodes
        self.llm = llm
        self.top_k = top_k
        self.hybrid_weight_vector = hybrid_weight_vector

        self.vector_retriever = VectorIndexRetriever(index=vector_index, similarity_top_k=top_k * 2)

        try:
            self.bm25_retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=top_k * 2)
        except Exception:
            logger.warning("BM25 unavailable → using vector only")
            self.bm25_retriever = self.vector_retriever

        self.synthesizer = get_response_synthesizer(llm=llm, response_mode="tree_summarize")

        self.evaluator = SemanticSimilarityEvaluator(embed_model=Settings.embed_model)

        logger.info("RAG Pipeline ready")

    def _hybrid_retrieve(self, query_str: str) -> List[NodeWithScore]:
        vec_res = self.vector_retriever.retrieve(query_str)
        bm25_res = self.bm25_retriever.retrieve(query_str)

        def normalize(res_list):
            if not res_list:
                return {}
            scores = [r.score for r in res_list if r.score]
            if not scores:
                return {}
            min_s, max_s = min(scores), max(scores)
            if max_s == min_s:
                return {r.node.node_id: 1.0 for r in res_list}
            return {r.node.node_id: (r.score - min_s) / (max_s - min_s) for r in res_list if r.score}

        v_norm = normalize(vec_res)
        b_norm = normalize(bm25_res)

        fused: Dict[str, Tuple[float, NodeWithScore]] = {}

        for r in vec_res:
            nid = r.node.node_id
            score = v_norm.get(nid, 0) * self.hybrid_weight_vector
            if nid not in fused or score > fused[nid][0]:
                fused[nid] = (score, r)

        for r in bm25_res:
            nid = r.node.node_id
            score = b_norm.get(nid, 0) * (1 - self.hybrid_weight_vector)
            if nid not in fused or score > fused[nid][0]:
                fused[nid] = (score, r)

        ranked = [ns for _, ns in fused.values()]
        ranked.sort(key=lambda x: x.score or 0, reverse=True)
        return ranked[:self.top_k]

    def query(self, question: str, strategy: str = "auto") -> Response:
        try:
            if strategy in ("auto", "hybrid"):
                nodes = self._hybrid_retrieve(question)
            elif strategy == "vector":
                nodes = self.vector_retriever.retrieve(question)[:self.top_k]
            elif strategy == "keyword":
                nodes = self.bm25_retriever.retrieve(question)[:self.top_k]
            else:
                logger.warning(f"Unknown strategy: {strategy} → using hybrid")
                nodes = self._hybrid_retrieve(question)

            if not nodes:
                return Response(response="No relevant information found.", source_nodes=[])

            return self.synthesizer.synthesize(question, nodes=nodes)

        except Exception as e:
            logger.error(f"Query failed: {e}")
            return Response(response="Internal error occurred.", source_nodes=[])

    def evaluate(
        self,
        queries: List[str],
        references: List[str],
        threshold: float = 0.78
    ) -> Dict:
        if len(queries) != len(references):
            raise ValueError("Length mismatch between queries and references")

        results = []
        for q, ref in zip(queries, references):
            resp = self.query(q)
            try:
                eval_res = self.evaluator.evaluate_response(
                    query=QueryBundle(q),
                    response=resp,
                    reference=ref
                )
                score = eval_res.score
            except Exception as e:
                logger.warning(f"Eval failed for '{q[:50]}...': {e}")
                score = 0.0

            results.append({
                "query": q,
                "score": round(score, 3),
                "passed": score >= threshold
            })

        passed = sum(1 for r in results if r["passed"])
        return {
            "total": len(queries),
            "passed": passed,
            "pass_rate": round((passed / len(queries)) * 100, 1) if queries else 0,
            "threshold": threshold,
            "details": results
        }


# ─── Main ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    docs = [Document(text=t) for t in SAMPLE_DOCUMENTS]
    lab = AdvancedRetrieversLab(docs)

    pipeline = ProductionRAGPipeline(
        vector_index=lab.vector_index,
        nodes=lab.nodes,
        llm=Settings.llm,
        top_k=6
    )

    # Smoke test
    q = "What are the main types of machine learning?"
    print(f"\nQuery: {q}")
    print("-" * 60)
    r = pipeline.query(q)
    print("Answer:", r.response.strip())
    print("\nSources:")
    for i, n in enumerate(r.source_nodes[:3], 1):
        print(f"  {i}. {n.node.get_content()[:120]}...")

    # Evaluation
    eval_queries = [
        "What is machine learning?",
        "How does deep learning work?",
        "What is reinforcement learning?"
    ]
    eval_refs = [
        "Machine learning is a subset of AI focused on learning from data.",
        "Deep learning uses deep neural networks with multiple layers.",
        "Reinforcement learning learns through trial, reward and punishment."
    ]

    result = pipeline.evaluate(eval_queries, eval_refs, threshold=0.78)
    print(f"\nEvaluation: {result['pass_rate']}% pass rate ({result['passed']}/{result['total']})")