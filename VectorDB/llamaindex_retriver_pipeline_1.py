import logging
from typing import List, Dict, Optional, Tuple
from llama_index.core import (
    Response,
    Document,
    Settings,
    VectorStoreIndex,
    DocumentSummaryIndex,
    KeywordTableIndex,
    get_response_synthesizer,
    QueryBundle,
)

from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.schema import NodeWithScore
from llama_index.core.evaluation import SemanticSimilarityEvaluator
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# ─── Logging setup ───────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ProductionRAG")

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



# ─── Production-grade RAG Pipeline ───────────────────────────────────────────────
class ProductionRAGPipeline:
    """
    Production-ready RAG pipeline with:
    - Multiple retrieval strategies
    - Query routing (auto / vector / keyword / hybrid)
    - Response synthesis
    - Basic evaluation with semantic similarity
    """

    def __init__(
        self,
        vector_index: VectorStoreIndex,
        nodes: List[Document],  # for BM25
        llm: Ollama,
        top_k: int = 6,
        hybrid_vector_weight: float = 0.65,
    ):
        self.vector_index = vector_index
        self.nodes = nodes
        self.llm = llm
        self.top_k = top_k
        self.hybrid_vector_weight = hybrid_vector_weight

        # Retrievers
        self.vector_retriever = VectorIndexRetriever(
            index=vector_index,
            similarity_top_k=top_k * 2  # more candidates → better fusion
        )

        try:
            self.bm25_retriever = BM25Retriever.from_defaults(
                nodes=nodes,
                similarity_top_k=top_k * 2
            )
        except Exception as e:
            logger.warning(f"BM25Retriever failed: {e}. Falling back to vector only.")
            self.bm25_retriever = self.vector_retriever

        # Response synthesizer (tree summarize style is usually good balance)
        self.response_synthesizer = get_response_synthesizer(
            llm=llm,
            response_mode="tree_summarize",
            verbose=False
        )

        # Evaluator for semantic similarity
        self.evaluator = SemanticSimilarityEvaluator(
            embed_model=Settings.embed_model
        )

        logger.info("Production RAG Pipeline initialized successfully")

    def _retrieve_hybrid(self, query: str) -> List[NodeWithScore]:
        """Internal hybrid retrieval (vector + BM25 fusion)"""
        vec_results = self.vector_retriever.retrieve(query)
        bm25_results = self.bm25_retriever.retrieve(query)

        # Min-max normalization per retriever
        def normalize(results):
            if not results:
                return {}
            scores = [r.score for r in results if r.score is not None]
            if not scores:
                return {}
            min_s, max_s = min(scores), max(scores)
            if max_s == min_s:
                return {r.node.node_id: 1.0 for r in results}
            return {
                r.node.node_id: (r.score - min_s) / (max_s - min_s)
                for r in results if r.score is not None
            }

        vec_norm = normalize(vec_results)
        bm25_norm = normalize(bm25_results)

        # Simple weighted fusion (using node_id for deduplication)
        fused_scores: Dict[str, Tuple[float, NodeWithScore]] = {}

        for r in vec_results:
            nid = r.node.node_id
            score = vec_norm.get(nid, 0) * self.hybrid_vector_weight
            if nid not in fused_scores or score > fused_scores[nid][0]:
                fused_scores[nid] = (score, r)

        for r in bm25_results:
            nid = r.node.node_id
            score = bm25_norm.get(nid, 0) * (1 - self.hybrid_vector_weight)
            if nid not in fused_scores or score > fused_scores[nid][0]:
                fused_scores[nid] = (score, r)

        # Sort & return
        ranked = [ns for _, ns in fused_scores.values()]
        ranked.sort(key=lambda x: x.score if x.score else 0, reverse=True)
        return ranked[:self.top_k]

    def query(self, question: str, strategy: str = "auto") -> Response:
        """
        Main query method with strategy routing
        
        Strategies:
            - "auto"      → hybrid (recommended default)
            - "vector"    → pure semantic
            - "keyword"   → pure BM25
            - "hybrid"    → explicit hybrid fusion
        """
        try:
            if strategy == "auto" or strategy == "hybrid":
                nodes = self._retrieve_hybrid(question)
            elif strategy == "vector":
                nodes = self.vector_retriever.retrieve(question)[:self.top_k]
            elif strategy == "keyword":
                nodes = self.bm25_retriever.retrieve(question)[:self.top_k]
            else:
                logger.warning(f"Unknown strategy '{strategy}', falling back to hybrid")
                nodes = self._retrieve_hybrid(question)

            if not nodes:
                return Response(response="No relevant information found.", source_nodes=[])

            response = self.response_synthesizer.synthesize(
                question,
                nodes=nodes
            )

            logger.info(f"Query processed successfully. Strategy: {strategy}")
            return response

        except Exception as e:
            logger.error(f"Error during query: {str(e)}")
            return Response(
                response="Sorry, an error occurred while processing your question.",
                source_nodes=[]
            )

    def evaluate(
        self,
        test_queries: List[str],
        expected_answers: List[str],
        similarity_threshold: float = 0.72
    ) -> Dict:
        """
        Fixed evaluation using correct Response object
        """
        if len(test_queries) != len(expected_answers):
            raise ValueError("Queries and expected answers must match in length")

        results = []
        total = len(test_queries)

        for q, expected in zip(test_queries, expected_answers):
            # Important: get the FULL Response object, not just .response
            full_response = self.query(q, strategy="auto")
            generated_text = full_response.response

            try:
                eval_result = self.evaluator.evaluate_response(
                    query=QueryBundle(query_str=q),          # ← wrap query
                    response=full_response,                  # ← pass full Response object!
                    reference=expected
                )
                similarity = eval_result.score
            except Exception as e:
                logger.warning(f"Evaluation failed for '{q}': {e}")
                similarity = 0.0

            passed = similarity >= similarity_threshold
            results.append({
                "query": q,
                "similarity": round(similarity, 3),
                "passed": passed,
                "generated": generated_text[:80] + "..."   # for debugging
            })

        passed_count = sum(1 for r in results if r["passed"])
        pass_rate = (passed_count / total) * 100 if total > 0 else 0

        summary = {
            "total_queries": total,
            "passed": passed_count,
            "pass_rate_%": round(pass_rate, 1),
            "threshold_used": similarity_threshold,
            "detailed_results": results
        }

        logger.info(f"Evaluation complete. Pass rate: {pass_rate:.1f}%")
        return summary

# ─── Example Usage & Testing ─────────────────────────────────────────────────────
if __name__ == "__main__":
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
    # Assuming you already have lab from previous code
    # Prepare documents
    docs = [Document(text=txt) for txt in SAMPLE_DOCUMENTS]

    # Build lab (indexes)
    lab = AdvancedRetrieversLab(documents=docs)

    # For standalone test - minimal setup
    docs = [Document(text=t) for t in SAMPLE_DOCUMENTS]
    vector_index = VectorStoreIndex.from_documents(docs)
    nodes = SentenceSplitter(chunk_size=512).get_nodes_from_documents(docs)

    pipeline = ProductionRAGPipeline(
        vector_index=vector_index,
        nodes=nodes,
        llm=Settings.llm,
        top_k=6
    )

    # Quick smoke test
    test_q = "What are the main types of machine learning?"
    print("\nSmoke test query:", test_q)
    print("-" * 60)
    response = pipeline.query(test_q, strategy="auto")
    print("Answer:", response.response)
    print("\nTop sources:")
    for i, n in enumerate(response.source_nodes[:3], 1):
        print(f"{i}. {n.node.get_content()[:140]}...")

    # Evaluation example
    print("\n" + "="*70)
    print("BASIC EVALUATION EXAMPLE")
    print("="*70)

    eval_queries = [
        "What is machine learning?",
        "How does deep learning work?",
        "What is reinforcement learning?"
    ]

    eval_expected = [
        "Machine learning is a branch of AI focused on algorithms that learn from data.",
        "Deep learning uses multi-layered neural networks to model complex patterns.",
        "Reinforcement learning involves agents learning via rewards and penalties."
    ]

    eval_result = pipeline.evaluate(eval_queries, eval_expected, similarity_threshold=0.70)

    print(f"Pass rate: {eval_result['pass_rate_%']}% ({eval_result['passed']}/{eval_result['total_queries']})")
    for r in eval_result["detailed_results"]:
        status = "PASS" if r["passed"] else "FAIL"
        print(f"  {status}  |  {r['similarity']:.3f}  |  {r['query'][:60]}...")