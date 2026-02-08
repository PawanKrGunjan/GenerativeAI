import logging
import os
from typing import List, Optional

# Suppress noisy logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("ollama").setLevel(logging.WARNING)

from llama_index.core import (
    Document,
    VectorStoreIndex,
    StorageContext,
    Settings,
    SimpleDirectoryReader,
    get_response_synthesizer,
    load_index_from_storage,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode, IndexNode
from llama_index.core.retrievers import AutoMergingRetriever, VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine

# Ollama
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding


def create_or_load_hierarchical_index(
    data_dir: str = "./data",
    persist_dir: str = "./storage/hierarchical_v2",
    parent_chunk_size: int = 1024,
    child_chunk_sizes: List[int] = [128, 256, 512],
    llm_model: str = "qwen2.5:3b",
    embed_model_name: str = "nomic-embed-text",
) -> VectorStoreIndex:
    """
    Creates a new hierarchical index or loads an existing one from disk.
    """
    print(f"Checking for existing index in {persist_dir}...")

    # ─── TRY TO LOAD EXISTING INDEX ─────────────────────────────────────
    if os.path.exists(persist_dir):
        try:
            # Correct way in 0.11+ / 0.14.x
            storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
            index = load_index_from_storage(storage_context)
            print("→ Successfully loaded existing hierarchical index")
            return index
        except Exception as e:
            print(f"→ Failed to load index: {str(e)}")
            print("→ Will create a new index...\n")

    # ─── CREATE NEW INDEX ───────────────────────────────────────────────
    print(f"→ Reading documents from {data_dir}...")
    documents = SimpleDirectoryReader(data_dir).load_data()
    print(f"→ Loaded {len(documents)} documents")

    # Configure global settings
    Settings.llm = Ollama(model=llm_model, request_timeout=180.0, temperature=0.0)
    Settings.embed_model = OllamaEmbedding(model_name=embed_model_name)

    print("Creating new hierarchical index...")

    # Parent nodes (larger chunks)
    parent_parser = SentenceSplitter(chunk_size=parent_chunk_size, chunk_overlap=200)
    parent_nodes = parent_parser.get_nodes_from_documents(documents)
    print(f"→ {len(parent_nodes)} parent nodes (size ~{parent_chunk_size})")

    # All nodes: parents + child IndexNodes that reference parents
    all_nodes: List[TextNode] = []

    child_parsers = [SentenceSplitter(chunk_size=sz, chunk_overlap=40) for sz in child_chunk_sizes]

    for parent_node in parent_nodes:
        all_nodes.append(parent_node)

        # Create a document from parent content for child splitting
        parent_doc = Document(
            text=parent_node.get_content(),
            metadata=parent_node.metadata,
            excluded_embed_metadata_keys=parent_node.excluded_embed_metadata_keys,
            excluded_llm_metadata_keys=parent_node.excluded_llm_metadata_keys,
        )

        for parser in child_parsers:
            child_nodes = parser.get_nodes_from_documents([parent_doc])
            for child_node in child_nodes:
                index_node = IndexNode(
                    text=child_node.get_content(),
                    index_id=parent_node.node_id,
                    embedding=None,
                    metadata=child_node.metadata,
                    excluded_embed_metadata_keys=child_node.excluded_embed_metadata_keys,
                    excluded_llm_metadata_keys=child_node.excluded_llm_metadata_keys,
                    relationships=child_node.relationships,
                )
                all_nodes.append(index_node)

    print(f"→ Total nodes: {len(all_nodes)} (incl. {len(child_chunk_sizes)} child levels)")

    # Build vector index
    index = VectorStoreIndex(
        nodes=all_nodes,
        show_progress=True,
    )

    # Save to disk
    os.makedirs(persist_dir, exist_ok=True)
    index.storage_context.persist(persist_dir=persist_dir)
    print(f"→ Index saved to: {persist_dir}")

    print("New hierarchical index created.")
    return index


if __name__ == "__main__":
    # ────────────────────────────────────────────────────────────────
    #   CREATE / LOAD INDEX
    # ────────────────────────────────────────────────────────────────
    index = create_or_load_hierarchical_index(
        data_dir="./data",
        persist_dir="./storage/hierarchical_v2",
        parent_chunk_size=1024,
        child_chunk_sizes=[128, 256, 512],
        llm_model="qwen2.5:3b",
        embed_model_name="nomic-embed-text",
    )

    # ────────────────────────────────────────────────────────────────
    #   SET UP RETRIEVER + AUTO-MERGING
    # ────────────────────────────────────────────────────────────────
    vector_retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=10,          # adjust as needed (8–15 good balance)
    )

    auto_merging_retriever = AutoMergingRetriever(
        vector_retriever=vector_retriever,
        storage_context=index.storage_context,
        simple_ratio_thresh=0.5,      # ← CORRECT parameter name in 0.14.x
        verbose=True,
    )

    # ────────────────────────────────────────────────────────────────
    #   CREATE QUERY ENGINE (tree summarization)
    # ────────────────────────────────────────────────────────────────
    response_synthesizer = get_response_synthesizer(
        llm=Settings.llm,
        response_mode="tree_summarize",
        verbose=True,                 # set True during initial testing
    )

    query_engine = RetrieverQueryEngine(
        retriever=auto_merging_retriever,
        response_synthesizer=response_synthesizer,
    )

    print("\n" + "═" * 70)
    print("  HIERARCHICAL + AUTO-MERGING RAG READY (llama-index 0.14.x)")
    print("═" * 70)

    # ────────────────────────────────────────────────────────────────
    #   TEST QUESTIONS
    # ────────────────────────────────────────────────────────────────
    questions = [
        "What is the candidate's full name as per the offer letter?",
        "What is the exact job title / designation offered?",
        "What is the CTC / total compensation package mentioned?",
        "What is the joining date and location?",
        "Extract ALL personal details: full name, email, phone, address, employee ID if any.",
        "List all allowances, bonuses, and benefits mentioned in the offer letter.",
        "What is the candidate ID or employee code mentioned anywhere?",
    ]

    for q in questions:
        print(f"\nQ: {q}")
        try:
            response = query_engine.query(q)
            print(f"A: {response.response.strip() or '(no information found)'}")
            print(f"   → Sources: {len(response.source_nodes)} nodes")
            # Uncomment for debugging
            # for i, node in enumerate(response.source_nodes, 1):
            #     print(f"   [{i}] {node.node.get_content()[:180]}...")
        except Exception as e:
            print(f"   → Error: {str(e)}")

    print("\nDone.")