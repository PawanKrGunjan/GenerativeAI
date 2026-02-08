import os
import asyncio
import argparse
from typing import List, Optional
import logging

# Suppress noisy logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("ollama").setLevel(logging.WARNING)

from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
    get_response_synthesizer,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode, IndexNode
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.core.tools import QueryEngineTool
from llama_index.core.agent import ReActAgent
from llama_index.core.workflow import Context
from llama_index.core.query_engine import RetrieverQueryEngine

from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding


def build_or_load_hierarchical_index(
    data_dir: str,
    persist_dir: Optional[str] = None,
    required_exts: Optional[List[str]] = None,
    refresh: bool = False,
    parent_chunk_size: int = 1024,
    child_chunk_sizes: List[int] = [128, 256, 512],
    llm_model: str = "qwen2.5:3b",
    embed_model_name: str = "nomic-embed-text",
) -> VectorStoreIndex:
    """
    Build hierarchical index (parent + child IndexNodes) or load from disk.
    """
    if not refresh and persist_dir and os.path.isdir(persist_dir) and os.listdir(persist_dir):
        try:
            storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
            index = load_index_from_storage(storage_context)
            print(f"→ Loaded existing hierarchical index from {persist_dir}")
            return index
        except Exception as e:
            print(f"→ Failed to load index: {e}. Will build new index...")

    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    print(f"→ Reading documents from {data_dir}...")
    reader = SimpleDirectoryReader(
        input_dir=data_dir,
        recursive=True,
        required_exts=required_exts,
        filename_as_id=True,
    )
    documents = reader.load_data()

    if not documents:
        raise ValueError(f"No supported documents found in {data_dir}")

    print(f"→ Loaded {len(documents)} documents")

    # Set global LLM & embedding model
    Settings.llm = Ollama(
        model=llm_model,
        request_timeout=180.0,
        temperature=0.0,
    )
    Settings.embed_model = OllamaEmbedding(model_name=embed_model_name)

    print("Building hierarchical index...")

    # Parent level
    parent_parser = SentenceSplitter(chunk_size=parent_chunk_size, chunk_overlap=200)
    parent_nodes = parent_parser.get_nodes_from_documents(documents)
    print(f"→ {len(parent_nodes)} parent nodes (~{parent_chunk_size} tokens)")

    # Build hierarchy: parents + smaller child IndexNodes pointing to parents
    all_nodes: List[TextNode] = []
    child_parsers = [SentenceSplitter(chunk_size=sz, chunk_overlap=40) for sz in child_chunk_sizes]

    for parent_node in parent_nodes:
        all_nodes.append(parent_node)

        # Create document-like object for child splitting
        parent_content = parent_node.get_content()
        parent_doc = TextNode(
            text=parent_content,
            metadata=parent_node.metadata,
            excluded_embed_metadata_keys=parent_node.excluded_embed_metadata_keys,
            excluded_llm_metadata_keys=parent_node.excluded_llm_metadata_keys,
        )

        for parser in child_parsers:
            child_nodes = parser.get_nodes_from_documents([parent_doc])
            for child in child_nodes:
                index_node = IndexNode(
                    text=child.get_content(),
                    index_id=parent_node.node_id,
                    metadata=child.metadata,
                    excluded_embed_metadata_keys=child.excluded_embed_metadata_keys,
                    excluded_llm_metadata_keys=child.excluded_llm_metadata_keys,
                )
                all_nodes.append(index_node)

    print(f"→ Total nodes: {len(all_nodes)} (incl. {len(child_chunk_sizes)} child levels)")

    # Create vector index
    index = VectorStoreIndex(
        nodes=all_nodes,
        show_progress=True,
    )

    if persist_dir:
        os.makedirs(persist_dir, exist_ok=True)
        index.storage_context.persist(persist_dir=persist_dir)
        print(f"→ Index saved to {persist_dir}")

    return index

async def chat_loop(
    agent: ReActAgent,
    ctx: Context,
    max_iterations: int = 12,
    timeout: int = 300,
) -> None:
    print("\nLocal Hierarchical RAG + ReAct Agent is ready.")
    print("   Type 'exit', 'quit', 'q' or Ctrl+C to stop.\n")
    print("You can ask things like:")
    print("  • What is the candidate's full name?")
    print("  • What is the CTC / total compensation?")
    print("  • Summarize the offer letter")
    print("  • List all allowances and benefits")
    print("  • What is the joining date and location?\n")

    while True:
        try:
            user_input = input("You: ").strip()
            if not user_input:
                continue

            if user_input.lower() in {"exit", "quit", "q", "bye"}:
                print("Goodbye!")
                break

            print("Thinking...", flush=True)

            response = await agent.run(
                user_msg=user_input,
                ctx=ctx,
                max_iterations=max_iterations,
                early_stopping_method="generate",
                timeout=timeout,
            )

            # ───────────────────────────────────────────────
            # Handle different possible response types safely
            # ───────────────────────────────────────────────
            content = ""

            if hasattr(response, "response"):
                content = response.response
            elif hasattr(response, "content"):
                content = response.content
            elif isinstance(response, str):
                content = response
            elif hasattr(response, "message") and hasattr(response.message, "content"):
                content = response.message.content
            else:
                # Last resort: convert to string
                content = str(response)

            # Clean up and display
            cleaned = str(content).strip()
            if cleaned:
                print("\nBot:", cleaned)
            else:
                print("\nBot: (no clear final answer received)")
                print("  [DEBUG] Raw response type:", type(response))
                if hasattr(response, "__dict__"):
                    print("  [DEBUG] Response attributes:", vars(response))

            print()

        except asyncio.TimeoutError:
            print("\nBot: Request timed out.")
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {str(e)}")
            print("  → Make sure 'ollama serve' is running")
            print("  → Try smaller/faster model if slow: qwen2.5:3b, phi3.5:mini, llama3.2:3b")

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Local hierarchical RAG + ReAct agent with Ollama"
    )
    parser.add_argument("--data_dir", default="./data", help="Folder with documents")
    parser.add_argument("--persist_dir", default="./storage/hierarchical", help="Index storage folder")
    parser.add_argument("--top_k", type=int, default=8, help="Number of chunks to retrieve")
    parser.add_argument("--max_iterations", type=int, default=12, help="Max ReAct steps")
    parser.add_argument("--timeout", type=int, default=300, help="Timeout in seconds")
    parser.add_argument("--llm_model", default="qwen2.5:3b", help="Ollama LLM model")
    parser.add_argument("--embed_model", default="nomic-embed-text", help="Embedding model")
    parser.add_argument(
        "--file-types",
        nargs="*",
        default=None,
        help="Limit file extensions (.pdf .docx .txt .md ...)",
    )
    parser.add_argument(
        "--refresh-index",
        action="store_true",
        help="Force rebuild index even if exists",
    )
    args = parser.parse_args()

    # Prefer CPU
    os.environ["OLLAMA_NUM_GPU"] = "0"

    # Build / load hierarchical index
    try:
        index = build_or_load_hierarchical_index(
            data_dir=args.data_dir,
            persist_dir=args.persist_dir,
            required_exts=args.file_types,
            refresh=args.refresh_index,
            llm_model=args.llm_model,
            embed_model_name=args.embed_model,
        )
    except Exception as e:
        print(f"Index creation/loading failed: {e}")
        return

    # Auto-merging retriever
    vector_retriever = index.as_retriever(similarity_top_k=args.top_k)

    auto_merging_retriever = AutoMergingRetriever(
        vector_retriever,
        index.storage_context,
        verbose=False,
    )

    # Response synthesizer
    response_synthesizer = get_response_synthesizer(
        llm=Settings.llm,
        response_mode="tree_summarize",
        verbose=False,
    )

    # Query engine
    query_engine = RetrieverQueryEngine(
        retriever=auto_merging_retriever,
        response_synthesizer=response_synthesizer,
    )

    # Tool for agent
    search_tool = QueryEngineTool.from_defaults(
        query_engine=query_engine,
        name="document_search",
        description=(
            "Search and answer questions based on local documents. "
            "Use this tool for any question that requires information from the offer letter, "
            "agreement, or any other uploaded file. "
            "Always use this tool before giving an answer."
        ),
    )

    # Create ReAct agent - compatible with 0.14.x
    agent = ReActAgent(
        tools=[search_tool],
        llm=Settings.llm,
        verbose=True,           # shows reasoning steps
    )

    ctx = Context(agent)

    print("\n" + "═" * 70)
    print("  Hierarchical RAG + Auto-Merging + ReAct Agent ready")
    print("═" * 70)

    asyncio.run(chat_loop(
        agent=agent,
        ctx=ctx,
        max_iterations=args.max_iterations,
        timeout=args.timeout,
    ))
    
if __name__ == "__main__":
    main()