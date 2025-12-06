import sentence_transformers
import faiss
import numpy as np
import pypdf
import llama_cpp
# main.py - Example usage
from src.rag_system import RAGSystem
import sys
from pathlib import Path
from utils.logger_config import setup_logger

logger = setup_logger('RAG')
logger.info(f"sentence-transformers: {sentence_transformers.__version__}")
logger.info(f"FAISS: {faiss.__version__}")
logger.info(f"NumPy: {np.__version__}")
logger.info(f"llama-cpp-python: {llama_cpp.__version__}")


def main():
    # Configuration
    MODEL_PATH = "./models/Phi-3.5-mini-instruct-Q4_K_M.gguf" #gemma-2-2b-it-Q4_K_M.gguf"   #
    DOCUMENTS_DIR = "./data/documents"
    INDEX_DIR = "./data/processed"
    
    # Check if model exists
    if not Path(MODEL_PATH).exists():
        logger.error(f"Error: Model not found at {MODEL_PATH}")
        logger.error("Please download a GGUF model and update MODEL_PATH")
        sys.exit(1)
    
    # Initialize RAG system
    logger.info("=== Initializing RAG System ===")
    rag = RAGSystem(
        model_path=MODEL_PATH,
        index_dir=INDEX_DIR,
        embedding_model="all-MiniLM-L6-v2",
        chunk_size=500,
        chunk_overlap=100
    )
    
    # Index documents (only if no existing index)
    if not Path(INDEX_DIR).exists() or \
       not (Path(INDEX_DIR) / "faiss_index.bin").exists():
        rag.index_documents(DOCUMENTS_DIR, save=True)
    
    # Example queries
    queries = [
        "What is an LLM?",
        "What is a diffusion model?",
        "Explain the different Stages in RAG",
        "What’s the best small LLM for CPU-only RAG under 8 GB RAM?",
        "What is chunking",
        "What is the best chunk size and overlap for PDFs?",
        "What are hallucinations in LLMs?",
        "Which embedding model should I use in 2025?",
    ]
    
    # Process queries
    for query in queries:
        result = rag.query(
            query,
            k=3,
            temperature=0.7,
            show_context=False
        )
        
        logger.info(f"\nQuestion: {result['query']}")
        logger.info(f"Answer: {result['answer']}")
        logger.info(f"Sources: {', '.join(result['sources'])}")
        logger.info("-" * 80)
    
    # Interactive mode
    logger.info("\n=== Interactive Mode ===")
    logger.info("Enter questions (or 'quit' to exit):")
    
    while True:
        question = input("\nYou: ").strip()
        
        if question.lower() in ['quit', 'exit', 'q']:
            logger.info("Goodbye!")
            break
        
        if not question:
            continue
        
        result = rag.query(question, k=3)
        logger.info(f"\nAssistant: {result['answer']}")
        if result['sources']:
            logger.info(f"(Sources: {', '.join(set(result['sources']))})")

if __name__ == "__main__":
    main()



