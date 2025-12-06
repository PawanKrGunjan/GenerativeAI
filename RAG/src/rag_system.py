# src/rag_system.py
from .document_processor import DocumentProcessor
from .embedding_manager import EmbeddingManager
from .generator import Generator
from pathlib import Path
from typing import List, Dict
import logging

logger = logging.getLogger('RAG')

class RAGSystem:
    """Complete RAG system orchestrating all components"""
    
    def __init__(
        self,
        model_path: str,
        index_dir: str = "./data/processed",
        embedding_model: str = "all-MiniLM-L6-v2",
        chunk_size: int = 500,
        chunk_overlap: int = 100
    ):
        """
        Initialize complete RAG system
        
        Args:
            model_path: Path to LLM model file
            index_dir: Directory for saving/loading index
            embedding_model: Sentence transformer model name
            chunk_size: Words per chunk
            chunk_overlap: Overlapping words between chunks
        """
        self.index_dir = index_dir
        
        # Initialize components
        self.processor = DocumentProcessor(chunk_size, chunk_overlap)
        self.embedding_manager = EmbeddingManager(embedding_model)
        self.generator = Generator(model_path)
        
        # Try to load existing index
        if Path(index_dir).exists() and \
           (Path(index_dir) / "faiss_index.bin").exists():
            logger.info(f"Loading existing index from {index_dir}")
            self.embedding_manager.load(index_dir)
    
    def index_documents(self, documents_dir: str, save: bool = True):
        """
        Process and index all documents in directory
        
        Args:
            documents_dir: Directory containing documents
            save: Whether to save index to disk
        """
        logger.info(f"\n=== Indexing Documents from {documents_dir} ===")
        
        # Process documents
        chunks = self.processor.process_directory(documents_dir)
        
        if not chunks:
            logger.info("No chunks created. Check document directory.")
            return
        
        # Add to vector database
        self.embedding_manager.add_chunks(chunks)
        
        # Save index
        if save:
            self.embedding_manager.save(self.index_dir)
    
    def query(
        self,
        question: str,
        k: int = 3,
        max_tokens: int = 512,
        temperature: float = 0.7,
        show_context: bool = False
    ) -> Dict:
        """
        Query the RAG system
        
        Args:
            question: User's question
            k: Number of chunks to retrieve
            max_tokens: Maximum tokens for answer
            temperature: Generation temperature
            show_context: Whether to logger.info retrieved context
            
        Returns:
            Dictionary with answer and metadata
        """
        logger.info(f"\n=== Processing Query ===")
        logger.info(f"Question: {question}")
        
        # Retrieve relevant chunks
        logger.info(f"Retrieving top {k} relevant chunks...")
        results = self.embedding_manager.search(question, k)
        
        if not results:
            return {
                'query': question,
                'answer': "No relevant information found in the knowledge base.",
                'sources': [],
                'num_chunks': 0
            }
        
        # Extract chunks and scores
        chunks = [chunk for chunk, score in results]
        scores = [score for chunk, score in results]
        
        logger.info(f"Retrieved {len(chunks)} chunks (avg similarity: {sum(scores)/len(scores):.3f})")
        
        # Show retrieved context if requested
        if show_context:
            logger.info("\n=== Retrieved Context ===")
            for i, (chunk, score) in enumerate(zip(chunks, scores), 1):
                logger.info(f"\n[Chunk {i}] (score: {score:.3f})")
                logger.info(f"Source: {chunk.get('source', 'Unknown')}")
                logger.info(f"Text preview: {chunk['text'][:200]}...")
        
        # Generate answer
        logger.info("\n=== Generating Answer ===")
        result = self.generator.generate(
            question,
            chunks,
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        # Add retrieval scores to result
        result['retrieval_scores'] = scores
        
        return result


if __name__ == "__main__":
    rag = RAGSystem(
        model_path="./models/llama-2-7b-chat.Q4_K_M.gguf",  # or your local GGUF path
        index_dir="data/processed",
        embedding_model="all-MiniLM-L6-v2"
    )

    # Index once (uncomment first time)
    # rag.index_documents("data/documents")

    print("RAG System Ready! Ask questions (type 'quit' to exit)\n")
    while True:
        try:
            query = input("You: ").strip()
            if query.lower() in ["quit", "exit", "bye"]:
                break
            if not query:
                continue

            result = rag.query(query, k=4, show_context=True)
            print(f"\nAnswer: {result['answer']}\n")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break