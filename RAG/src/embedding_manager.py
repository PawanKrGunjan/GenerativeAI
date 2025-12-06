# src/embedding_manager.py
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple
import pickle
from pathlib import Path
import logging

logger = logging.getLogger('RAG')

class EmbeddingManager:
    """Manage embeddings and vector database"""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize embedding manager
        
        Args:
            model_name: Name of sentence-transformers model
        """
        logger.info(f"Loading embedding model: {model_name}")
        #self.model = SentenceTransformer(model_name)
        self.model = SentenceTransformer(model_name, device="cpu")

        self.dimension = self.model.get_sentence_embedding_dimension()
        
        # Initialize FAISS index
        self.index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine similarity
        self.chunks = []  # Store chunk metadata
        
        logger.info(f"Embedding dimension: {self.dimension}")
    
    def embed_texts(self, texts: List[str], show_progress: bool = True) -> np.ndarray:
        """
        Generate embeddings for list of texts
        
        Args:
            texts: List of text strings to embed
            show_progress: Whether to show progress bar
            
        Returns:
            Numpy array of embeddings
        """
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=show_progress,
            normalize_embeddings=True  # L2 normalize for cosine similarity
        )
        return embeddings.astype('float32')
    
    def add_chunks(self, chunks: List[Dict]):
        """
        Add chunks to the vector database
        
        Args:
            chunks: List of chunk dictionaries with 'text' field
        """
        if not chunks:
            logger.info("No chunks to add")
            return
        
        logger.info(f"Generating embeddings for {len(chunks)} chunks...")
        
        # Extract text from chunks
        texts = [chunk['text'] for chunk in chunks]
        
        # Generate embeddings
        embeddings = self.embed_texts(texts)
        
        # Add to FAISS index
        self.index.add(embeddings)
        
        # Store chunk metadata
        self.chunks.extend(chunks)
        
        logger.info(f"Added {len(chunks)} chunks to index (total: {len(self.chunks)})")
    
    def search(self, query: str, k: int = 5) -> List[Tuple[Dict, float]]:
        """
        Search for similar chunks
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of (chunk, score) tuples
        """
        if len(self.chunks) == 0:
            logger.info("Warning: No chunks in database")
            return []
        
        # Embed query
        query_embedding = self.embed_texts([query], show_progress=False)
        
        # Search FAISS index
        scores, indices = self.index.search(query_embedding, k)
        
        # Prepare results
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < len(self.chunks):
                results.append((self.chunks[idx], float(score)))
        
        return results
    
    def save(self, directory: str):
        """
        Save index and chunks to disk
        
        Args:
            directory: Directory to save files
        """
        Path(directory).mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        index_path = Path(directory) / "faiss_index.bin"
        faiss.write_index(self.index, str(index_path))
        
        # Save chunks
        chunks_path = Path(directory) / "chunks.pkl"
        with open(chunks_path, 'wb') as f:
            pickle.dump(self.chunks, f)
        
        logger.info(f"Saved index and chunks to {directory}")
    
    def load(self, directory: str):
        """
        Load index and chunks from disk
        
        Args:
            directory: Directory containing saved files
        """
        # Load FAISS index
        index_path = Path(directory) / "faiss_index.bin"
        self.index = faiss.read_index(str(index_path))
        
        # Load chunks
        chunks_path = Path(directory) / "chunks.pkl"
        with open(chunks_path, 'rb') as f:
            self.chunks = pickle.load(f)
        
        logger.info(f"Loaded {len(self.chunks)} chunks from {directory}")
