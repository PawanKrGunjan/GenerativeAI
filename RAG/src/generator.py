# src/generator.py
from llama_cpp import Llama
from typing import List, Dict
import logging

logger = logging.getLogger('RAG')

class Generator:
    """Handle answer generation with local LLM"""
    
    def __init__(self, model_path: str, n_ctx: int = 4096, n_threads: int = 8):
        """
        Initialize generator
        
        Args:
            model_path: Path to GGUF model file
            n_ctx: Context window size
            n_threads: CPU threads to use
        """
        logger.info(f"Loading LLM from {model_path}")
        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            verbose=False
        )
        logger.info("LLM loaded successfully")
    
    def build_prompt(self, query: str, context_chunks: List[Dict]) -> str:
        """
        Build RAG prompt with query and retrieved context
        
        Args:
            query: User's question
            context_chunks: Retrieved chunks with metadata
            
        Returns:
            Complete prompt string
        """
        # Build context section
        context_parts = []
        for i, chunk in enumerate(context_chunks, 1):
            source = chunk.get('source', 'Unknown')
            text = chunk['text']
            context_parts.append(f"[Source {i}: {source}]\n{text}")
        
        context = "\n\n".join(context_parts)
        
        # Build complete prompt
        prompt = f"""You are a helpful assistant that answers questions based on provided context.

Context:
{context}

Question: {query}

Instructions:
- Answer the question using ONLY information from the provided context
- If the answer is not in the context, say "I don't have enough information to answer that question"
- Cite sources by mentioning [Source X] when using information
- Be concise and direct

Answer:"""
        
        return prompt
    
    def generate(self, query: str, context_chunks: List[Dict], 
                max_tokens: int = 512, temperature: float = 0.7) -> Dict:
        """
        Generate answer using retrieved context
        
        Args:
            query: User's question
            context_chunks: Retrieved chunks
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Dictionary with answer and metadata
        """
        # Build prompt
        prompt = self.build_prompt(query, context_chunks)
        
        # Generate answer
        response = self.llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=["Question:", "\n\n\n"],
            echo=False
        )
        
        answer = response['choices'][0]['text'].strip()
        
        # Prepare result
        result = {
            'query': query,
            'answer': answer,
            'sources': [chunk.get('source', 'Unknown') for chunk in context_chunks],
            'num_chunks': len(context_chunks)
        }
        
        return result
