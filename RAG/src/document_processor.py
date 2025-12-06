# src/document_processor.py
import os
from typing import List, Dict
from pathlib import Path
import pypdf
from tqdm import tqdm
import logging

logger = logging.getLogger('RAG')

class DocumentProcessor:
    """Handle document loading and chunking for RAG"""
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        """
        Initialize document processor
        
        Args:
            chunk_size: Target words per chunk
            chunk_overlap: Words to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def load_pdf(self, filepath: str) -> str:
        """
        Extract text from PDF file
        
        Args:
            filepath: Path to PDF file
            
        Returns:
            Extracted text
        """
        text = ""
        with open(filepath, 'rb') as file:
            pdf_reader = pypdf.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        return text
    
    def load_text(self, filepath: str) -> str:
        """Load plain text file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    
    def load_document(self, filepath: str) -> str:
        """
        Load document based on file extension
        
        Args:
            filepath: Path to document
            
        Returns:
            Document text
        """
        ext = Path(filepath).suffix.lower()
        
        if ext == '.pdf':
            return self.load_pdf(filepath)
        elif ext in ['.txt', '.md', '.Md']:
            return self.load_text(filepath)
        else:
            raise ValueError(f"Unsupported file type: {ext}")
    
    def chunk_text(self, text: str, metadata: Dict = None) -> List[Dict]:
        """
        Split text into overlapping chunks
        
        Args:
            text: Input text to chunk
            metadata: Optional metadata to attach to each chunk
            
        Returns:
            List of chunk dictionaries with text and metadata
        """
        # Split into words
        words = text.split()
        chunks = []
        
        # Create overlapping chunks
        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk_words = words[i:i + self.chunk_size]
            
            # Skip very small chunks
            if len(chunk_words) < 50:
                continue
            
            chunk_text = ' '.join(chunk_words)
            
            chunk_data = {
                'text': chunk_text,
                'word_count': len(chunk_words),
                'char_count': len(chunk_text),
                'chunk_index': len(chunks)
            }
            
            # Add user-provided metadata
            if metadata:
                chunk_data.update(metadata)
            
            chunks.append(chunk_data)
        
        return chunks
    
    def process_directory(self, directory: str) -> List[Dict]:
        """
        Process all supported documents in a directory
        
        Args:
            directory: Path to directory containing documents
            
        Returns:
            List of all chunks from all documents
        """
        all_chunks = []
        supported_extensions = ['.pdf', '.txt', '.md']
        
        # Find all supported files
        files = []
        for ext in supported_extensions:
            files.extend(Path(directory).glob(f'**/*{ext}'))
        
        logger.info(f"Found {len(files)} documents to process")
        
        # Process each file
        for filepath in tqdm(files, desc="Processing documents"):
            try:
                # Load document
                text = self.load_document(str(filepath))
                
                # Create metadata for this document
                metadata = {
                    'source': filepath.name,
                    'filepath': str(filepath),
                    'file_type': filepath.suffix
                }
                
                # Chunk the document
                chunks = self.chunk_text(text, metadata)
                all_chunks.extend(chunks)
                
            except Exception as e:
                logger.info(f"Error processing {filepath}: {e}")
                continue
        
        logger.info(f"Created {len(all_chunks)} chunks from {len(files)} documents")
        return all_chunks
