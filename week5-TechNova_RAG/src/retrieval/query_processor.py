# src/retrieval/query_processor.py - UPDATED

import os
import openai
from typing import List, Dict, Any, Optional
import sys

# Add the project root to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.database.vector_store import VectorStore

class QueryProcessor:
    """
    Processes user queries for the RAG system by generating embeddings
    and retrieving relevant chunks from the vector database.
    """
    
    def __init__(self, vector_store: VectorStore, embedding_model: str = "text-embedding-3-small"):
        """
        Initialize the QueryProcessor.
        
        Args:
            vector_store: The vector database interface
            embedding_model: The OpenAI embedding model to use
        """
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        # Load API key from environment or .env file
        openai.api_key = os.getenv("OPENAI_API_KEY")
        if not openai.api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate an embedding vector for the given text.
        
        Args:
            text: The text to embed
            
        Returns:
            A list of floats representing the embedding vector
        """
        try:
            response = openai.embeddings.create(
                model=self.embedding_model,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error generating embedding: {e}")
            raise
    
    def retrieve_relevant_chunks(
        self, 
        query: str, 
        k: int = 5, 
        min_similarity: float = 0.3,  # Adjusted default to account for potential negative scores
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[Any, Any]]:
        """
        Retrieve the most relevant chunks for a given query.
        
        Args:
            query: The user's question
            k: Number of chunks to retrieve
            min_similarity: Minimum similarity threshold (adjusted for potential negative scores)
            filter_metadata: Optional metadata filters
            
        Returns:
            List of relevant chunks with metadata and similarity scores
        """
        try:
            # Generate embedding for the query
            query_embedding = self.generate_embedding(query)
            
            # Retrieve similar chunks from the vector store using the query method
            results = self.vector_store.query(
                query_embedding=query_embedding,
                n_results=k,
                filter_metadata=filter_metadata
            )
            
            # Filter by minimum similarity if specified
            if min_similarity > 0:
                # Only include results with similarity above the threshold
                filtered_results = [r for r in results if r.get('similarity', 0) >= min_similarity]
                # But if this results in no chunks, return at least the best match
                if not filtered_results and results:
                    filtered_results = [max(results, key=lambda x: x.get('similarity', 0))]
                return filtered_results
            
            return results
        except Exception as e:
            print(f"Error retrieving chunks: {e}")
            raise