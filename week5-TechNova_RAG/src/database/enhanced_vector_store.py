# src/database/enhanced_vector_store.py

"""
Enhanced vector database functionality with improved retrieval options.
"""

import chromadb
import numpy as np
import json
import os
from typing import List, Dict, Any, Optional, Tuple

from src.config.retrieval_config import DEFAULT_RETRIEVAL_CONFIG

class EnhancedVectorStore:
    """
    Enhanced vector store with advanced retrieval capabilities.
    Built on top of ChromaDB with additional filtering and re-ranking.
    """
    
    def __init__(self, db_path: str, collection_name: str = None, config: Dict[str, Any] = None):
        """
        Initialize the enhanced vector store.
        
        Args:
            db_path: Path to the ChromaDB directory
            collection_name: Name of the collection to use (will list all collections if not found)
            config: Configuration dictionary (defaults to DEFAULT_RETRIEVAL_CONFIG)
        """
        self.config = config or DEFAULT_RETRIEVAL_CONFIG
        self.client = chromadb.PersistentClient(path=db_path)
        
        # Get available collections
        all_collections = self.client.list_collections()
        collection_names = [c.name for c in all_collections]
        
        # Print available collections for debugging
        print(f"Available collections: {collection_names}")
        
        # Use provided collection name or the first one available
        if collection_name and collection_name in collection_names:
            self.collection_name = collection_name
        elif collection_names:
            self.collection_name = collection_names[0]
            print(f"Collection '{collection_name if collection_name else 'technova_docs'}' not found. Using '{self.collection_name}' instead.")
        else:
            raise ValueError(f"No collections found in the database at {db_path}")
            
        self.collection = self.client.get_collection(self.collection_name)
        print(f"Initialized vector store with {self.collection.count()} documents in collection '{self.collection_name}'")
        
    def query(
        self, 
        query_text: str, 
        num_results: int = None,
        filters: Dict[str, Any] = None,
        threshold: float = None
    ) -> List[Dict[str, Any]]:
        """
        Query the vector store with enhanced options.
        
        Args:
            query_text: The text query to search for
            num_results: Number of results to return (overrides config)
            filters: Metadata filters to apply (overrides config)
            threshold: Minimum similarity score (overrides config)
            
        Returns:
            List of document dictionaries with content, metadata, and scores
        """
        # Set parameters, prioritizing explicit args over config values
        n_results = num_results or self.config["num_chunks"]
        sim_threshold = threshold or self.config["similarity_threshold"]
        
        # Handle reranking (fetch more results if reranking is enabled)
        fetch_count = n_results
        if self.config["enable_reranking"]:
            fetch_count = int(n_results * self.config["reranking_fetch_multiplier"])
        
        # Prepare filters
        where_filter = None
        if self.config["enable_metadata_filtering"]:
            # Start with default filters from config
            combined_filters = dict(self.config["default_filters"])
            # Add/override with explicit filters
            if filters:
                combined_filters.update(filters)
            # Only set where_filter if we have actual filters
            if combined_filters:
                where_filter = combined_filters
        
        # Execute query
        results = self.collection.query(
            query_texts=[query_text],
            n_results=fetch_count,
            where=where_filter
        )
        
        # Extract and format results
        formatted_results = []
        for i in range(len(results["ids"][0])):
            similarity_score = results["distances"][0][i] if "distances" in results else 0
            # Skip results below threshold
            if similarity_score < sim_threshold:
                continue
                
            doc_id = results["ids"][0][i]
            metadata = json.loads(results["metadatas"][0][i])
            document = {
                "id": doc_id,
                "content": results["documents"][0][i],
                "metadata": metadata,
                "score": similarity_score
            }
            formatted_results.append(document)
        
        # Apply re-ranking if enabled
        if self.config["enable_reranking"] and len(formatted_results) > n_results:
            formatted_results = self._rerank_results(formatted_results, query_text)
        
        # Truncate to requested number of results
        return formatted_results[:n_results]
    
    def _rerank_results(
        self, 
        results: List[Dict[str, Any]], 
        query_text: str
    ) -> List[Dict[str, Any]]:
        """
        Re-rank results using Maximum Marginal Relevance to balance relevance and diversity.
        
        Args:
            results: Initial ranked results
            query_text: Original query text
            
        Returns:
            Re-ranked list of results
        """
        if not results:
            return results
            
        # If MMR lambda is 1, just return the sorted results by score
        if self.config["mmr_lambda"] == 1.0:
            return sorted(results, key=lambda x: x["score"], reverse=True)
            
        # Implementation of Maximum Marginal Relevance
        # Select most relevant document first
        selected_indices = [0]  # Start with highest scoring document
        remaining_indices = list(range(1, len(results)))
        
        mmr_lambda = self.config["mmr_lambda"]
        
        while len(selected_indices) < min(self.config["num_chunks"], len(results)):
            # Find document with highest MMR
            max_mmr = -float('inf')
            max_idx = -1
            
            for i in remaining_indices:
                # Relevance component
                relevance = results[i]["score"]
                
                # Diversity component (minimize similarity to already selected docs)
                max_similarity = 0
                for j in selected_indices:
                    # Calculate cosine similarity between documents (simplified)
                    # In a real implementation, you would use actual embeddings
                    similarity = self._calculate_similarity(results[i], results[j])
                    max_similarity = max(max_similarity, similarity)
                
                # MMR score
                mmr_score = mmr_lambda * relevance - (1 - mmr_lambda) * max_similarity
                
                if mmr_score > max_mmr:
                    max_mmr = mmr_score
                    max_idx = i
            
            if max_idx == -1:
                break
                
            selected_indices.append(max_idx)
            remaining_indices.remove(max_idx)
        
        # Return re-ranked results
        return [results[i] for i in selected_indices]
    
    def _calculate_similarity(self, doc1: Dict[str, Any], doc2: Dict[str, Any]) -> float:
        """
        Calculate similarity between two documents.
        This is a simplified implementation - in a real system you would use embeddings.
        
        Args:
            doc1: First document
            doc2: Second document
            
        Returns:
            Similarity score between 0 and 1
        """
        # Simple heuristic: if they're from the same page, they're more similar
        if (doc1["metadata"].get("page_title") == doc2["metadata"].get("page_title")):
            return 0.8
        # If they share header paths, they're somewhat similar
        elif (doc1["metadata"].get("header_path") and 
              doc2["metadata"].get("header_path") and
              any(h in doc2["metadata"]["header_path"] for h in doc1["metadata"]["header_path"])):
            return 0.5
        # Otherwise assume low similarity
        return 0.1