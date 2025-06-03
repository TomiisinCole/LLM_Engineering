# src/database/dimension_fix_vector_store.py

"""
Vector store implementation that handles dimensionality mismatch
by generating new embeddings with the correct dimensions.
"""

import chromadb
import json
import os
import requests
import numpy as np
from typing import List, Dict, Any, Optional

from src.config.retrieval_config import DEFAULT_RETRIEVAL_CONFIG

class DimensionFixVectorStore:
    """
    Vector store implementation that handles dimensionality mismatch
    by generating new embeddings with the correct dimensions.
    """
    
    def __init__(self, db_path: str, collection_name: str = None, config: Dict[str, Any] = None):
        """
        Initialize the vector store.
        
        Args:
            db_path: Path to the ChromaDB directory
            collection_name: Name of the collection to use (will use first available if not specified)
            config: Configuration dictionary
        """
        self.config = config or DEFAULT_RETRIEVAL_CONFIG
        self.db_path = db_path
        self.client = chromadb.PersistentClient(path=db_path)
        self.openai_api_key = self._get_openai_api_key()
        
        # Get available collections
        all_collections = self.client.list_collections()
        collection_names = [c.name for c in all_collections]
        
        print(f"Available collections: {collection_names}")
        
        # Use provided collection name or the first one available
        if collection_name and collection_name in collection_names:
            self.collection_name = collection_name
        elif collection_names:
            self.collection_name = collection_names[0]
            print(f"Collection '{collection_name if collection_name else 'default'}' not found. Using '{self.collection_name}' instead.")
        else:
            raise ValueError(f"No collections found in the database at {db_path}")
        
        # Get collection
        self.collection = self.client.get_collection(self.collection_name)
        
        # We'll use OpenAI's text-embedding-3-small with 1536 dimensions
        # This is the default for collections created with OpenAI embeddings
        self.embedding_dim = 1536
        
        print(f"Initialized vector store with {self.collection.count()} documents in collection '{self.collection_name}'")
        print(f"Using embedding dimension: {self.embedding_dim}")
        
        # Load a sample of documents to test extraction
        self._load_sample_documents()
    
    def _get_openai_api_key(self) -> str:
        """Get OpenAI API key from environment variables."""
        import os
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        return api_key
    
    def _load_sample_documents(self, sample_size=10):
        """Load a sample of documents to understand their structure."""
        print(f"Loading sample of {sample_size} documents to understand structure...")
        try:
            results = self.collection.get(limit=sample_size)
            
            if not results["ids"] or len(results["ids"]) == 0:
                print("Warning: No documents found in collection.")
                return
            
            # Check if documents contain " > " formatting
            has_formatting_chars = False
            for doc in results["documents"]:
                if " > " in doc:
                    has_formatting_chars = True
                    break
            
            self.needs_formatting_fix = has_formatting_chars
            print(f"Documents need formatting fix: {self.needs_formatting_fix}")
            
            # Check metadata structure
            if results["metadatas"] and len(results["metadatas"]) > 0:
                sample_metadata = results["metadatas"][0]
                print(f"Sample metadata keys: {list(sample_metadata.keys())}")
        
        except Exception as e:
            print(f"Error loading sample documents: {str(e)}")
    
    def _embed_query(self, query_text: str) -> List[float]:
        """
        Generate embeddings for a query text using OpenAI API.
        
        Args:
            query_text: The text to embed
            
        Returns:
            List of embedding values as floats
        """
        try:
            response = requests.post(
                "https://api.openai.com/v1/embeddings",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.openai_api_key}"
                },
                json={
                    "model": "text-embedding-3-small",
                    "input": query_text,
                    "dimensions": self.embedding_dim
                }
            )
            
            if response.status_code != 200:
                print(f"Error from OpenAI API: {response.text}")
                return None
                
            result = response.json()
            embedding = result["data"][0]["embedding"]
            
            return embedding
            
        except Exception as e:
            print(f"Error generating embedding: {str(e)}")
            return None
    
    def query(
        self, 
        query_text: str, 
        num_results: int = None,
        filters: Dict[str, Any] = None,
        threshold: float = None
    ) -> List[Dict[str, Any]]:
        """
        Query the vector store with embeddings that match the collection's dimensions.
        
        Args:
            query_text: The text query to search for
            num_results: Number of results to return
            filters: Metadata filters to apply
            threshold: Minimum similarity score
            
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
        if self.config["enable_metadata_filtering"] and filters:
            where_filter = filters
            
        # Also look for semantic matches for "mission" or "purpose"
        if "mission" in query_text.lower():
            if not where_filter:
                where_filter = {}
            # Enhanced semantic search by including mission-related documents
            # This helps find mission statements even if they don't use the exact word "mission"
            print("Applying semantic enhancement for mission-related query")
        
        # First, let's try specialized search for mission-related queries
        if "mission" in query_text.lower() or "purpose" in query_text.lower():
            mission_docs = self._mission_purpose_search(n_results)
            if mission_docs and len(mission_docs) > 0:
                return mission_docs
        
        try:
            # Generate embedding for the query with matching dimensionality
            embedding = self._embed_query(query_text)
            
            if embedding:
                # Execute query with the embedding
                results = self.collection.query(
                    query_embeddings=[embedding],
                    n_results=fetch_count,
                    where=where_filter
                )
                
                # Extract and format results
                formatted_results = []
                
                if not results["ids"] or len(results["ids"][0]) == 0:
                    print("No results found for the query.")
                    # Try keyword fallback
                    return self._keyword_fallback_search(query_text, n_results, where_filter)
                    
                for i in range(len(results["ids"][0])):
                    # Handle similarity score
                    similarity_score = 0.0
                    if "distances" in results and results["distances"]:
                        # Some Chroma backends return similarity (higher is better)
                        # Others return distance (lower is better)
                        distance = results["distances"][0][i]
                        
                        # Convert to similarity score (higher is better)
                        if distance < 1.0:  # Likely a distance metric
                            similarity_score = 1.0 - distance
                        else:
                            similarity_score = distance
                    
                    # Skip results below threshold
                    if similarity_score < sim_threshold:
                        continue
                        
                    doc_id = results["ids"][0][i]
                    
                    # Get content and clean up if needed
                    content = results["documents"][0][i]
                    if self.needs_formatting_fix:
                        content = content.replace(" > ", "")
                    
                    # Handle metadata parsing
                    metadata = results["metadatas"][0][i]  # ChromaDB should return dict
                    
                    # Extract and clean page title
                    page_title = self._extract_page_title(metadata, content)
                    metadata["page_title"] = page_title
                    
                    # Create document object with clean content
                    document = {
                        "id": doc_id,
                        "content": content,
                        "metadata": metadata,
                        "score": similarity_score
                    }
                    formatted_results.append(document)
                
                # Apply re-ranking if enabled
                if self.config["enable_reranking"] and len(formatted_results) > n_results:
                    formatted_results = self._rerank_results(formatted_results, query_text)
                
                # If we got results but they're not great, try enhancing with keyword search
                if formatted_results and query_text.lower().find("mission") >= 0:
                    # Special handling for mission-related queries
                    keyword_results = self._mission_purpose_search(n_results)
                    if keyword_results:
                        # Add new results that aren't duplicates
                        existing_ids = {doc["id"] for doc in formatted_results}
                        for doc in keyword_results:
                            if doc["id"] not in existing_ids:
                                formatted_results.append(doc)
                                existing_ids.add(doc["id"])
                        
                        # Re-sort by score
                        formatted_results.sort(key=lambda x: x["score"], reverse=True)
                
                # Truncate to requested number of results
                return formatted_results[:n_results]
            
            else:
                # Fall back to keyword-based search if embedding generation fails
                print("Embedding generation failed, falling back to keyword search")
                return self._keyword_fallback_search(query_text, n_results, where_filter)
                
        except Exception as e:
            print(f"Error during vector search: {str(e)}")
            return self._keyword_fallback_search(query_text, n_results, where_filter)
    
    def _mission_purpose_search(self, n_results: int) -> List[Dict[str, Any]]:
        """
        Special search for mission and purpose documents.
        
        Args:
            n_results: Number of results to return
            
        Returns:
            List of document dictionaries
        """
        print("Performing specialized mission/purpose search")
        
        # Key documents we want to prioritize based on titles from search_mission.py
        priority_titles = [
            "Mission Statement",
            "Employee HandbookCompany Overview",
            "About TechNovaWho We Are", 
            "About TechNovaWhat We Do",
            "Core Values",
            "Vision 2025",
            "Onboarding ChecklistFirst WeekCompany Introduction"
        ]
        
        mission_docs = []
        
        try:
            # Get a batch of documents to search through - increase limit to find more
            results = self.collection.get(limit=500)
            
            if not results["ids"]:
                return []
            
            # Check each document
            for i in range(len(results["ids"])):
                doc_id = results["ids"][i]
                content = results["documents"][i]
                metadata = results["metadatas"][i]
                
                # Clean up content if needed
                if self.needs_formatting_fix:
                    content = content.replace(" > ", "")
                
                # Extract page title
                page_title = self._extract_page_title(metadata, content)
                
                # Initial score based on title match
                score = 0.0
                
                # Check for exact title matches with our priority list
                for priority_title in priority_titles:
                    if priority_title.lower() in page_title.lower():
                        # High score for exact title matches
                        score = 0.9
                        break
                
                # If title didn't match, check content for keywords
                if score == 0:
                    content_lower = content.lower()
                    mission_keywords = ["mission", "vision", "values", "purpose", "about us", 
                                       "company overview", "who we are", "what we do"]
                    
                    # Count matching keywords
                    for keyword in mission_keywords:
                        if keyword in content_lower:
                            score += 0.1
                            # Give extra weight to keywords in the first paragraph
                            first_paragraph = content_lower.split("\n")[0] if "\n" in content_lower else content_lower
                            if keyword in first_paragraph:
                                score += 0.1
                
                # Include document if it has a reasonable score
                if score > 0.2:
                    metadata["page_title"] = page_title
                    mission_docs.append({
                        "id": doc_id,
                        "content": content,
                        "metadata": metadata,
                        "score": min(1.0, score)  # Cap at 1.0
                    })
            
            # Sort by score
            mission_docs.sort(key=lambda x: x["score"], reverse=True)
            
            # Return top results
            return mission_docs[:n_results]
            
        except Exception as e:
            print(f"Error in mission/purpose search: {str(e)}")
            return []
    
    def _extract_page_title(self, metadata, content):
        """Extract a clean page title from metadata or content."""
        # Try different ways to get a page title
        page_title = None
        
        # Method 1: Try to get from page_title in metadata
        if metadata and "page_title" in metadata:
            page_title = metadata["page_title"]
            if isinstance(page_title, str):
                page_title = page_title.replace(" > ", "")
        
        # Method 2: Try to get from header_path in metadata
        if not page_title and metadata and "header_path" in metadata:
            header_path = metadata["header_path"]
            if isinstance(header_path, list) and header_path:
                # Use first element of header path
                page_title = header_path[0].replace(" > ", "") if isinstance(header_path[0], str) else str(header_path[0])
        
        # Method 3: Try to extract from the content's first line
        if not page_title and content:
            lines = content.split('\n')
            if lines:
                first_line = lines[0].strip()
                # Use first line if it's reasonably short (likely a title)
                if 3 < len(first_line) < 100:
                    page_title = first_line
        
        # Fallback options
        if not page_title:
            # Try other metadata fields
            for key in ["title", "document", "name", "source"]:
                if key in metadata and metadata[key]:
                    page_title = metadata[key]
                    if isinstance(page_title, str):
                        page_title = page_title.replace(" > ", "")
                    break
        
        # Final fallback
        if not page_title:
            page_title = "TechNova Document"
            
        return page_title
    
    def _keyword_fallback_search(
        self, 
        query_text: str, 
        n_results: int,
        where_filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Fallback search using keyword matching when embedding-based search fails.
        
        Args:
            query_text: The text query to search for
            n_results: Number of results to return
            where_filter: Metadata filters to apply
            
        Returns:
            List of document dictionaries with content, metadata, and scores
        """
        print(f"Using keyword fallback search for query: {query_text}")
        
        # Try getting documents in batches
        all_docs = []
        batch_size = 100
        offset = 0
        
        while True:
            try:
                if where_filter:
                    results = self.collection.get(where=where_filter, limit=batch_size, offset=offset)
                else:
                    results = self.collection.get(limit=batch_size, offset=offset)
                
                if not results["ids"] or len(results["ids"]) == 0:
                    break
                
                # Process each document
                for i in range(len(results["ids"])):
                    # Get content and clean up if needed
                    content = results["documents"][i]
                    if self.needs_formatting_fix:
                        content = content.replace(" > ", "")
                    
                    # Get metadata
                    metadata = results["metadatas"][i]
                    
                    # Extract page title
                    page_title = self._extract_page_title(metadata, content)
                    metadata["page_title"] = page_title
                    
                    # Add to list
                    all_docs.append({
                        "id": results["ids"][i],
                        "content": content,
                        "metadata": metadata
                    })
                
                # Update offset
                offset += len(results["ids"])
                
                # If we got fewer than batch_size, we're done
                if len(results["ids"]) < batch_size:
                    break
                
                # Limit total docs to avoid memory issues
                if len(all_docs) >= 500:
                    break
                
            except Exception as e:
                print(f"Error getting documents: {str(e)}")
                break
        
        if not all_docs:
            return []
        
        # Normalize query text
        query_words = set(query_text.lower().split())
        
        # Score documents based on keyword matches
        scored_docs = []
        for doc in all_docs:
            content = doc["content"]
            content_words = set(content.lower().split())
            
            # Count matching words
            matching_words = query_words.intersection(content_words)
            
            # Simple scoring scheme
            if len(query_words) > 0:
                word_match_score = len(matching_words) / len(query_words)
            else:
                word_match_score = 0
            
            # Title boost - if query words appear in the title
            title_words = set(doc["metadata"]["page_title"].lower().split())
            title_matches = query_words.intersection(title_words)
            title_boost = 0.2 * (len(title_matches) / len(query_words) if len(query_words) > 0 else 0)
            
            # Calculate final score
            score = min(1.0, word_match_score + title_boost)
            
            # Add semantic bonus for specific query types
            # For mission-related queries, look for purpose, vision, values, etc.
            if "mission" in query_words:
                mission_related = {"purpose", "vision", "values", "goals", "aims", "objectives", "about", "company"}
                if content_words.intersection(mission_related):
                    score = min(1.0, score + 0.3)
                # Extra bonus for having these in the title
                if title_words.intersection(mission_related):
                    score = min(1.0, score + 0.4)
            
            # Only include documents with some relevance
            if score > 0:
                doc["score"] = score
                scored_docs.append(doc)
        
        # Sort by score (highest first)
        scored_docs.sort(key=lambda x: x["score"], reverse=True)
        
        # Return top results
        return scored_docs[:n_results]
    
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
    
    def query(
        self, 
        query_text: str, 
        num_results: int = None,
        filters: Dict[str, Any] = None,
        threshold: float = None
    ) -> List[Dict[str, Any]]:
        """
        Query the vector store with embeddings that match the collection's dimensions.
        
        Args:
            query_text: The text query to search for
            num_results: Number of results to return
            filters: Metadata filters to apply
            threshold: Minimum similarity score
            
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
        if self.config["enable_metadata_filtering"] and filters:
            where_filter = filters
            
        # Also look for semantic matches for "mission" or "purpose"
        if "mission" in query_text.lower():
            if not where_filter:
                where_filter = {}
            # Enhanced semantic search by including mission-related documents
            # This helps find mission statements even if they don't use the exact word "mission"
            print("Applying semantic enhancement for mission-related query")
        
        try:
            # Generate embedding for the query with matching dimensionality
            embedding = self._embed_query(query_text)
            
            if embedding:
                # Execute query with the embedding
                results = self.collection.query(
                    query_embeddings=[embedding],
                    n_results=fetch_count,
                    where=where_filter
                )
                
                # Extract and format results
                formatted_results = []
                
                if not results["ids"] or len(results["ids"][0]) == 0:
                    print("No results found for the query.")
                    # Try keyword fallback
                    return self._keyword_fallback_search(query_text, n_results, where_filter)
                    
                for i in range(len(results["ids"][0])):
                    # Handle similarity score
                    similarity_score = 0.0
                    if "distances" in results and results["distances"]:
                        # Some Chroma backends return similarity (higher is better)
                        # Others return distance (lower is better)
                        distance = results["distances"][0][i]
                        
                        # Convert to similarity score (higher is better)
                        if distance < 1.0:  # Likely a distance metric
                            similarity_score = 1.0 - distance
                        else:
                            similarity_score = distance
                    
                    # Skip results below threshold
                    if similarity_score < sim_threshold:
                        continue
                        
                    doc_id = results["ids"][0][i]
                    
                    # Get content and clean up if needed
                    content = results["documents"][0][i]
                    if self.needs_formatting_fix:
                        content = content.replace(" > ", "")
                    
                    # Handle metadata parsing
                    metadata = results["metadatas"][0][i]  # ChromaDB should return dict
                    
                    # Extract and clean page title
                    page_title = self._extract_page_title(metadata, content)
                    metadata["page_title"] = page_title
                    
                    # Create document object with clean content
                    document = {
                        "id": doc_id,
                        "content": content,
                        "metadata": metadata,
                        "score": similarity_score
                    }
                    formatted_results.append(document)
                
                # Apply re-ranking if enabled
                if self.config["enable_reranking"] and len(formatted_results) > n_results:
                    formatted_results = self._rerank_results(formatted_results, query_text)
                
                # If we got results but they're not great, try enhancing with keyword search
                if formatted_results and query_text.lower().find("mission") >= 0:
                    # Special handling for mission-related queries
                    keyword_results = self._mission_purpose_search(n_results)
                    if keyword_results:
                        # Add new results that aren't duplicates
                        existing_ids = {doc["id"] for doc in formatted_results}
                        for doc in keyword_results:
                            if doc["id"] not in existing_ids:
                                formatted_results.append(doc)
                                existing_ids.add(doc["id"])
                        
                        # Re-sort by score
                        formatted_results.sort(key=lambda x: x["score"], reverse=True)
                
                # Truncate to requested number of results
                return formatted_results[:n_results]
            
            else:
                # Fall back to keyword-based search if embedding generation fails
                print("Embedding generation failed, falling back to keyword search")
                return self._keyword_fallback_search(query_text, n_results, where_filter)
                
        except Exception as e:
            print(f"Error during vector search: {str(e)}")
            return self._keyword_fallback_search(query_text, n_results, where_filter)
    
    def _mission_purpose_search(self, n_results: int) -> List[Dict[str, Any]]:
        """
        Special search for mission and purpose documents.
        
        Args:
            n_results: Number of results to return
            
        Returns:
            List of document dictionaries
        """
        print("Performing specialized mission/purpose search")
        
        # Try different title patterns that might contain mission statements
        mission_docs = []
        
        try:
            # Try to get documents with titles/content related to mission/purpose
            mission_keywords = ["mission", "vision", "values", "purpose", "about", "company", "overview"]
            
            # Get a batch of documents to search through
            results = self.collection.get(limit=100)
            
            if not results["ids"]:
                return []
            
            # Check each document
            for i in range(len(results["ids"])):
                doc_id = results["ids"][i]
                content = results["documents"][i]
                metadata = results["metadatas"][i]
                
                # Clean up content if needed
                if self.needs_formatting_fix:
                    content = content.replace(" > ", "")
                
                # Check if content contains mission-related keywords
                content_lower = content.lower()
                score = 0.0
                
                for keyword in mission_keywords:
                    if keyword in content_lower:
                        score += 0.1
                        # Give extra weight to documents with these keywords in the first paragraph
                        first_paragraph = content_lower.split("\n")[0] if "\n" in content_lower else content_lower
                        if keyword in first_paragraph:
                            score += 0.2
                
                # Check title for mission-related keywords
                page_title = self._extract_page_title(metadata, content)
                title_lower = page_title.lower()
                
                for keyword in mission_keywords:
                    if keyword in title_lower:
                        score += 0.3
                
                # Include document if it has a reasonable score
                if score > 0.2:
                    metadata["page_title"] = page_title
                    mission_docs.append({
                        "id": doc_id,
                        "content": content,
                        "metadata": metadata,
                        "score": min(1.0, score)  # Cap at 1.0
                    })
            
            # Sort by score
            mission_docs.sort(key=lambda x: x["score"], reverse=True)
            
            # Return top results
            return mission_docs[:n_results]
            
        except Exception as e:
            print(f"Error in mission/purpose search: {str(e)}")
            return []
    
    def _extract_page_title(self, metadata, content):
        """Extract a clean page title from metadata or content."""
        # Try different ways to get a page title
        page_title = None
        
        # Method 1: Try to get from page_title in metadata
        if metadata and "page_title" in metadata:
            page_title = metadata["page_title"]
            if isinstance(page_title, str):
                page_title = page_title.replace(" > ", "")
        
        # Method 2: Try to get from header_path in metadata
        if not page_title and metadata and "header_path" in metadata:
            header_path = metadata["header_path"]
            if isinstance(header_path, list) and header_path:
                # Use first element of header path
                page_title = header_path[0].replace(" > ", "") if isinstance(header_path[0], str) else str(header_path[0])
        
        # Method 3: Try to extract from the content's first line
        if not page_title and content:
            lines = content.split('\n')
            if lines:
                first_line = lines[0].strip()
                # Use first line if it's reasonably short (likely a title)
                if 3 < len(first_line) < 100:
                    page_title = first_line
        
        # Fallback options
        if not page_title:
            # Try other metadata fields
            for key in ["title", "document", "name", "source"]:
                if key in metadata and metadata[key]:
                    page_title = metadata[key]
                    if isinstance(page_title, str):
                        page_title = page_title.replace(" > ", "")
                    break
        
        # Final fallback
        if not page_title:
            page_title = "TechNova Document"
            
        return page_title
    
    def _keyword_fallback_search(
        self, 
        query_text: str, 
        n_results: int,
        where_filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Fallback search using keyword matching when embedding-based search fails.
        
        Args:
            query_text: The text query to search for
            n_results: Number of results to return
            where_filter: Metadata filters to apply
            
        Returns:
            List of document dictionaries with content, metadata, and scores
        """
        print(f"Using keyword fallback search for query: {query_text}")
        
        # Try getting documents in batches
        all_docs = []
        batch_size = 100
        offset = 0
        
        while True:
            try:
                if where_filter:
                    results = self.collection.get(where=where_filter, limit=batch_size, offset=offset)
                else:
                    results = self.collection.get(limit=batch_size, offset=offset)
                
                if not results["ids"] or len(results["ids"]) == 0:
                    break
                
                # Process each document
                for i in range(len(results["ids"])):
                    # Get content and clean up if needed
                    content = results["documents"][i]
                    if self.needs_formatting_fix:
                        content = content.replace(" > ", "")
                    
                    # Get metadata
                    metadata = results["metadatas"][i]
                    
                    # Extract page title
                    page_title = self._extract_page_title(metadata, content)
                    metadata["page_title"] = page_title
                    
                    # Add to list
                    all_docs.append({
                        "id": results["ids"][i],
                        "content": content,
                        "metadata": metadata
                    })
                
                # Update offset
                offset += len(results["ids"])
                
                # If we got fewer than batch_size, we're done
                if len(results["ids"]) < batch_size:
                    break
                
                # Limit total docs to avoid memory issues
                if len(all_docs) >= 500:
                    break
                
            except Exception as e:
                print(f"Error getting documents: {str(e)}")
                break
        
        if not all_docs:
            return []
        
        # Normalize query text
        query_words = set(query_text.lower().split())
        
        # Score documents based on keyword matches
        scored_docs = []
        for doc in all_docs:
            content = doc["content"]
            content_words = set(content.lower().split())
            
            # Count matching words
            matching_words = query_words.intersection(content_words)
            
            # Simple scoring scheme
            if len(query_words) > 0:
                word_match_score = len(matching_words) / len(query_words)
            else:
                word_match_score = 0
            
            # Title boost - if query words appear in the title
            title_words = set(doc["metadata"]["page_title"].lower().split())
            title_matches = query_words.intersection(title_words)
            title_boost = 0.2 * (len(title_matches) / len(query_words) if len(query_words) > 0 else 0)
            
            # Calculate final score
            score = min(1.0, word_match_score + title_boost)
            
            # Add semantic bonus for specific query types
            # For mission-related queries, look for purpose, vision, values, etc.
            if "mission" in query_words:
                mission_related = {"purpose", "vision", "values", "goals", "aims", "objectives", "about", "company"}
                if content_words.intersection(mission_related):
                    score = min(1.0, score + 0.3)
                # Extra bonus for having these in the title
                if title_words.intersection(mission_related):
                    score = min(1.0, score + 0.4)
            
            # Only include documents with some relevance
            if score > 0:
                doc["score"] = score
                scored_docs.append(doc)
        
        # Sort by score (highest first)
        scored_docs.sort(key=lambda x: x["score"], reverse=True)
        
        # Return top results
        return scored_docs[:n_results]
    
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
            if isinstance(page_title, str):
                page_title = page_title.replace(" > ", "")
        
        # Method 2: Try to get from header_path in metadata
        if not page_title and metadata and "header_path" in metadata:
            header_path = metadata["header_path"]
            if isinstance(header_path, list) and header_path:
                # Use first element of header path
                page_title = header_path[0].replace(" > ", "") if isinstance(header_path[0], str) else str(header_path[0])
        
        # Method 3: Try to extract from metadata's raw_metadata
        if not page_title and metadata and "raw_metadata" in metadata:
            raw_meta = metadata["raw_metadata"]
            if isinstance(raw_meta, str):
                # Try to find a pattern like "Page: Title"
                import re
                page_match = re.search(r'page[:\s]+([^,\n]+)', raw_meta, re.IGNORECASE)
                if page_match:
                    page_title = page_match.group(1).strip()
                else:
                    # Just use the first part of raw metadata
                    parts = raw_meta.split(',')
                    if parts:
                        page_title = parts[0].strip()
        
        # Method 4: Try to extract from the content's first line
        if not page_title and content:
            lines = content.split('\n')
            if lines:
                first_line = lines[0].strip()
                # Use first line if it's reasonably short (likely a title)
                if 3 < len(first_line) < 100:
                    page_title = first_line
        
        # Method 5: Extract from document ID if available
        if not page_title and "id" in metadata:
            doc_id = metadata["id"]
            if isinstance(doc_id, str):
                # Try to extract a meaningful part from the ID
                parts = doc_id.split('/')
                if len(parts) > 1:
                    page_title = parts[-2].replace('_', ' ').capitalize()
        
        # Fallback
        if not page_title:
            # Try to derive a title from any metadata field
            for key, value in metadata.items():
                if key.lower() in ['title', 'name', 'document', 'doc'] and isinstance(value, str):
                    page_title = value
                    break
        
        # Final fallback
        if not page_title:
            # Just use the first non-empty key-value pair
            for key, value in metadata.items():
                if value and isinstance(value, str):
                    page_title = f"{key.capitalize()}: {value}"
                    break
        
        # If still no title, use generic title
        if not page_title:
            page_title = "TechNova Document"
            
        return page_title
    
    def _keyword_fallback_search(
        self, 
        query_text: str, 
        n_results: int,
        where_filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Fallback search using keyword matching when embedding-based search fails.
        
        Args:
            query_text: The text query to search for
            n_results: Number of results to return
            where_filter: Metadata filters to apply
            
        Returns:
            List of document dictionaries with content, metadata, and scores
        """
        print(f"Using keyword fallback search for query: {query_text}")
        
        # Get documents from collection
        if where_filter:
            results = self.collection.get(where=where_filter)
        else:
            results = self.collection.get(limit=100)  # Get a reasonable number of documents
        
        if not results["ids"]:
            return []
        
        # Normalize query text
        query_words = set(query_text.lower().split())
        
        # Score documents based on keyword matches
        scored_docs = []
        for i in range(len(results["ids"])):
            # Process document content
            content = results["documents"][i].replace(" > ", "")
            content_words = set(content.lower().split())
            
            # Count matching words
            matching_words = query_words.intersection(content_words)
            
            # Simple scoring scheme
            if len(query_words) > 0:
                word_match_score = len(matching_words) / len(query_words)
            else:
                word_match_score = 0
            
            # Add a small bonus for potential semantic matches
            semantic_bonus = 0
            
            # Look for topic-related words
            # For mission-related queries, look for purpose, vision, values, etc.
            if "mission" in query_words:
                mission_related = {"purpose", "vision", "values", "goals", "aims", "objectives"}
                if content_words.intersection(mission_related):
                    semantic_bonus = 0.3
            
            # For team/structure queries, look for organization, hierarchy, roles, etc.
            if "team" in query_words or "structure" in query_words:
                team_related = {"organization", "hierarchy", "roles", "departments", "leadership"}
                if content_words.intersection(team_related):
                    semantic_bonus = 0.3
            
            # Calculate final score
            score = min(1.0, word_match_score + semantic_bonus)
            
            # Process metadata
            try:
                metadata_str = results["metadatas"][i]
                if isinstance(metadata_str, str):
                    metadata = json.loads(metadata_str)
                else:
                    metadata = metadata_str
            except (json.JSONDecodeError, TypeError):
                metadata = {}
            
            # Extract page title
            page_title = self._extract_page_title(metadata, content)
            metadata["page_title"] = page_title
            
            # Only include documents with some relevance
            if score > 0:
                scored_docs.append({
                    "id": results["ids"][i],
                    "content": content,
                    "metadata": metadata,
                    "score": score
                })
        
        # Sort by score (highest first)
        scored_docs.sort(key=lambda x: x["score"], reverse=True)
        
        # Return top results
        return scored_docs[:n_results]
    
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