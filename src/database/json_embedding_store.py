# src/database/json_embedding_store.py

"""
Vector store implementation that uses embeddings stored in JSON files.
"""

import os
import json
import numpy as np
import time
from typing import List, Dict, Any, Optional
import requests
from dotenv import load_dotenv

# Load environment variables for API keys
load_dotenv()

class JsonEmbeddingStore:
    """
    Vector store that uses embeddings stored in JSON files.
    Optimized for TechNova's data structure where each JSON file contains
    both the document and its embedding.
    """
    
    def __init__(self, embeddings_path: str = "data/embedded"):
        """
        Initialize the vector store with embeddings stored in JSON files.
        
        Args:
            embeddings_path: Path to the directory containing embedded JSON files
        """
        self.embeddings_path = embeddings_path
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        
        print(f"Initializing vector store with embeddings from: {embeddings_path}")
        
        # Load the documents with embeddings
        self.documents = self._load_documents()
        
        print(f"Loaded {len(self.documents)} documents with embeddings")
        
        # Print document structure for debugging
        if self.documents:
            print("Document structure sample:")
            sample_doc = self.documents[0]
            print(f"  chunk_id: {sample_doc['chunk_id']}")
            print(f"  text length: {len(sample_doc['text'])}")
            if 'metadata' in sample_doc:
                print(f"  metadata keys: {list(sample_doc['metadata'].keys())}")
            print(f"  embedding length: {len(sample_doc['embedding'])}")
    
    def _load_documents(self) -> List[Dict[str, Any]]:
        """
        Load documents with embeddings from JSON files.
        
        Returns:
            List of document dictionaries with embeddings
        """
        documents = []
        
        # Check if embeddings path exists
        if not os.path.exists(self.embeddings_path):
            raise ValueError(f"Embeddings path does not exist: {self.embeddings_path}")
        
        # Get all JSON files in the directory
        json_files = [f for f in os.listdir(self.embeddings_path) if f.endswith('.json')]
        print(f"Found {len(json_files)} JSON files in {self.embeddings_path}")
        
        # Load each JSON file
        file_count = 0
        start_time = time.time()
        
        for json_file in json_files:
            try:
                with open(os.path.join(self.embeddings_path, json_file), 'r') as f:
                    doc_data = json.load(f)
                
                # Ensure the document has an embedding
                if 'embedding' not in doc_data:
                    print(f"Warning: Document {json_file} has no embedding, skipping")
                    continue
                
                # Add the document to the list
                documents.append(doc_data)
                file_count += 1
                
                # Print progress every 100 files
                if file_count % 100 == 0:
                    elapsed = time.time() - start_time
                    print(f"Loaded {file_count} files in {elapsed:.2f} seconds")
                
            except Exception as e:
                print(f"Error loading document {json_file}: {str(e)}")
        
        print(f"Successfully loaded {len(documents)} documents with embeddings")
        return documents
    
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
                    "input": query_text
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
    
    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Args:
            a: First vector
            b: Second vector
            
        Returns:
            Cosine similarity (between -1 and 1, higher is more similar)
        """
        # Convert to numpy arrays
        a_np = np.array(a)
        b_np = np.array(b)
        
        # Calculate dot product and magnitudes
        dot_product = np.dot(a_np, b_np)
        norm_a = np.linalg.norm(a_np)
        norm_b = np.linalg.norm(b_np)
        
        # Calculate cosine similarity
        if norm_a == 0 or norm_b == 0:
            return 0  # Handle division by zero
        else:
            return dot_product / (norm_a * norm_b)
    
    def query(
        self, 
        query_text: str, 
        num_results: int = 5, 
        filters: Dict[str, Any] = None,
        threshold: float = 0.2  # Lower threshold because of embedding model differences
    ) -> List[Dict[str, Any]]:
        """
        Query the vector store using embedding similarity.
        
        Args:
            query_text: The text query to search for
            num_results: Number of results to return
            filters: Metadata filters to apply
            threshold: Minimum similarity score
            
        Returns:
            List of document dictionaries with content, metadata, and scores
        """
        query_lower = query_text.lower()
        
        # Special handling for mission/vision/values queries
        is_mission_query = "mission" in query_lower
        is_vision_query = "vision" in query_lower or "2025" in query_lower
        is_values_query = "values" in query_lower or "principles" in query_lower
        
        print(f"Processing query: {query_text}")
        
        # Get embeddings for the query
        query_embedding = self._embed_query(query_text)
        
        if query_embedding is None:
            print("Failed to generate embedding for query, falling back to keyword search")
            return self._keyword_search(query_text, num_results, filters)
        
        # Calculate similarity scores
        print("Calculating similarity scores...")
        start_time = time.time()
        
        similarities = []
        for i, doc in enumerate(self.documents):
            # Ensure the document has an embedding
            if 'embedding' not in doc:
                continue
            
            # Apply filters if specified
            if filters:
                # Check if document metadata matches filters
                doc_metadata = doc.get("metadata", {})
                matches_filter = True
                for key, value in filters.items():
                    if key not in doc_metadata or doc_metadata[key] != value:
                        matches_filter = False
                        break
                
                # Skip if document doesn't match filters
                if not matches_filter:
                    continue
            
            # Calculate similarity
            similarity = self._cosine_similarity(query_embedding, doc['embedding'])
            
            # Add document and similarity score if above threshold
            if similarity >= threshold:
                # Extract page title
                metadata = doc.get("metadata", {})
                
                # Adjust raw metadata to match our expected format
                if 'header_path' not in metadata and 'source_title' in metadata:
                    metadata['page_title'] = metadata['source_title']
                    
                page_title = self._extract_page_title(metadata, doc.get("text", ""))
                metadata["page_title"] = page_title
                
                # Special handling for mission/vision/values queries
                # Boost scores for documents with relevant titles
                boost = 0.0
                title_lower = page_title.lower()
                
                if is_mission_query and "mission" in title_lower:
                    boost = 0.2
                    print(f"Boosting mission document: {page_title}")
                elif is_vision_query and ("vision" in title_lower or "2025" in title_lower):
                    boost = 0.2
                    print(f"Boosting vision document: {page_title}")
                elif is_values_query and ("values" in title_lower or "principles" in title_lower):
                    boost = 0.2
                    print(f"Boosting values document: {page_title}")
                
                # Add boosted similarity score
                boosted_similarity = min(1.0, similarity + boost)
                
                # Extra boost for exact title matches
                if "mission statement" in title_lower and "mission" in query_lower:
                    boosted_similarity = min(1.0, boosted_similarity + 0.3)
                    print(f"Extra boost for Mission Statement: {page_title}")
                elif "vision 2025" in title_lower and "vision" in query_lower:
                    boosted_similarity = min(1.0, boosted_similarity + 0.3)
                    print(f"Extra boost for Vision 2025: {page_title}")
                elif "core values" in title_lower and "values" in query_lower:
                    boosted_similarity = min(1.0, boosted_similarity + 0.3)
                    print(f"Extra boost for Core Values: {page_title}")
                
                similarities.append({
                    "index": i,
                    "document": doc,
                    "similarity": boosted_similarity
                })
        
        elapsed_time = time.time() - start_time
        print(f"Calculated similarity scores for {len(self.documents)} documents in {elapsed_time:.2f} seconds")
        
        # Sort by similarity score (descending)
        similarities.sort(key=lambda x: x["similarity"], reverse=True)
        
        print(f"Found {len(similarities)} documents above threshold {threshold}")
        
        # Show top scoring documents
        if similarities:
            print("Top scoring documents:")
            for i, item in enumerate(similarities[:5]):
                print(f"  {i+1}. {item['document'].get('metadata', {}).get('page_title', 'Unknown')} (Score: {item['similarity']:.2f})")
        
        # Return top results
        results = []
        for i, item in enumerate(similarities[:num_results]):
            doc = item["document"]
            metadata = doc.get("metadata", {}).copy()  # Create a copy to avoid modifying the original
            
            # Cleanup and enrich metadata
            if 'page_title' not in metadata:
                metadata['page_title'] = self._extract_page_title(metadata, doc.get("text", ""))
                
            # Format the result
            result = {
                "id": doc.get("chunk_id", f"doc_{i}"),
                "content": doc.get("text", ""),  # Critical: Make sure we're getting the content correctly
                "metadata": metadata,
                "score": item["similarity"]
            }
            
            # Debug the content being returned
            content_preview = result["content"][:100] + "..." if len(result["content"]) > 100 else result["content"]
            print(f"Result {i+1} content preview: {content_preview}")
            
            results.append(result)
        
        # If no results found with embeddings, fall back to keyword search
        if not results:
            print("No results found with embeddings, falling back to keyword search")
            return self._keyword_search(query_text, num_results, filters)
        
        return results
    
    def _keyword_search(
        self, 
        query_text: str, 
        num_results: int,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search using keyword matching as a fallback.
        
        Args:
            query_text: The text query to search for
            num_results: Number of results to return
            filters: Metadata filters to apply
            
        Returns:
            List of document dictionaries
        """
        print(f"Performing keyword search for: {query_text}")
        
        # Normalize query text
        query_words = set(query_text.lower().split())
        
        # Score documents based on keyword matches
        scored_docs = []
        for i, doc in enumerate(self.documents):
            content = doc.get("text", "")
            metadata = doc.get("metadata", {})
            
            # Apply filters if specified
            if filters:
                matches_filter = True
                for key, value in filters.items():
                    if key not in metadata or metadata[key] != value:
                        matches_filter = False
                        break
                
                # Skip if document doesn't match filters
                if not matches_filter:
                    continue
            
            # Score by keyword matching
            content_words = set(content.lower().split())
            
            # Count matching words
            matching_words = query_words.intersection(content_words)
            
            if len(query_words) > 0:
                word_match_score = len(matching_words) / len(query_words)
            else:
                word_match_score = 0
            
            # Get page title
            page_title = self._extract_page_title(metadata, content)
            
            # Special handling for mission/vision/values
            boost = 0.0
            title_lower = page_title.lower()
            query_lower = query_text.lower()
            
            if "mission" in query_lower and "mission" in title_lower:
                boost = 0.4
            elif "vision" in query_lower and ("vision" in title_lower or "2025" in title_lower):
                boost = 0.4
            elif "values" in query_lower and ("values" in title_lower or "principles" in title_lower):
                boost = 0.4
            
            # Title boost
            title_words = set(page_title.lower().split())
            title_matches = query_words.intersection(title_words)
            title_boost = 0.2 * (len(title_matches) / len(query_words) if len(query_words) > 0 else 0)
            
            # Calculate final score
            score = min(1.0, word_match_score + title_boost + boost)
            
            # Only include if score is positive
            if score > 0:
                # Create a copy of metadata and add page_title
                metadata_copy = metadata.copy()
                metadata_copy["page_title"] = page_title
                
                scored_docs.append({
                    "id": doc.get("chunk_id", f"doc_{i}"),
                    "content": content,
                    "metadata": metadata_copy,
                    "score": score
                })
        
        # Sort by score
        scored_docs.sort(key=lambda x: x["score"], reverse=True)
        
        # Return top results
        return scored_docs[:num_results]
    
    def _extract_page_title(self, metadata, content):
        """Extract a clean page title from metadata or content."""
        # Try different ways to get a page title
        page_title = None
        
        # Method 1: Try to get from existing page_title in metadata
        if metadata and "page_title" in metadata:
            page_title = metadata["page_title"]
            if isinstance(page_title, str):
                page_title = page_title.replace(" > ", "")
        
        # Method 2: Try to get from source_title in metadata
        if not page_title and metadata and "source_title" in metadata:
            page_title = metadata["source_title"]
            if isinstance(page_title, str):
                page_title = page_title.replace(" > ", "")
        
        # Method 3: Try to get from header_path in metadata
        if not page_title and metadata and "header_path" in metadata:
            header_path = metadata["header_path"]
            if isinstance(header_path, list) and header_path:
                # Use first element of header path
                page_title = header_path[0].replace(" > ", "") if isinstance(header_path[0], str) else str(header_path[0])
            elif isinstance(header_path, str):
                page_title = header_path.replace(" > ", "")
        
        # Method 4: Try to extract from the content's first line
        if not page_title and content:
            lines = content.split('\n')
            if lines:
                first_line = lines[0].strip()
                # Use first line if it's reasonably short (likely a title)
                if 3 < len(first_line) < 100:
                    page_title = first_line
        
        # Final fallback
        if not page_title:
            page_title = "TechNova Document"
            
        return page_title