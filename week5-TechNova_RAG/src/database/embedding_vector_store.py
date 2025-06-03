# src/database/embedding_vector_store.py

"""
Vector store implementation that uses pre-computed embeddings.
"""

import os
import json
import numpy as np
import pickle
from typing import List, Dict, Any, Optional
import requests
from dotenv import load_dotenv

# Load environment variables for API keys
load_dotenv()

class EmbeddingVectorStore:
    """
    Vector store that uses pre-computed embeddings for semantic search.
    """
    
    def __init__(self, embeddings_path: str = "data/embedded", documents_path: str = "data/chunked"):
        """
        Initialize the vector store with pre-computed embeddings.
        
        Args:
            embeddings_path: Path to the pre-computed embeddings
            documents_path: Path to the chunked documents
        """
        self.embeddings_path = embeddings_path
        self.documents_path = documents_path
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        
        print(f"Initializing vector store with embeddings from: {embeddings_path}")
        print(f"Loading documents from: {documents_path}")
        
        # Load the embeddings and documents
        self.documents, self.embeddings = self._load_data()
        
        print(f"Loaded {len(self.documents)} documents with embeddings")
    
    def _load_data(self) -> tuple:
        """
        Load pre-computed embeddings and corresponding documents.
        
        Returns:
            Tuple of (documents, embeddings)
        """
        documents = []
        embeddings = []
        
        # Check if embeddings path exists
        if not os.path.exists(self.embeddings_path):
            raise ValueError(f"Embeddings path does not exist: {self.embeddings_path}")
        
        # Load documents first
        if os.path.exists(os.path.join(self.documents_path, "chunks.json")):
            # If there's a consolidated JSON file
            with open(os.path.join(self.documents_path, "chunks.json"), "r") as f:
                documents = json.load(f)
            print(f"Loaded {len(documents)} documents from chunks.json")
        else:
            # Otherwise, try to load individual JSON files
            document_files = [f for f in os.listdir(self.documents_path) if f.endswith('.json')]
            for doc_file in document_files:
                try:
                    with open(os.path.join(self.documents_path, doc_file), "r") as f:
                        doc_data = json.load(f)
                        if isinstance(doc_data, list):
                            documents.extend(doc_data)
                        else:
                            documents.append(doc_data)
                except Exception as e:
                    print(f"Error loading document {doc_file}: {str(e)}")
            print(f"Loaded {len(documents)} documents from individual JSON files")
        
        # Now load embeddings
        # Check if there's a consolidated embeddings file
        if os.path.exists(os.path.join(self.embeddings_path, "embeddings.pkl")):
            with open(os.path.join(self.embeddings_path, "embeddings.pkl"), "rb") as f:
                embeddings = pickle.load(f)
            print(f"Loaded {len(embeddings)} embeddings from embeddings.pkl")
        else:
            # Otherwise, try to load individual embedding files
            embedding_files = [f for f in os.listdir(self.embeddings_path) if f.endswith('.pkl') or f.endswith('.npy')]
            for emb_file in embedding_files:
                try:
                    file_path = os.path.join(self.embeddings_path, emb_file)
                    if emb_file.endswith('.pkl'):
                        with open(file_path, "rb") as f:
                            emb_data = pickle.load(f)
                            if isinstance(emb_data, list):
                                embeddings.extend(emb_data)
                            else:
                                embeddings.append(emb_data)
                    elif emb_file.endswith('.npy'):
                        emb_data = np.load(file_path)
                        embeddings.append(emb_data)
                except Exception as e:
                    print(f"Error loading embedding {emb_file}: {str(e)}")
            print(f"Loaded {len(embeddings)} embeddings from individual files")
        
        # Ensure we have the same number of documents and embeddings
        if len(documents) != len(embeddings):
            print(f"Warning: Number of documents ({len(documents)}) does not match number of embeddings ({len(embeddings)})")
            # Use the smaller of the two
            min_len = min(len(documents), len(embeddings))
            documents = documents[:min_len]
            embeddings = embeddings[:min_len]
            print(f"Using the first {min_len} documents and embeddings")
        
        # Convert embeddings to numpy arrays if they aren't already
        processed_embeddings = []
        for emb in embeddings:
            if isinstance(emb, list):
                processed_embeddings.append(np.array(emb))
            else:
                processed_embeddings.append(emb)
        
        return documents, processed_embeddings
    
    def _embed_query(self, query_text: str) -> np.ndarray:
        """
        Generate embeddings for a query text using OpenAI API.
        
        Args:
            query_text: The text to embed
            
        Returns:
            Numpy array of embedding values
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
            
            return np.array(embedding)
            
        except Exception as e:
            print(f"Error generating embedding: {str(e)}")
            return None
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Args:
            a: First vector
            b: Second vector
            
        Returns:
            Cosine similarity (between -1 and 1, higher is more similar)
        """
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    def query(
        self, 
        query_text: str, 
        num_results: int = 5, 
        filters: Dict[str, Any] = None,
        threshold: float = 0.7
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
        
        # Get embeddings for the query
        query_embedding = self._embed_query(query_text)
        
        if query_embedding is None:
            print("Failed to generate embedding for query, falling back to keyword search")
            return self._keyword_search(query_text, num_results, filters)
        
        # Calculate similarity scores
        similarities = []
        for i, doc_embedding in enumerate(self.embeddings):
            # Skip if embeddings dimensions don't match
            if doc_embedding.shape != query_embedding.shape:
                continue
                
            # Calculate similarity
            similarity = self._cosine_similarity(query_embedding, doc_embedding)
            
            # Apply filters if specified
            if filters:
                # Check if document metadata matches filters
                doc_metadata = self.documents[i].get("metadata", {})
                matches_filter = True
                for key, value in filters.items():
                    if key not in doc_metadata or doc_metadata[key] != value:
                        matches_filter = False
                        break
                
                # Skip if document doesn't match filters
                if not matches_filter:
                    continue
            
            # Add document and similarity score if above threshold
            if similarity >= threshold:
                doc = self.documents[i]
                
                # Extract page title
                metadata = doc.get("metadata", {})
                page_title = self._extract_page_title(metadata, doc.get("content", ""))
                metadata["page_title"] = page_title
                
                # Special handling for mission/vision/values queries
                # Boost scores for documents with relevant titles
                boost = 0.0
                title_lower = page_title.lower()
                
                if is_mission_query and "mission" in title_lower:
                    boost = 0.2
                elif is_vision_query and ("vision" in title_lower or "2025" in title_lower):
                    boost = 0.2
                elif is_values_query and ("values" in title_lower or "principles" in title_lower):
                    boost = 0.2
                
                # Add boosted similarity score
                boosted_similarity = min(1.0, similarity + boost)
                
                similarities.append({
                    "index": i,
                    "document": doc,
                    "similarity": boosted_similarity
                })
        
        # Sort by similarity score (descending)
        similarities.sort(key=lambda x: x["similarity"], reverse=True)
        
        # Return top results
        results = []
        for i, item in enumerate(similarities[:num_results]):
            doc = item["document"]
            metadata = doc.get("metadata", {})
            
            # Format the result
            result = {
                "id": doc.get("id", f"doc_{i}"),
                "content": doc.get("content", ""),
                "metadata": metadata,
                "score": item["similarity"]
            }
            results.append(result)
        
        # If no results found with embeddings, fall back to keyword search
        if not results:
            print("No results found with embeddings, falling back to keyword search")
            return self._keyword_search(query_text, num_results, filters)
        
        print(f"Found {len(results)} relevant documents using embedding similarity")
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
            content = doc.get("content", "")
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
            metadata["page_title"] = page_title
            
            # Title boost
            title_words = set(page_title.lower().split())
            title_matches = query_words.intersection(title_words)
            title_boost = 0.2 * (len(title_matches) / len(query_words) if len(query_words) > 0 else 0)
            
            # Calculate final score
            score = min(1.0, word_match_score + title_boost)
            
            # Only include if score is positive
            if score > 0:
                scored_docs.append({
                    "id": doc.get("id", f"doc_{i}"),
                    "content": content,
                    "metadata": metadata,
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
        
        # Method 3: Try document title or source title
        if not page_title:
            for key in ["title", "source_title", "document_title"]:
                if key in metadata and metadata[key]:
                    page_title = metadata[key]
                    if isinstance(page_title, str):
                        page_title = page_title.replace(" > ", "")
                    break
        
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