# src/database/simple_vector_store.py

"""
Simplified vector store implementation that doesn't rely on embeddings.
This is a fallback implementation for when we can't use the ChromaDB embeddings directly.
"""

import chromadb
import json
import random
from typing import List, Dict, Any, Optional

class SimpleVectorStore:
    """
    Simplified vector store that uses basic string matching instead of embeddings.
    This is a fallback solution for when we can't use the ChromaDB embeddings directly.
    """
    
    def __init__(self, db_path: str, collection_name: str = None):
        """
        Initialize the simple vector store.
        
        Args:
            db_path: Path to the ChromaDB directory
            collection_name: Name of the collection to use (will use first available if not specified)
        """
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
            print(f"Collection '{collection_name if collection_name else 'default'}' not found. Using '{self.collection_name}' instead.")
        else:
            raise ValueError(f"No collections found in the database at {db_path}")
        
        # Load all documents from the collection
        self._load_documents()
        print(f"Loaded {len(self.documents)} documents from collection '{self.collection_name}'")
    
    def _load_documents(self):
        """Load all documents from the collection into memory."""
        self.documents = []
        
        # Get collection
        collection = self.client.get_collection(self.collection_name)
        
        # We'll load documents in batches to avoid memory issues with large collections
        batch_size = 100
        offset = 0
        
        while True:
            try:
                # Get a batch of documents
                results = collection.get(limit=batch_size, offset=offset)
                
                if not results["ids"] or len(results["ids"]) == 0:
                    break
                
                # Process each document
                for i in range(len(results["ids"])):
                    # Try to parse metadata
                    try:
                        metadata_str = results["metadatas"][i]
                        if isinstance(metadata_str, str):
                            try:
                                metadata = json.loads(metadata_str)
                            except json.JSONDecodeError:
                                # If can't parse JSON, use as-is
                                metadata = {"raw_metadata": metadata_str}
                        else:
                            metadata = metadata_str if metadata_str else {}
                    except (TypeError, IndexError):
                        metadata = {}
                    
                    # Fix content formatting - remove ">" characters that separate letters
                    content = ""
                    if i < len(results["documents"]):
                        content = results["documents"][i]
                        # Clean up the content - remove ">" characters between letters
                        content = content.replace(" > ", "")
                    
                    # Extract and clean up page title
                    page_title = self._extract_page_title(metadata, content)
                    
                    # Update metadata with clean page title
                    metadata["page_title"] = page_title
                    
                    # Create document object
                    document = {
                        "id": results["ids"][i],
                        "content": content,
                        "metadata": metadata
                    }
                    self.documents.append(document)
                
                # Update offset for next batch
                offset += len(results["ids"])
                
                # If we got fewer documents than batch_size, we're done
                if len(results["ids"]) < batch_size:
                    break
                    
            except Exception as e:
                print(f"Error loading documents: {str(e)}")
                break
    
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
    
    def query(
        self, 
        query_text: str, 
        num_results: int = 5,
        filters: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Query the vector store using simple keyword matching.
        
        Args:
            query_text: The text query to search for
            num_results: Number of results to return
            filters: Metadata filters to apply
            
        Returns:
            List of document dictionaries with content, metadata, and scores
        """
        print(f"Using simple keyword matching for query: {query_text}")
        
        # Normalize query text
        query_words = set(query_text.lower().split())
        
        # Score documents based on keyword matches
        scored_docs = []
        for doc in self.documents:
            # Apply filters if specified
            if filters and not self._matches_filters(doc, filters):
                continue
                
            # Calculate simple score based on word overlap
            content_words = set(doc["content"].lower().split())
            
            # Count matching words
            matching_words = query_words.intersection(content_words)
            
            # Simple scoring scheme:
            # 1. Base score is the proportion of query words found in the document
            if len(query_words) > 0:
                word_match_score = len(matching_words) / len(query_words)
            else:
                word_match_score = 0
                
            # 2. Add a small bonus if query words appear in title or section headers
            title_bonus = 0
            if "page_title" in doc["metadata"]:
                title_words = set(doc["metadata"]["page_title"].lower().split())
                title_matches = query_words.intersection(title_words)
                title_bonus = 0.2 * (len(title_matches) / len(query_words) if len(query_words) > 0 else 0)
            
            # 3. Calculate final score (max 1.0)
            score = min(1.0, word_match_score + title_bonus)
            
            # Only include documents with some relevance
            if score > 0:
                scored_docs.append({
                    "id": doc["id"],
                    "content": doc["content"],
                    "metadata": doc["metadata"],
                    "score": score
                })
        
        # Sort by score (highest first)
        scored_docs.sort(key=lambda x: x["score"], reverse=True)
        
        # If we don't have enough results, add some random documents
        if len(scored_docs) < num_results:
            print(f"Adding random documents to meet requested count (found {len(scored_docs)}, need {num_results})")
            # Filter unselected documents that match filters
            remaining_docs = [
                doc for doc in self.documents 
                if doc["id"] not in [d["id"] for d in scored_docs]
                and (not filters or self._matches_filters(doc, filters))
            ]
            
            # Randomly select additional documents
            random.shuffle(remaining_docs)
            for doc in remaining_docs[:num_results - len(scored_docs)]:
                scored_docs.append({
                    "id": doc["id"],
                    "content": doc["content"],
                    "metadata": doc["metadata"],
                    "score": 0.1  # Low score for random docs
                })
        
        # Return top results
        return scored_docs[:num_results]
    
    def _matches_filters(self, doc: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """
        Check if a document matches the specified filters.
        
        Args:
            doc: Document to check
            filters: Dictionary of metadata filters
            
        Returns:
            True if document matches all filters, False otherwise
        """
        for key, value in filters.items():
            # Check if the key exists in metadata
            if key not in doc["metadata"]:
                return False
            
            # Check if the value matches
            doc_value = doc["metadata"][key]
            
            # Handle different types of values
            if isinstance(value, list):
                # Check if any value in the list matches
                if doc_value not in value:
                    return False
            else:
                # Direct comparison
                if doc_value != value:
                    return False
        
        return True