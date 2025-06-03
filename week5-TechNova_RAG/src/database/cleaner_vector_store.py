# src/database/cleaner_vector_store.py

"""
CleanerVectorStore - A direct implementation to fix all the issues we've encountered
This implementation handles keyword-based search for mission/purpose queries.
"""

import chromadb
import json
import os
import requests
from typing import List, Dict, Any, Optional

class CleanerVectorStore:
    """
    A cleaner, issue-free implementation of the vector store
    for TechNova Knowledge Navigator.
    """
    
    def __init__(self, db_path: str, collection_name: str = None):
        """
        Initialize the vector store.
        
        Args:
            db_path: Path to the ChromaDB directory
            collection_name: Name of the collection to use (will use first available if not specified)
        """
        self.db_path = db_path
        self.client = chromadb.PersistentClient(path=db_path)
        
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
        
        print(f"Initialized vector store with {self.collection.count()} documents in collection '{self.collection_name}'")
        
        # Load a sample of documents to test extraction
        self._load_sample_documents()
    
    def _load_sample_documents(self, sample_size=5):
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
    
    def query(self, query_text: str, num_results: int = 5, filters: Dict[str, Any] = None):
        """
        Query the vector store using keyword matching.
        
        Args:
            query_text: The text query to search for
            num_results: Number of results to return
            filters: Metadata filters to apply
            
        Returns:
            List of document dictionaries with content, metadata, and scores
        """
        query_lower = query_text.lower()
        
        # Check if this is a mission/purpose query
        is_mission_query = any(word in query_lower for word in [
            "mission", "purpose", "vision", "values", "about", 
            "what is technova", "who is technova", "what does technova do"
        ])
        
        if is_mission_query:
            print("Detected mission-related query, using specialized search")
            results = self._mission_purpose_search(num_results)
            if results and len(results) > 0:
                return results
                
        # Fall back to general keyword search
        return self._keyword_search(query_text, num_results, filters)
    
    def _mission_purpose_search(self, num_results: int) -> List[Dict[str, Any]]:
        """
        Special search for mission and purpose documents.
        
        Args:
            num_results: Number of results to return
            
        Returns:
            List of document dictionaries
        """
        print("Performing specialized mission/purpose search")
        
        # Key documents we want to prioritize - the exact titles that should contain mission info
        # CRITICAL: The "Mission Statement" should be the top priority
        priority_titles = [
            "Mission Statement",  # Top priority document
            "Mission StatementWhy It Matters",
            "Mission StatementHow We Execute Our Mission", 
            "Mission StatementMeasuring Our Impact",
            "Core Values",
            "Core ValuesOur Guiding Principles",
            "Vision 2025",
            "Vision 2025Our North Star",
            "Employee HandbookCompany Overview",
            "About TechNovaWho We Are",
            "About TechNovaWhat We Do",
            "Onboarding ChecklistFirst WeekCompany Introduction"
        ]
        
        # Manually assign priority scores to each title
        title_priority = {
            "mission statement": 1.0,  # Highest priority
            "missionstatement": 1.0,   # No spaces version
            "mission statement why it matters": 0.98,
            "missionstatementwhyitmatters": 0.98,
            "mission statement how we execute our mission": 0.96,
            "missionstatementhowweexecuteourmission": 0.96,
            "mission statement measuring our impact": 0.94,
            "missionstatementmeasuringourimpact": 0.94,
            "core values": 0.92,
            "corevalues": 0.92,
            "vision 2025": 0.90,
            "vision2025": 0.90,
            "employee handbook company overview": 0.85,
            "employeehandbookcompanyoverview": 0.85,
            "about technova who we are": 0.80,
            "abouttechnovawhoweare": 0.80,
            "about technova what we do": 0.75,
            "abouttechnova": 0.75
        }
        
        mission_docs = []
        
        try:
            # Get enough documents to search through - increase limit to find more
            results = self.collection.get(limit=500)
            
            if not results["ids"]:
                return []
            
            # First pass - look specifically for exact Mission Statement document
            mission_statement_docs = []
            
            for i in range(len(results["ids"])):
                doc_id = results["ids"][i]
                content = results["documents"][i]
                metadata = results["metadatas"][i]
                
                # Clean up content if needed
                if self.needs_formatting_fix:
                    content = content.replace(" > ", "")
                
                # Extract page title
                page_title = self._extract_page_title(metadata, content)
                
                # Convert to lowercase and remove spaces for flexible matching
                clean_title = page_title.lower().replace(" ", "")
                
                # Check if this is a mission statement document
                if "missionstatement" in clean_title:
                    print(f"Found Mission Statement document: {page_title}")
                    score = title_priority.get(clean_title, 0.95)  # Default high score for any mission statement doc
                    metadata["page_title"] = page_title
                    mission_statement_docs.append({
                        "id": doc_id,
                        "content": content,
                        "metadata": metadata,
                        "score": score
                    })
            
            # If we found any mission statement docs, return those immediately
            if mission_statement_docs:
                print(f"Found {len(mission_statement_docs)} Mission Statement documents - using these")
                mission_statement_docs.sort(key=lambda x: x["score"], reverse=True)
                return mission_statement_docs[:num_results]
            
            # Second pass - broader search for any mission-related documents
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
                
                # Convert to lowercase and remove spaces for flexible matching
                clean_title = page_title.lower().replace(" ", "")
                
                # Check against our priority title list
                for priority_title in priority_titles:
                    clean_priority = priority_title.lower().replace(" ", "")
                    if clean_priority in clean_title:
                        # Use predefined priority scores if available
                        score = title_priority.get(clean_title, 0.5)
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
                        "score": score
                    })
            
            # Sort by score
            mission_docs.sort(key=lambda x: x["score"], reverse=True)
            
            # Print what we found to help with debugging
            print(f"Found {len(mission_docs)} mission-related documents")
            if mission_docs:
                print("Top documents found:")
                for i, doc in enumerate(mission_docs[:5]):
                    print(f"  {i+1}. {doc['metadata']['page_title']} (Score: {doc['score']:.2f})")
            
            # Return top results
            return mission_docs[:num_results]
            
        except Exception as e:
            print(f"Error in mission/purpose search: {str(e)}")
            return []
    
    def _keyword_search(
        self, 
        query_text: str, 
        num_results: int,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search using keyword matching.
        
        Args:
            query_text: The text query to search for
            num_results: Number of results to return
            filters: Metadata filters to apply
            
        Returns:
            List of document dictionaries
        """
        print(f"Performing keyword search for: {query_text}")
        
        # Try getting documents in batches
        all_docs = []
        batch_size = 100
        offset = 0
        
        while True:
            try:
                if filters:
                    results = self.collection.get(where=filters, limit=batch_size, offset=offset)
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