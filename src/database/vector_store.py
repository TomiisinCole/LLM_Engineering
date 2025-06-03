"""
Vector database module for TechNova Knowledge Navigator.
Implements Chroma DB integration for storing and retrieving embeddings.
"""
import os
import json
import time
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path
import numpy as np
#from loguru import logger
import chromadb
from chromadb.utils import embedding_functions

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


class VectorDBConfig:
    """Configuration for vector database."""
    
    def __init__(
        self,
        persist_directory: str = "./data/chroma_db",
        collection_name: str = "technova_knowledge",
        embedding_dimension: int = 1536,
        distance_func: str = "cosine"
    ):
        """
        Initialize vector database configuration.
        
        Args:
            persist_directory: Directory to persist the database
            collection_name: Name of the collection
            embedding_dimension: Dimension of embeddings
            distance_func: Distance function for similarity search
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embedding_dimension = embedding_dimension
        self.distance_func = distance_func


class VectorStore:
    """Vector database wrapper for Chroma DB."""
    
    def __init__(self, config: Optional[VectorDBConfig] = None):
        """
        Initialize vector database.
        
        Args:
            config: Vector database configuration
        """
        self.config = config or VectorDBConfig()
        
        # Ensure persist directory exists
        Path(self.config.persist_directory).mkdir(parents=True, exist_ok=True)
        
        # Initialize Chroma client
        self.client = chromadb.PersistentClient(path=self.config.persist_directory)
        
        # Create or get collection
        try:
            self.collection = self.client.get_collection(
                name=self.config.collection_name,
                embedding_function=None  # We'll provide embeddings directly
            )
            logger.info(f"Using existing collection '{self.config.collection_name}'")
        except Exception:
            logger.info(f"Creating new collection '{self.config.collection_name}'")
            self.collection = self.client.create_collection(
                name=self.config.collection_name,
                embedding_function=None,  # We'll provide embeddings directly
                metadata={"dimension": self.config.embedding_dimension}
            )
    
    def add_documents(
        self,
        documents: List[Dict[str, Any]],
        batch_size: int = 100
    ) -> Dict[str, Any]:
        """
        Add documents with embeddings to the vector store.
        
        Args:
            documents: List of document dictionaries with embeddings
            batch_size: Number of documents to add in a single batch
            
        Returns:
            Dictionary with statistics about the operation
        """
        if not documents:
            logger.warning("No documents to add")
            return {"added": 0, "errors": 0}
        
        # Prepare documents for insertion
        ids = []
        embeddings = []
        metadatas = []
        documents_text = []
        
        for doc in documents:
            ids.append(doc["chunk_id"])
            embeddings.append(doc["embedding"])
            
            # Include essential metadata for retrieval
            metadata = {
                "source_id": doc["metadata"]["source_id"],
                "source_title": doc["metadata"]["source_title"],
                "source_url": doc["metadata"]["source_url"],
                "header_path": doc["metadata"]["header_path"],
                "token_count": doc["metadata"]["token_count"],
                "chunk_index": doc["metadata"]["chunk_index"]
            }
            metadatas.append(metadata)
            
            # Store the actual text content
            documents_text.append(doc["text"])
        
        # Add documents in batches
        total_docs = len(documents)
        batches = [
            (
                ids[i:i+batch_size],
                embeddings[i:i+batch_size],
                metadatas[i:i+batch_size],
                documents_text[i:i+batch_size]
            )
            for i in range(0, total_docs, batch_size)
        ]
        
        stats = {"added": 0, "errors": 0}
        
        for i, (batch_ids, batch_embeddings, batch_metadatas, batch_texts) in enumerate(batches):
            try:
                logger.info(f"Adding batch {i+1}/{len(batches)} with {len(batch_ids)} documents")
                
                self.collection.add(
                    ids=batch_ids,
                    embeddings=batch_embeddings,
                    metadatas=batch_metadatas,
                    documents=batch_texts
                )
                
                stats["added"] += len(batch_ids)
                
                # Brief pause between batches
                if i < len(batches) - 1:
                    time.sleep(0.1)
            
            except Exception as e:
                logger.error(f"Error adding batch {i+1}: {e}")
                stats["errors"] += len(batch_ids)
        
        logger.info(f"Added {stats['added']} documents to vector store")
        if stats["errors"] > 0:
            logger.warning(f"Failed to add {stats['errors']} documents")
        
        return stats
    
    def query(
        self,
        query_embedding: List[float],
        n_results: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
        include_embeddings: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Query the vector store for similar documents.
        
        Args:
            query_embedding: Query embedding vector
            n_results: Number of results to return
            filter_metadata: Optional metadata filter
            include_embeddings: Whether to include embeddings in results
            
        Returns:
            List of similar documents with metadata
        """
        # Prepare where filter if needed
        where = filter_metadata if filter_metadata else None
        
        # Query the collection
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where,
            include=["documents", "metadatas", "distances"]
        )
        
        # Format results
        formatted_results = []
        
        if results["ids"] and len(results["ids"][0]) > 0:
            for i, doc_id in enumerate(results["ids"][0]):
                result = {
                    "chunk_id": doc_id,
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "similarity": 1 - results["distances"][0][i]  # Convert distance to similarity
                }
                formatted_results.append(result)
        
        return formatted_results
    
    def update_document(
        self,
        chunk_id: str,
        embedding: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        text: Optional[str] = None
    ) -> bool:
        """
        Update a document in the vector store.
        
        Args:
            chunk_id: ID of the document to update
            embedding: Optional new embedding
            metadata: Optional new metadata
            text: Optional new text
            
        Returns:
            True if update was successful, False otherwise
        """
        try:
            # Check if document exists
            results = self.collection.get(ids=[chunk_id])
            if not results["ids"]:
                logger.warning(f"Document {chunk_id} not found")
                return False
            
            # Prepare update arguments
            update_args = {"ids": [chunk_id]}
            
            if embedding is not None:
                update_args["embeddings"] = [embedding]
            
            if metadata is not None:
                update_args["metadatas"] = [metadata]
            
            if text is not None:
                update_args["documents"] = [text]
            
            # Update the document
            self.collection.update(**update_args)
            logger.info(f"Updated document {chunk_id}")
            
            return True
        
        except Exception as e:
            logger.error(f"Error updating document {chunk_id}: {e}")
            return False
    
    def delete_document(self, chunk_id: str) -> bool:
        """
        Delete a document from the vector store.
        
        Args:
            chunk_id: ID of the document to delete
            
        Returns:
            True if deletion was successful, False otherwise
        """
        try:
            self.collection.delete(ids=[chunk_id])
            logger.info(f"Deleted document {chunk_id}")
            return True
        
        except Exception as e:
            logger.error(f"Error deleting document {chunk_id}: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store.
        
        Returns:
            Dictionary with statistics
        """
        stats = {}
        
        # Get collection info
        try:
            # Count documents
            count = self.collection.count()
            stats["document_count"] = count
            
            # Get collection metadata
            collection_info = self.client.get_collection(self.config.collection_name)
            stats["collection_name"] = self.config.collection_name
            stats["dimension"] = self.config.embedding_dimension
            
            # Get IDs for a sample of documents
            if count > 0:
                sample = min(5, count)
                sample_results = self.collection.get(limit=sample)
                stats["sample_ids"] = sample_results["ids"]
        
        except Exception as e:
            logger.error(f"Error getting vector store stats: {e}")
            stats["error"] = str(e)
        
        return stats


def load_embedded_chunks(
    input_dir: str,
    limit: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Load chunks with embeddings from the input directory.
    
    Args:
        input_dir: Directory containing chunks with embeddings
        limit: Optional limit on number of chunks to load
        
    Returns:
        List of chunk dictionaries with embeddings
    """
    # Find all chunk files
    chunk_files = list(Path(input_dir).glob("*.json"))
    chunk_files = [f for f in chunk_files if f.name not in ["embeddings_index.json", "embedding_stats.json"]]
    
    # Apply limit if specified
    if limit is not None:
        chunk_files = chunk_files[:limit]
    
    logger.info(f"Loading {len(chunk_files)} chunks with embeddings")
    
    # Load chunks
    chunks = []
    
    for chunk_file in chunk_files:
        try:
            with open(chunk_file, 'r', encoding='utf-8') as f:
                chunk = json.load(f)
            
            # Ensure chunk has required fields
            if "chunk_id" not in chunk or "embedding" not in chunk:
                logger.warning(f"Chunk {chunk_file} missing required fields, skipping")
                continue
            
            chunks.append(chunk)
        
        except Exception as e:
            logger.error(f"Error loading chunk {chunk_file}: {e}")
    
    logger.info(f"Loaded {len(chunks)} chunks with embeddings")
    
    return chunks


def build_vector_db(
    input_dir: str,
    db_config: Optional[VectorDBConfig] = None,
    limit: Optional[int] = None
) -> Dict[str, Any]:
    """
    Build vector database from chunks with embeddings.
    
    Args:
        input_dir: Directory containing chunks with embeddings
        db_config: Optional vector database configuration
        limit: Optional limit on number of chunks to load
        
    Returns:
        Dictionary with statistics about the operation
    """
    # Load chunks with embeddings
    chunks = load_embedded_chunks(input_dir, limit)
    
    if not chunks:
        logger.warning("No chunks with embeddings found")
        return {"added": 0, "errors": 0}
    
    # Initialize vector store
    vector_store = VectorStore(db_config)
    
    # Add chunks to vector store
    stats = vector_store.add_documents(chunks)
    
    # Get vector store stats
    db_stats = vector_store.get_stats()
    
    # Combine statistics
    result_stats = {**stats, **db_stats}
    
    # Save statistics
    stats_file = os.path.join(input_dir, "vector_db_stats.json")
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(result_stats, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Vector database built with {stats['added']} documents")
    logger.info(f"Database statistics: {result_stats}")
    
    return result_stats