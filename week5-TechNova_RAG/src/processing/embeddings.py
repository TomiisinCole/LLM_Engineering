"""
Embeddings module for TechNova Knowledge Navigator.
Handles generation of embeddings from text chunks using OpenAI API.
"""
import os
import json
import time
import math
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from tqdm import tqdm
import backoff
import requests
from loguru import logger
from dotenv import load_dotenv
from openai import OpenAI


# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = "text-embedding-3-small"  # OpenAI's embedding model
EMBEDDING_DIMS = 1536  # Dimensions for text-embedding-3-small
BATCH_SIZE = 20  # Number of texts to embed in a single API call
MAX_TOKENS_PER_BATCH = 8000  # Maximum tokens per batch (to stay within limits)


class EmbeddingConfig:
    """Configuration for embedding generation."""
    
    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL,
        dimensions: int = EMBEDDING_DIMS,
        batch_size: int = BATCH_SIZE,
        max_tokens_per_batch: int = MAX_TOKENS_PER_BATCH,
        api_key: Optional[str] = None
    ):
        """
        Initialize embedding configuration.
        
        Args:
            model_name: Name of the embedding model
            dimensions: Embedding dimensions
            batch_size: Number of texts to embed in a single API call
            max_tokens_per_batch: Maximum tokens per batch
            api_key: OpenAI API key (if None, uses env var)
        """
        self.model_name = model_name
        self.dimensions = dimensions
        self.batch_size = batch_size
        self.max_tokens_per_batch = max_tokens_per_batch
        self.api_key = api_key or OPENAI_API_KEY
        
        # Simple client initialization without extra parameters
        self.client = OpenAI(api_key=self.api_key)


@backoff.on_exception(
    backoff.expo,
    (requests.exceptions.RequestException, Exception),
    max_tries=5,
    giveup=lambda e: "invalid_request_error" in str(e),
)
def get_embedding(
    text: str,
    config: EmbeddingConfig,
    retry_count: int = 0
) -> List[float]:
    """
    Generate embedding for a single text using OpenAI API.
    Uses exponential backoff for handling rate limits.
    
    Args:
        text: Text to embed
        config: Embedding configuration
        retry_count: Current retry attempt (for logging)
        
    Returns:
        Embedding vector as a list of floats
    """
    try:
        response = config.client.embeddings.create(
            model=config.model_name,
            input=text
        )
        
        # Extract embedding from response
        embedding = response.data[0].embedding
        
        return embedding
    
    except Exception as e:
        # If we're already retrying, log the error
        if retry_count > 0:
            logger.warning(f"Retry {retry_count} failed for embedding generation: {e}")
        
        # Let backoff handle the retry
        raise


def batch_embed_texts(
    texts: List[str],
    config: EmbeddingConfig
) -> List[List[float]]:
    """
    Generate embeddings for multiple texts in batches.
    
    Args:
        texts: List of texts to embed
        config: Embedding configuration
        
    Returns:
        List of embedding vectors
    """
    if not texts:
        return []
    
    # Create batches based on count and token limits
    batches = []
    current_batch = []
    current_batch_tokens = 0
    estimated_tokens_per_text = [len(text.split()) * 1.3 for text in texts]  # Rough estimate
    
    for text, est_tokens in zip(texts, estimated_tokens_per_text):
        # If adding this text would exceed batch size or token limit, start a new batch
        if (len(current_batch) >= config.batch_size or
                current_batch_tokens + est_tokens > config.max_tokens_per_batch):
            if current_batch:  # Only add non-empty batches
                batches.append(current_batch)
            current_batch = []
            current_batch_tokens = 0
        
        current_batch.append(text)
        current_batch_tokens += est_tokens
    
    # Add the last batch if not empty
    if current_batch:
        batches.append(current_batch)
    
    # Process each batch
    all_embeddings = []
    total_batches = len(batches)
    
    for i, batch in enumerate(batches):
        try:
            logger.info(f"Processing batch {i+1}/{total_batches} with {len(batch)} texts")
            
            # Get embeddings for the batch
            response = config.client.embeddings.create(
                model=config.model_name,
                input=batch
            )
            
            # Extract embeddings from response
            batch_embeddings = [data.embedding for data in response.data]
            all_embeddings.extend(batch_embeddings)
            
            # Throttle requests to avoid rate limits
            if i < total_batches - 1:  # No need to wait after the last batch
                time.sleep(0.5)  # 500ms pause between batches
        
        except Exception as e:
            logger.error(f"Error processing batch {i+1}: {e}")
            
            # Fall back to individual processing if batch fails
            logger.info(f"Falling back to individual processing for batch {i+1}")
            for text in batch:
                try:
                    embedding = get_embedding(text, config)
                    all_embeddings.append(embedding)
                    # Brief pause between individual requests
                    time.sleep(0.1)
                except Exception as text_e:
                    # If individual embedding fails, add a zero vector as placeholder
                    logger.error(f"Failed to embed text: {text_e}")
                    all_embeddings.append([0.0] * config.dimensions)
    
    return all_embeddings


def calculate_embedding_cost(texts: List[str], model: str = EMBEDDING_MODEL) -> float:
    """
    Calculate the approximate cost of generating embeddings.
    Based on OpenAI's pricing (subject to change).
    
    Args:
        texts: List of texts to embed
        model: Embedding model name
        
    Returns:
        Estimated cost in USD
    """
    # Rough token count estimation
    total_tokens = sum(len(text.split()) * 1.3 for text in texts)
    
    # Pricing as of May 2025 (adjust as needed)
    if model == "text-embedding-3-small":
        cost_per_1k_tokens = 0.00002  # $0.02 per million tokens
    elif model == "text-embedding-3-large":
        cost_per_1k_tokens = 0.00013  # $0.13 per million tokens
    else:
        # Default to small model pricing
        cost_per_1k_tokens = 0.00002
    
    # Calculate cost
    estimated_cost = (total_tokens / 1000) * cost_per_1k_tokens
    
    return estimated_cost


def process_chunked_data(
    input_dir: str,
    output_dir: str,
    config: Optional[EmbeddingConfig] = None
) -> Dict[str, Any]:
    """
    Process chunked data to generate embeddings.
    
    Args:
        input_dir: Directory containing chunked data
        output_dir: Directory to save processed data with embeddings
        config: Optional embedding configuration
        
    Returns:
        Dictionary with processing statistics
    """
    # Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Use default config if none provided
    if config is None:
        config = EmbeddingConfig()
    
    # Load chunks
    chunk_files = list(Path(input_dir).glob("*.json"))
    chunk_files = [f for f in chunk_files if f.name not in ["chunks_index.json", "chunking_stats.json"]]
    
    logger.info(f"Found {len(chunk_files)} chunks to process")
    
    # Initialize statistics
    stats = {
        "total_chunks": len(chunk_files),
        "processed_chunks": 0,
        "embedding_model": config.model_name,
        "embedding_dimensions": config.dimensions,
        "batch_size": config.batch_size,
        "processing_time": 0
    }
    
    # Load all chunks
    chunks = []
    chunk_ids = []
    texts = []
    
    for chunk_file in chunk_files:
        try:
            with open(chunk_file, 'r', encoding='utf-8') as f:
                chunk = json.load(f)
            
            chunks.append(chunk)
            chunk_ids.append(chunk["chunk_id"])
            texts.append(chunk["text"])
        
        except Exception as e:
            logger.error(f"Error loading chunk {chunk_file}: {e}")
    
    # Calculate estimated cost
    estimated_cost = calculate_embedding_cost(texts, config.model_name)
    logger.info(f"Estimated cost for generating embeddings: ${estimated_cost:.4f}")
    
    # Generate embeddings in batches
    start_time = time.time()
    embeddings = batch_embed_texts(texts, config)
    end_time = time.time()
    processing_time = end_time - start_time
    
    stats["processing_time"] = processing_time
    logger.info(f"Generated {len(embeddings)} embeddings in {processing_time:.2f} seconds")
    
    # Save chunks with embeddings
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        chunk["embedding"] = embedding
        
        # Save to output directory
        output_file = os.path.join(output_dir, f"{chunk['chunk_id']}.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(chunk, f, ensure_ascii=False, indent=2)
        
        stats["processed_chunks"] += 1
        
        # Log progress
        if (i + 1) % 100 == 0 or i + 1 == len(chunks):
            logger.info(f"Processed {i + 1}/{len(chunks)} chunks")
    
    # Save index file
    index = {}
    for chunk in chunks:
        index[chunk["chunk_id"]] = {
            "source_id": chunk["metadata"]["source_id"],
            "source_title": chunk["metadata"]["source_title"],
            "token_count": chunk["metadata"]["token_count"],
            "header_path": chunk["metadata"]["header_path"]
        }
    
    index_file = os.path.join(output_dir, "embeddings_index.json")
    with open(index_file, 'w', encoding='utf-8') as f:
        json.dump(index, f, ensure_ascii=False, indent=2)
    
    # Save statistics
    stats_file = os.path.join(output_dir, "embedding_stats.json")
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Embedding generation complete. Processed {stats['processed_chunks']} chunks.")
    logger.info(f"Results saved to {output_dir}")
    
    return stats