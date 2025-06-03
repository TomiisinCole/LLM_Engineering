"""
Chunking module for processing Notion content.
Implements a header-based chunking strategy with overlap.
"""
import json
import re
import os
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
import tiktoken
from loguru import logger


class ChunkingConfig:
    """Configuration for chunking parameters."""
    
    def __init__(
        self,
        target_chunk_size: int = 400,
        overlap_percentage: float = 0.1,
        min_chunk_size: int = 40, #can be higher
        max_chunk_size: int = 800,
        encoding_name: str = "cl100k_base"  # OpenAI's embedding model encoding
    ):
        """
        Initialize chunking configuration.
        
        Args:
            target_chunk_size: Target size of chunks in tokens
            overlap_percentage: Percentage of overlap between chunks (0-1)
            min_chunk_size: Minimum chunk size in tokens
            max_chunk_size: Maximum chunk size in tokens
            encoding_name: Tokenizer encoding name for token counting
        """
        self.target_chunk_size = target_chunk_size
        self.overlap_percentage = overlap_percentage
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.encoding_name = encoding_name
        
        # Initialize tokenizer for token counting
        self.tokenizer = tiktoken.get_encoding(encoding_name)
    
    def calculate_overlap_tokens(self) -> int:
        """Calculate the number of tokens to overlap between chunks."""
        return max(10, int(self.target_chunk_size * self.overlap_percentage))


def count_tokens(text: str, encoding_name: str = "cl100k_base") -> int:
    """
    Count the number of tokens in the text.
    
    Args:
        text: The text to count tokens for
        encoding_name: The name of the encoding to use
        
    Returns:
        The number of tokens in the text
    """
    encoding = tiktoken.get_encoding(encoding_name)
    return len(encoding.encode(text))


def chunk_by_headers(
    content: List[Dict[str, Any]],
    config: ChunkingConfig
) -> List[Dict[str, Any]]:
    """
    Chunk content based on header hierarchy.
    
    Args:
        content: List of content blocks from a Notion page
        config: Chunking configuration
        
    Returns:
        List of chunks with metadata
    """
    chunks = []
    current_chunk = []
    current_headers = []
    current_token_count = 0
    
    # Track the header levels we've seen
    header_levels = {f"heading_{i}": i for i in range(1, 7)}
    
    for block in content:
        block_type = block.get("type", "")
        text = block.get("text", "")
        
        # Count tokens in this block
        block_tokens = count_tokens(text, config.encoding_name)
        
        # Check if this is a header
        if block_type in header_levels:
            # If we have a chunk in progress, add it to chunks
            if current_chunk and current_token_count >= config.min_chunk_size:
                chunks.append({
                    "content": current_chunk.copy(),
                    "headers": current_headers.copy(),
                    "token_count": current_token_count
                })
            
            # Start a new chunk with this header
            current_chunk = [block]
            
            # Update headers based on level
            header_level = header_levels[block_type]
            
            # Truncate header path to current level
            current_headers = [h for h in current_headers if h["level"] < header_level]
            current_headers.append({
                "level": header_level,
                "text": text,
                "type": block_type
            })
            
            current_token_count = block_tokens
        else:
            # Check if adding this block would exceed max chunk size
            if current_token_count + block_tokens > config.max_chunk_size and current_chunk:
                # Save current chunk
                chunks.append({
                    "content": current_chunk.copy(),
                    "headers": current_headers.copy(),
                    "token_count": current_token_count
                })
                
                # Start new chunk with overlap from headers
                current_chunk = []
                current_token_count = 0
                
                # Add headers to the new chunk for context
                for header in current_headers:
                    if "text" in header:
                        current_token_count += count_tokens(header["text"], config.encoding_name)
            
            # Add block to current chunk
            current_chunk.append(block)
            current_token_count += block_tokens
    
    # Add the last chunk if it's not empty
    if current_chunk and current_token_count >= config.min_chunk_size:
        chunks.append({
            "content": current_chunk,
            "headers": current_headers,
            "token_count": current_token_count
        })
    
    return chunks


def chunk_with_overlap(
    content: List[Dict[str, Any]],
    config: ChunkingConfig
) -> List[Dict[str, Any]]:
    """
    Create overlapping chunks of text from content blocks.
    Used for sections that don't have headers but need to be chunked.
    
    Args:
        content: List of content blocks
        config: Chunking configuration
        
    Returns:
        List of overlapping chunks
    """
    chunks = []
    blocks = []
    current_tokens = 0
    overlap_tokens = config.calculate_overlap_tokens()
    
    # Combine all text from blocks
    all_text = []
    for block in content:
        text = block.get("text", "")
        if text.strip():
            all_text.append(text)
    
    # Combine into one text
    full_text = "\n".join(all_text)
    
    # If text is short enough, return as single chunk
    if count_tokens(full_text, config.encoding_name) <= config.max_chunk_size:
        return [{
            "content": content,
            "headers": [],
            "token_count": count_tokens(full_text, config.encoding_name)
        }]
    
    # Otherwise, chunk with overlap
    current_start = 0
    tokenizer = config.tokenizer
    all_tokens = tokenizer.encode(full_text)
    
    while current_start < len(all_tokens):
        # Calculate end position
        current_end = min(current_start + config.target_chunk_size, len(all_tokens))
        
        # Extract chunk tokens
        chunk_tokens = all_tokens[current_start:current_end]
        chunk_text = tokenizer.decode(chunk_tokens)
        
        # Create chunk with the text
        chunk = {
            "content": [{"type": "paragraph", "text": chunk_text}],
            "headers": [],
            "token_count": len(chunk_tokens)
        }
        
        chunks.append(chunk)
        
        # Move start position for next chunk, including overlap
        current_start += config.target_chunk_size - overlap_tokens
    
    return chunks


def create_chunk_with_metadata(
    chunk: Dict[str, Any], 
    page_metadata: Dict[str, Any],
    chunk_index: int
) -> Dict[str, Any]:
    """
    Create a chunk with metadata from the source document.
    
    Args:
        chunk: The chunk content
        page_metadata: Metadata from the source page
        chunk_index: Index of this chunk in the sequence
        
    Returns:
        Chunk with complete metadata
    """
    # Extract header path as a string
    header_path = []
    for header in chunk.get("headers", []):
        if "text" in header:
            header_path.append(header["text"])
    
    # Create the chunk with metadata
    return {
        "chunk_id": f"{page_metadata['id']}-chunk_{chunk_index:03d}",
        "text": extract_text_from_chunk(chunk),
        "metadata": {
            "source_id": page_metadata.get("id", ""),
            "source_title": page_metadata.get("title", ""),
            "source_url": page_metadata.get("url", ""),
            "created_time": page_metadata.get("created_time", ""),
            "last_edited_time": page_metadata.get("last_edited_time", ""),
            "chunk_index": chunk_index,
            "header_path": " > ".join(header_path) if header_path else "",
            "token_count": chunk.get("token_count", 0)
        },
        "content_blocks": chunk.get("content", [])
    }


def extract_text_from_chunk(chunk: Dict[str, Any]) -> str:
    """
    Extract plain text from a chunk for embedding.
    
    Args:
        chunk: The chunk with content blocks
        
    Returns:
        Plain text representation of the chunk
    """
    texts = []
    
    # Add header path as context
    header_path = []
    for header in chunk.get("headers", []):
        if "text" in header:
            header_path.append(header["text"])
    
    if header_path:
        texts.append(" > ".join(header_path))
    
    # Add content from each block
    for block in chunk.get("content", []):
        text = block.get("text", "").strip()
        if text:
            texts.append(text)
    
    return "\n".join(texts)


def process_page_content(
    page_data: Dict[str, Any],
    config: ChunkingConfig
) -> List[Dict[str, Any]]:
    """
    Process a page's content into chunks with metadata.
    
    Args:
        page_data: The page data from Notion with content blocks
        config: Chunking configuration
        
    Returns:
        List of chunks with metadata
    """
    content_blocks = page_data.get("content", [])
    page_metadata = {
        "id": page_data.get("id", ""),
        "title": page_data.get("title", ""),
        "url": page_data.get("url", ""),
        "created_time": page_data.get("created_time", ""),
        "last_edited_time": page_data.get("last_edited_time", "")
    }
    
    # Check if content exists
    if not content_blocks:
        logger.warning(f"No content blocks found for page {page_metadata['title']}")
        return []
    
    # First, chunk by headers
    header_chunks = chunk_by_headers(content_blocks, config)
    
    # For any chunks that are too large, apply secondary chunking with overlap
    final_chunks = []
    for chunk in header_chunks:
        if chunk["token_count"] > config.max_chunk_size:
            # Apply secondary chunking
            sub_chunks = chunk_with_overlap(chunk["content"], config)
            for sub_chunk in sub_chunks:
                # Copy header path from parent chunk
                sub_chunk["headers"] = chunk["headers"]
                final_chunks.append(sub_chunk)
        else:
            final_chunks.append(chunk)
    
    # FALLBACK: If no chunks were created but content exists, create a single chunk with all content
    if not final_chunks and content_blocks:
        # Count total tokens in all content blocks
        total_tokens = 0
        for block in content_blocks:
            text = block.get("text", "")
            if text:
                total_tokens += count_tokens(text, config.encoding_name)
        
        # Create a single chunk with all content, regardless of min_chunk_size
        fallback_chunk = {
            "content": content_blocks,
            "headers": [],
            "token_count": total_tokens
        }
        final_chunks.append(fallback_chunk)
        logger.info(f"Created fallback chunk for page '{page_metadata['title']}' with {total_tokens} tokens")
    
    # Create chunks with metadata
    chunks_with_metadata = []
    for i, chunk in enumerate(final_chunks):
        chunk_with_metadata = create_chunk_with_metadata(chunk, page_metadata, i + 1)
        chunks_with_metadata.append(chunk_with_metadata)
    
    logger.info(f"Processed page '{page_metadata['title']}' into {len(chunks_with_metadata)} chunks")
    
    return chunks_with_metadata


def chunk_all_pages(
    input_dir: str = "./data/raw",
    output_dir: str = "./data/chunked",
    config: Optional[ChunkingConfig] = None
) -> Dict[str, Any]:
    """
    Process all extracted pages into chunks.
    
    Args:
        input_dir: Directory containing extracted pages
        output_dir: Directory to save chunked output
        config: Optional chunking configuration
        
    Returns:
        Dictionary with chunking statistics
    """
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Use default config if none provided
    if config is None:
        config = ChunkingConfig()
    
    # Get list of all page files
    page_files = list(Path(input_dir).glob("*.json"))
    page_files = [f for f in page_files if f.name != "index.json"]
    
    # Initialize statistics
    stats = {
        "total_pages": len(page_files),
        "total_chunks": 0,
        "pages_processed": 0,
        "chunks_by_page": {},
        "avg_chunk_size": 0,
        "config": {
            "target_chunk_size": config.target_chunk_size,
            "overlap_percentage": config.overlap_percentage,
            "min_chunk_size": config.min_chunk_size,
            "max_chunk_size": config.max_chunk_size
        }
    }
    
    # Process each page
    all_chunks = []
    total_token_count = 0
    
    for page_file in page_files:
        try:
            # Load page data
            with open(page_file, 'r', encoding='utf-8') as f:
                page_data = json.load(f)
            
            # Process page
            page_chunks = process_page_content(page_data, config)
            
            # Update statistics
            stats["pages_processed"] += 1
            stats["total_chunks"] += len(page_chunks)
            stats["chunks_by_page"][page_data.get("title", page_file.stem)] = len(page_chunks)
            
            # Track tokens
            for chunk in page_chunks:
                total_token_count += chunk["metadata"]["token_count"]
            
            # Add to all chunks
            all_chunks.extend(page_chunks)
            
            # Log progress
            logger.info(f"Processed {stats['pages_processed']}/{stats['total_pages']} pages")
            
        except Exception as e:
            logger.error(f"Error processing page {page_file}: {e}")
    
    # Calculate average chunk size
    if stats["total_chunks"] > 0:
        stats["avg_chunk_size"] = total_token_count / stats["total_chunks"]
    
    # Save all chunks to output directory
    chunks_index = {}
    
    for chunk in all_chunks:
        chunk_id = chunk["chunk_id"]
        output_file = os.path.join(output_dir, f"{chunk_id}.json")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(chunk, f, ensure_ascii=False, indent=2)
        
        # Add to index
        chunks_index[chunk_id] = {
            "source_id": chunk["metadata"]["source_id"],
            "source_title": chunk["metadata"]["source_title"],
            "token_count": chunk["metadata"]["token_count"],
            "header_path": chunk["metadata"]["header_path"],
        }
    
    # Save index
    index_file = os.path.join(output_dir, "chunks_index.json")
    with open(index_file, 'w', encoding='utf-8') as f:
        json.dump(chunks_index, f, ensure_ascii=False, indent=2)
    
    # Save statistics
    stats_file = os.path.join(output_dir, "chunking_stats.json")
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Chunking complete. Processed {stats['pages_processed']} pages into {stats['total_chunks']} chunks.")
    logger.info(f"Average chunk size: {stats['avg_chunk_size']:.2f} tokens")
    
    return stats