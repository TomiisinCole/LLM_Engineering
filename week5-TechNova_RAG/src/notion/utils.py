"""
Utility functions for Notion integration.
"""
import os
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from loguru import logger


def setup_logging(log_level: str = None):
    """
    Set up logging configuration.
    
    Args:
        log_level: Log level (DEBUG, INFO, WARNING, ERROR)
    """
    if not log_level:
        log_level = os.getenv("LOG_LEVEL", "INFO")
    
    # Configure Loguru logger
    logger.remove()  # Remove default handler
    logger.add(
        "logs/technova_rag_{time}.log",
        rotation="10 MB",
        level=log_level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
    )
    
    # Also log to console
    logger.add(
        lambda msg: print(msg, end=""),
        level=log_level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
    )
    
    logger.info(f"Logging initialized at level {log_level}")


def create_project_directories():
    """
    Create necessary project directories if they don't exist.
    """
    directories = [
        "./data",
        "./data/raw",
        "./data/processed",
        "./logs",
        "./src",
        "./src/notion",
        "./src/processing",
        "./src/database",
        "./src/rag",
        "./src/ui"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        
    logger.info("Project directories created")


def load_extracted_pages(data_dir: str = "./data/raw") -> List[Dict[str, Any]]:
    """
    Load previously extracted pages from the data directory.
    
    Args:
        data_dir: Directory containing extracted page data
        
    Returns:
        List of page objects
    """
    # Check if index file exists
    index_path = Path(data_dir) / "index.json"
    if not index_path.exists():
        logger.warning(f"Index file not found at {index_path}")
        return []
    
    try:
        # Load index
        with open(index_path, 'r', encoding='utf-8') as f:
            index = json.load(f)
        
        # Load each page
        pages = []
        for page_meta in index:
            page_id = page_meta["id"]
            page_path = Path(data_dir) / f"{page_id}.json"
            
            if page_path.exists():
                with open(page_path, 'r', encoding='utf-8') as f:
                    page_data = json.load(f)
                    pages.append(page_data)
            else:
                logger.warning(f"Page file not found at {page_path}")
        
        logger.info(f"Loaded {len(pages)} pages from {data_dir}")
        return pages
    
    except Exception as e:
        logger.error(f"Error loading extracted pages: {e}")
        return []


def get_extraction_stats(data_dir: str = "./data/raw") -> Dict[str, Any]:
    """
    Get statistics about extracted pages.
    
    Args:
        data_dir: Directory containing extracted page data
        
    Returns:
        Dictionary with extraction statistics
    """
    pages = load_extracted_pages(data_dir)
    
    if not pages:
        return {
            "total_pages": 0,
            "total_blocks": 0,
            "extraction_date": None
        }
    
    # Count blocks
    total_blocks = sum(len(page.get("content", [])) for page in pages)
    
    # Get the latest extraction date
    latest_file = max(Path(data_dir).glob("*.json"), key=lambda p: p.stat().st_mtime)
    extraction_date = datetime.fromtimestamp(latest_file.stat().st_mtime)
    
    return {
        "total_pages": len(pages),
        "total_blocks": total_blocks,
        "extraction_date": extraction_date.isoformat()
    }


def print_extraction_summary(data_dir: str = "./data/raw"):
    """
    Print a summary of the extracted pages.
    
    Args:
        data_dir: Directory containing extracted page data
    """
    stats = get_extraction_stats(data_dir)
    
    if stats["total_pages"] == 0:
        logger.info("No pages have been extracted yet.")
        return
    
    logger.info(f"=== Extraction Summary ===")
    logger.info(f"Total pages: {stats['total_pages']}")
    logger.info(f"Total blocks: {stats['total_blocks']}")
    logger.info(f"Last extraction: {stats['extraction_date']}")
    
    # Load index for more detailed information
    index_path = Path(data_dir) / "index.json"
    if index_path.exists():
        with open(index_path, 'r', encoding='utf-8') as f:
            index = json.load(f)
        
        logger.info("\nExtracted pages:")
        for i, page in enumerate(index, 1):
            logger.info(f"{i}. {page['title']} - {page['url']}")