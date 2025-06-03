"""
Text cleaning and normalization module.
Implements functions for standardizing Notion content.
"""
import re
import html
from typing import List, Dict, Any, Optional
from loguru import logger


def remove_extra_whitespace(text: str) -> str:
    """
    Remove extra whitespace, including duplicate spaces and newlines.
    
    Args:
        text: Input text
        
    Returns:
        Text with normalized whitespace
    """
    # Replace multiple spaces with a single space
    text = re.sub(r' +', ' ', text)
    
    # Replace multiple newlines with a maximum of two
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Trim whitespace at the beginning and end
    return text.strip()


def standardize_newlines(text: str) -> str:
    """
    Standardize newlines across different platforms.
    
    Args:
        text: Input text
        
    Returns:
        Text with standardized newlines
    """
    # Convert all types of newlines to \n
    text = text.replace('\r\n', '\n')
    text = text.replace('\r', '\n')
    
    return text


def strip_special_characters(text: str, keep_punctuation: bool = True) -> str:
    """
    Remove or replace special characters that might cause issues.
    
    Args:
        text: Input text
        keep_punctuation: Whether to keep standard punctuation
        
    Returns:
        Cleaned text
    """
    # Decode HTML entities
    text = html.unescape(text)
    
    if keep_punctuation:
        # Keep alphanumeric chars, spaces, and common punctuation
        text = re.sub(r'[^\w\s.,;:!?()[\]{}\'\""-]', ' ', text)
    else:
        # Keep only alphanumeric chars and spaces
        text = re.sub(r'[^\w\s]', ' ', text)
    
    return text


def normalize_text(text: str, aggressive: bool = False) -> str:
    """
    Apply a series of normalization steps to the text.
    
    Args:
        text: Input text
        aggressive: Whether to apply more aggressive normalization
        
    Returns:
        Normalized text
    """
    # Apply standard normalization
    text = standardize_newlines(text)
    text = strip_special_characters(text, keep_punctuation=not aggressive)
    text = remove_extra_whitespace(text)
    
    # If aggressive, also lowercase the text
    if aggressive:
        text = text.lower()
    
    return text


def clean_notion_block(block: Dict[str, Any]) -> Dict[str, Any]:
    """
    Clean and normalize text in a Notion block.
    
    Args:
        block: Notion content block
        
    Returns:
        Cleaned block
    """
    # Create a copy of the block
    cleaned_block = block.copy()
    
    # Clean text field if exists
    if "text" in cleaned_block and cleaned_block["text"]:
        cleaned_block["text"] = normalize_text(cleaned_block["text"])
    
    return cleaned_block


def clean_notion_blocks(blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Clean and normalize a list of Notion blocks.
    
    Args:
        blocks: List of Notion content blocks
        
    Returns:
        List of cleaned blocks
    """
    return [clean_notion_block(block) for block in blocks]


def clean_notion_page(page_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Clean and normalize content in a Notion page.
    
    Args:
        page_data: Notion page data
        
    Returns:
        Cleaned page data
    """
    # Create a copy of the page data
    cleaned_page = page_data.copy()
    
    # Clean content blocks if they exist
    if "content" in cleaned_page and cleaned_page["content"]:
        cleaned_page["content"] = clean_notion_blocks(cleaned_page["content"])
    
    return cleaned_page


def clean_text_for_embedding(text: str) -> str:
    """
    Clean and prepare text specifically for embedding.
    This might involve more aggressive normalization.
    
    Args:
        text: Input text
        
    Returns:
        Cleaned text ready for embedding
    """
    # Standard cleaning
    text = normalize_text(text)
    
    # Additional cleaning steps for embedding
    # Replace multiple spaces with single space
    text = re.sub(r' +', ' ', text)
    
    # Remove URLs
    text = re.sub(r'https?://\S+', '', text)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+\.\S+', '', text)
    
    # Remove non-printable characters
    text = ''.join(c for c in text if c.isprintable() or c.isspace())
    
    return text.strip()


def format_headers_for_display(header_path: List[Dict[str, Any]]) -> str:
    """
    Format a header path for display.
    
    Args:
        header_path: List of header objects
        
    Returns:
        Formatted header path string
    """
    if not header_path:
        return ""
    
    headers = []
    for header in header_path:
        text = header.get("text", "")
        if text:
            headers.append(text)
    
    return " > ".join(headers)


def process_page_content_for_display(
    page_data: Dict[str, Any],
    include_metadata: bool = True
) -> str:
    """
    Process page content into a clean, readable format for display.
    
    Args:
        page_data: Notion page data
        include_metadata: Whether to include page metadata
        
    Returns:
        Processed text
    """
    result = []
    
    # Add metadata if requested
    if include_metadata:
        title = page_data.get("title", "Untitled")
        result.append(f"# {title}")
        result.append("")
        
        url = page_data.get("url", "")
        if url:
            result.append(f"Source: {url}")
            result.append("")
    
    # Process content blocks
    content = page_data.get("content", [])
    current_level = 0
    
    for block in content:
        block_type = block.get("type", "")
        text = block.get("text", "")
        level = block.get("level", 0)
        
        if not text.strip():
            continue
        
        # Format based on block type
        if block_type.startswith("heading_"):
            heading_level = int(block_type[-1])
            result.append(f"{'#' * heading_level} {text}")
            result.append("")
        
        elif block_type == "paragraph":
            result.append(text)
            result.append("")
        
        elif block_type == "bulleted_list_item":
            result.append(f"• {text}")
        
        elif block_type == "numbered_list_item":
            result.append(f"1. {text}")
        
        elif block_type == "to_do":
            checked = block.get("checked", False)
            result.append(f"{'[x]' if checked else '[ ]'} {text}")
        
        elif block_type == "code":
            language = block.get("language", "")
            result.append(f"```{language}")
            result.append(text)
            result.append("```")
            result.append("")
        
        elif block_type == "quote":
            result.append(f"> {text}")
            result.append("")
        
        elif block_type == "callout":
            result.append(f"ℹ️ {text}")
            result.append("")
        
        elif block_type == "divider":
            result.append("---")
            result.append("")
        
        else:
            result.append(text)
            result.append("")
    
    return "\n".join(result)