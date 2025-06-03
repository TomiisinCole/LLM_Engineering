"""
Notion content extractor module.
Handles listing and extracting content from Notion pages.
"""
import os
import json
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from tqdm import tqdm
from loguru import logger

from .auth import get_notion_client, load_env_variables


def list_all_pages(notion_client, workspace_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    List all pages in the workspace that the integration has access to.
    
    Args:
        notion_client: The authenticated Notion client
        workspace_id: Optional workspace ID to filter results
    
    Returns:
        List of page objects with id, title, and other metadata
    """
    if not workspace_id:
        env_vars = load_env_variables()
        workspace_id = env_vars['notion_workspace_id']
    
    logger.info(f"Listing all pages in workspace {workspace_id}")
    
    # Start with an empty list to collect all pages
    all_pages = []
    
    # Notion API uses pagination - we need to keep querying until we get all pages
    has_more = True
    start_cursor = None
    
    while has_more:
        search_params = {
            "filter": {"property": "object", "value": "page"},
            "page_size": 100,  # Maximum allowed by API
        }
        
        if start_cursor:
            search_params["start_cursor"] = start_cursor
        
        response = notion_client.search(**search_params)
        
        # Extract page metadata
        for page in response.get("results", []):
            page_metadata = extract_page_metadata(page)
            if page_metadata:
                all_pages.append(page_metadata)
        
        # Check if there are more pages to fetch
        has_more = response.get("has_more", False)
        if has_more:
            start_cursor = response.get("next_cursor")
    
    logger.info(f"Found {len(all_pages)} pages in workspace")
    return all_pages


def extract_page_metadata(page: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract metadata from a Notion page object.
    
    Args:
        page: The Notion page object
        
    Returns:
        Dictionary containing page metadata
    """
    if page.get("object") != "page":
        return None
    
    # Extract page ID
    page_id = page.get("id")
    
    # Extract page title
    title = ""
    properties = page.get("properties", {})
    
    # Try to get title from properties
    title_property = properties.get("title") or properties.get("Title") or properties.get("Name")
    if title_property and "title" in title_property:
        title_parts = title_property["title"]
        title = "".join([part.get("plain_text", "") for part in title_parts])
    
    # Extract timestamps
    created_time = page.get("created_time")
    last_edited_time = page.get("last_edited_time")
    
    # Extract URL
    url = page.get("url", "")
    
    return {
        "id": page_id,
        "title": title,
        "created_time": created_time,
        "last_edited_time": last_edited_time,
        "url": url
    }


def extract_page_content(notion_client, page_id: str) -> Dict[str, Any]:
    """
    Extract the full content of a Notion page, including all blocks.
    
    Args:
        notion_client: The authenticated Notion client
        page_id: The ID of the page to extract
        
    Returns:
        Dictionary containing page content
    """
    logger.info(f"Extracting content from page {page_id}")
    
    try:
        # Get page metadata
        page = notion_client.pages.retrieve(page_id)
        page_metadata = extract_page_metadata(page)
        
        if not page_metadata:
            logger.warning(f"Could not extract metadata for page {page_id}")
            return None
        
        # Get page blocks - FIX: Make sure to pass the block_id parameter correctly
        blocks = []
        has_more = True
        start_cursor = None
        
        while has_more:
            # FIX: The correct parameter name is "block_id", not "page_id"
            response = notion_client.blocks.children.list(
                block_id=page_id,  # This is the fix - using block_id instead of page_id
                start_cursor=start_cursor if start_cursor else None
            )
            
            blocks.extend(response.get("results", []))
            
            has_more = response.get("has_more", False)
            if has_more:
                start_cursor = response.get("next_cursor")
        
        # Process blocks to extract text content
        content = process_blocks(notion_client, blocks)
        
        # Combine metadata and content
        return {
            **page_metadata,
            "content": content
        }
    
    except Exception as e:
        logger.error(f"Error extracting content from page {page_id}: {e}")
        return None


def process_blocks(notion_client, blocks, level=0) -> List[Dict[str, Any]]:
    """
    Process Notion blocks recursively to extract structured content.
    
    Args:
        notion_client: The authenticated Notion client
        blocks: List of Notion blocks to process
        level: Current nesting level (for headers and lists)
        
    Returns:
        List of processed block content
    """
    processed_blocks = []
    
    for block in blocks:
        block_id = block.get("id")
        block_type = block.get("type")
        
        if not block_type:
            continue
        
        # Get block content based on type
        content = {}
        content["type"] = block_type
        content["level"] = level
        
        # Extract text for different block types
        if block_type == "paragraph":
            rich_text = block.get("paragraph", {}).get("rich_text", [])
            content["text"] = "".join([text.get("plain_text", "") for text in rich_text])
        
        elif block_type.startswith("heading_"):
            heading_level = int(block_type[-1])
            rich_text = block.get(block_type, {}).get("rich_text", [])
            content["text"] = "".join([text.get("plain_text", "") for text in rich_text])
            content["heading_level"] = heading_level
        
        elif block_type == "bulleted_list_item" or block_type == "numbered_list_item":
            rich_text = block.get(block_type, {}).get("rich_text", [])
            content["text"] = "".join([text.get("plain_text", "") for text in rich_text])
        
        elif block_type == "to_do":
            rich_text = block.get("to_do", {}).get("rich_text", [])
            content["text"] = "".join([text.get("plain_text", "") for text in rich_text])
            content["checked"] = block.get("to_do", {}).get("checked", False)
        
        elif block_type == "toggle":
            rich_text = block.get("toggle", {}).get("rich_text", [])
            content["text"] = "".join([text.get("plain_text", "") for text in rich_text])
        
        elif block_type == "code":
            rich_text = block.get("code", {}).get("rich_text", [])
            content["text"] = "".join([text.get("plain_text", "") for text in rich_text])
            content["language"] = block.get("code", {}).get("language", "")
        
        elif block_type == "callout":
            rich_text = block.get("callout", {}).get("rich_text", [])
            content["text"] = "".join([text.get("plain_text", "") for text in rich_text])
            content["icon"] = block.get("callout", {}).get("icon", {})
        
        elif block_type == "quote":
            rich_text = block.get("quote", {}).get("rich_text", [])
            content["text"] = "".join([text.get("plain_text", "") for text in rich_text])
        
        elif block_type == "divider":
            content["text"] = "---"
        
        elif block_type == "table":
            content["text"] = "[Table]"  # Placeholder, we'll handle tables separately
        
        else:
            # Default for other block types
            content["text"] = f"[{block_type}]"
        
        processed_blocks.append(content)
        
        # Recursively process children if the block has them
        if block.get("has_children", False):
            # FIX: Use block_id as the parameter name here as well
            children = notion_client.blocks.children.list(block_id=block_id).get("results", [])
            child_blocks = process_blocks(notion_client, children, level + 1)
            processed_blocks.extend(child_blocks)
    
    return processed_blocks


def extract_all_pages(output_dir: str = "./data/raw"):
    """
    Extract all pages from the workspace and save them to the output directory.
    
    Args:
        output_dir: Directory to save extracted page content
    """
    # Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Get Notion client
    notion_client = get_notion_client()
    
    # List all pages
    all_pages = list_all_pages(notion_client)
    logger.info(f"Found {len(all_pages)} pages in the workspace")
    
    # Extract content from each page
    extracted_pages = []
    
    for page in tqdm(all_pages, desc="Extracting pages"):
        page_id = page["id"]
        page_content = extract_page_content(notion_client, page_id)
        
        if page_content:
            # Save page content to file
            output_file = os.path.join(output_dir, f"{page_id}.json")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(page_content, f, ensure_ascii=False, indent=2)
            
            extracted_pages.append(page_content)
            logger.info(f"Extracted page {page_content.get('title')} to {output_file}")
    
    # Save index of all pages
    index_file = os.path.join(output_dir, "index.json")
    with open(index_file, 'w', encoding='utf-8') as f:
        index = [{"id": page["id"], "title": page["title"], "url": page["url"]} for page in extracted_pages]
        json.dump(index, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Extracted {len(extracted_pages)} pages to {output_dir}")
    logger.info(f"Index saved to {index_file}")
    
    return extracted_pages