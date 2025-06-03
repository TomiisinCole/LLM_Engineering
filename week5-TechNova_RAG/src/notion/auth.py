"""
Notion authentication module.
Handles authentication with Notion API and client initialization.
"""
import os
from dotenv import load_dotenv
from notion_client import Client
from loguru import logger


def load_env_variables():
    """Load environment variables from .env file."""
    load_dotenv()
    
    # Check for required environment variables
    required_vars = ['NOTION_API_KEY', 'NOTION_WORKSPACE_ID']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
        logger.error("Please check your .env file and add the missing variables.")
        raise EnvironmentError(f"Missing required environment variables: {', '.join(missing_vars)}")
    
    return {
        'notion_api_key': os.getenv('NOTION_API_KEY'),
        'notion_workspace_id': os.getenv('NOTION_WORKSPACE_ID')
    }


def get_notion_client():
    """
    Initialize and return a Notion client using API key from environment variables.
    
    Returns:
        notion_client.Client: Authenticated Notion client instance
    
    Raises:
        EnvironmentError: If required environment variables are missing
    """
    env_vars = load_env_variables()
    
    try:
        # Initialize Notion client
        notion = Client(auth=env_vars['notion_api_key'])
        
        # Test the connection
        notion.users.me()
        logger.info("Successfully connected to Notion API")
        
        return notion
    
    except Exception as e:
        logger.error(f"Failed to initialize Notion client: {e}")
        raise


def validate_workspace_access(notion_client):
    """
    Validate that the provided API key has access to the specified workspace.
    
    Args:
        notion_client (notion_client.Client): Authenticated Notion client
    
    Returns:
        bool: True if access is valid, raises exception otherwise
    
    Raises:
        PermissionError: If API key doesn't have access to the workspace
    """
    env_vars = load_env_variables()
    workspace_id = env_vars['notion_workspace_id']
    
    try:
        # Test access to the workspace by getting a list of pages
        # This will fail if the integration doesn't have access
        notion_client.search(
            filter={"property": "object", "value": "page"},
            page_size=1  # Just check one page to validate access
        )
        logger.info(f"Successfully validated access to workspace {workspace_id}")
        return True
    
    except Exception as e:
        logger.error(f"Failed to access workspace {workspace_id}: {e}")
        logger.error("Ensure the integration has been added to the workspace in Notion settings")
        raise PermissionError(f"Cannot access workspace {workspace_id}. Check integration permissions.")