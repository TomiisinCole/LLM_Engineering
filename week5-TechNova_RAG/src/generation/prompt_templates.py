# src/generation/prompt_templates.py

from string import Template
from typing import Dict, Any

class PromptTemplates:
    """
    Contains templates for RAG system prompts with context insertion.
    """
    
    def __init__(self):
        """Initialize the prompt templates."""
        # System message template
        self.system_template = Template("""
You are a knowledgeable assistant for TechNova. Your role is to provide accurate information about the company based ONLY on the context provided below.

Guidelines:
1. Answer ONLY based on the information in the context.
2. If the context doesn't contain enough information to answer, say "I don't have enough information to answer this question."
3. Use concise, direct language while preserving important details.
4. When answering, cite your sources using [Source Name] notation.
5. Do not make up or infer information that isn't explicitly in the context.

CONTEXT:
$context

Remember to only use information from the provided context.
""")
        
        # User message template (simple passthrough)
        self.user_template = Template("$query")
        
        # Insufficient information template
        self.insufficient_info_template = Template("""
I don't have enough information in my knowledge base to answer your question about "$query". 

This topic might not be documented in our TechNova Notion workspace, or it might require more specific details. 

Would you like me to help you with something else related to TechNova?
""")
    
    def format_system_prompt(self, context: str) -> str:
        """
        Format the system prompt with the given context.
        
        Args:
            context: The assembled context from retrieved chunks
            
        Returns:
            Formatted system prompt
        """
        return self.system_template.substitute(context=context)
    
    def format_user_prompt(self, query: str) -> str:
        """
        Format the user prompt with the given query.
        
        Args:
            query: The user's question
            
        Returns:
            Formatted user prompt
        """
        return self.user_template.substitute(query=query)
    
    def format_insufficient_info(self, query: str) -> str:
        """
        Format the insufficient information response.
        
        Args:
            query: The user's question
            
        Returns:
            Formatted insufficient info response
        """
        return self.insufficient_info_template.substitute(query=query)