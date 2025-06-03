# src/retrieval/context_assembler.py - UPDATED

from typing import List, Dict, Any

class ContextAssembler:
    """
    Assembles retrieved chunks into a coherent context for the LLM,
    with proper formatting and source attribution.
    """
    
    def __init__(self, max_context_length: int = 4000):
        """
        Initialize the ContextAssembler.
        
        Args:
            max_context_length: Maximum token length for the assembled context
        """
        self.max_context_length = max_context_length
    
    def assemble_context(self, chunks: List[Dict[Any, Any]]) -> str:
        """
        Assemble retrieved chunks into a formatted context string.
        
        Args:
            chunks: List of chunks with text and metadata
            
        Returns:
            Formatted context string with source attribution
        """
        if not chunks:
            return "No relevant information found."
        
        # Sort chunks by similarity score (highest first)
        sorted_chunks = sorted(chunks, key=lambda x: x.get("similarity", 0), reverse=True)
        
        # Group chunks by source document
        chunks_by_source = {}
        for chunk in sorted_chunks:
            # Extract source information from metadata
            metadata = chunk.get("metadata", {})
            source_title = metadata.get("source_title", "Unknown Source")
            
            if source_title not in chunks_by_source:
                chunks_by_source[source_title] = []
            chunks_by_source[source_title].append(chunk)
        
        # Assemble the context with document headers
        context_parts = []
        
        for source_title, source_chunks in chunks_by_source.items():
            # Add document header
            source_header = f"--- {source_title} ---"
            context_parts.append(source_header)
            
            # Add each chunk with its section path if available
            for chunk in source_chunks:
                metadata = chunk.get("metadata", {})
                header_path = metadata.get("header_path", "")
                if header_path:
                    section_path = f"[Section: {header_path}]"
                    context_parts.append(section_path)
                
                # Add the chunk text
                context_parts.append(chunk.get("text", ""))
                context_parts.append("")  # Empty line for separation
        
        # Join everything with newlines
        context = "\n".join(context_parts)
        
        # Simple character-based approximation for context length limit
        if len(context) > self.max_context_length * 4:  # Rough approximation
            context = context[:self.max_context_length * 4] + "..."
        
        return context