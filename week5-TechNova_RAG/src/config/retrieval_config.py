# src/config/retrieval_config.py

"""
Configuration settings for vector database retrieval and RAG engine.
"""

DEFAULT_RETRIEVAL_CONFIG = {
    # Vector database retrieval parameters
    "num_chunks": 7,                 # Number of chunks to retrieve - 5
    "similarity_threshold": 0.5,     # Minimum similarity score to consider - 0.7
    "mmr_lambda": 0.5,               # Diversity vs. similarity balance for MMR (0=max diversity, 1=max similarity)
    
    # Metadata filtering options
    "enable_metadata_filtering": True,
    "default_filters": {},           # Default metadata filters to apply
    
    # Re-ranking parameters
    "enable_reranking": True,        # Whether to apply re-ranking to results
    "reranking_fetch_multiplier": 3, # Fetch this many more results than needed for re-ranking
    
    # OpenAI API parameters
    "openai_model": "gpt-4o",        # Model to use for generation
    "openai_temperature": 0.1,       # Lower temperature for more factual responses
    "openai_max_tokens": 1000,       # Maximum tokens in response
}