# # src/rag_engine.py

# from typing import Dict, Any, Optional, List
# import os
# from dotenv import load_dotenv

# from src.database.vector_store import VectorStore
# from src.retrieval.query_processor import QueryProcessor
# from src.retrieval.context_assembler import ContextAssembler
# from src.generation.llm_generator import LLMGenerator

# class RAGEngine:
#     """
#     End-to-end RAG (Retrieval-Augmented Generation) engine that
#     combines retrieval and generation to answer questions.
#     """
    
#     def __init__(
#         self,
#         vector_store: Optional[VectorStore] = None,
#         query_processor: Optional[QueryProcessor] = None,
#         context_assembler: Optional[ContextAssembler] = None,
#         llm_generator: Optional[LLMGenerator] = None,
#         config: Optional[Dict[str, Any]] = None
#     ):
#         """
#         Initialize the RAG Engine.
        
#         Args:
#             vector_store: Vector database interface (created if None)
#             query_processor: Query processing component (created if None)
#             context_assembler: Context assembly component (created if None)
#             llm_generator: LLM generation component (created if None)
#             config: Configuration options
#         """
#         # Load environment variables
#         load_dotenv()
        
#         # Set up configuration
#         self.config = {
#             "retrieval_k": 20,  # Number of chunks to retrieve
#             "min_similarity": 0.15,  # Minimum similarity threshold
#             "max_context_length": 80000,  # Maximum context length 8000
#             "temperature": 0.2,  # LLM temperature
#             "model": "gpt-4-turbo",  # LLM model
#             "max_tokens": 14000,  # Maximum response length 4000
#         }
#         if config:
#             self.config.update(config)
        
#         # Initialize components
#         self.vector_store = vector_store or VectorStore()
#         self.query_processor = query_processor or QueryProcessor(self.vector_store)
#         self.context_assembler = context_assembler or ContextAssembler(
#             max_context_length=self.config["max_context_length"]
#         )
#         self.llm_generator = llm_generator or LLMGenerator(
#             model=self.config["model"],
#             temperature=self.config["temperature"],
#             max_tokens=self.config["max_tokens"]
#         )
    
#     def answer_question(
#         self,
#         query: str,
#         filter_metadata: Optional[Dict[str, Any]] = None,
#         stream: bool = False
#     ) -> Dict[str, Any]:
#         """
#         Process a user query and generate an answer with context.
        
#         Args:
#             query: The user's question
#             filter_metadata: Optional metadata filters
#             stream: Whether to stream the response
            
#         Returns:
#             Dict containing the answer, sources, and metadata
#         """
#         # Retrieve relevant chunks
#         chunks = self.query_processor.retrieve_relevant_chunks(
#             query=query,
#             k=self.config["retrieval_k"],
#             min_similarity=self.config["min_similarity"],
#             filter_metadata=filter_metadata
#         )
        
#         # Assemble context
#         context = self.context_assembler.assemble_context(chunks)
        
#         # Generate response
#         answer = self.llm_generator.generate_response(query, context, stream)
        
#         # Extract sources from chunks for citation
#         sources = []
#         for chunk in chunks:
#             metadata = chunk.get("metadata", {})
#             source = {
#                 "title": metadata.get("source_title", "Unknown Source"),
#                 "url": metadata.get("source_url", ""),
#                 "section": metadata.get("header_path", ""),
#                 "similarity": chunk.get("similarity", 0)
#             }
#             if source not in sources:
#                 sources.append(source)
        
#         # Return the complete result
#         return {
#             "query": query,
#             "answer": answer,
#             "sources": sources,
#             "context": context,
#             "chunks_retrieved": len(chunks)
#         }
    


# enhanced_rag_engine.py

from typing import Dict, Any, Optional, List
import os
import re
from dotenv import load_dotenv

from src.database.vector_store import VectorStore
from src.retrieval.query_processor import QueryProcessor
from src.retrieval.context_assembler import ContextAssembler
from src.generation.llm_generator import LLMGenerator

class EnhancedRAGEngine:
    """
    Enhanced RAG Engine with better defaults for comprehensive retrieval.
    """
    
    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        query_processor: Optional[QueryProcessor] = None,
        context_assembler: Optional[ContextAssembler] = None,
        llm_generator: Optional[LLMGenerator] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the Enhanced RAG Engine with better defaults.
        """
        # Load environment variables
        load_dotenv()
        
        # Enhanced default configuration - more comprehensive for all queries
        self.config = {
            "retrieval_k": 15,  # Increased from 5 - get more context by default
            "min_similarity": 0.15,  # Lowered from 0.3 - capture more relevant content
            "max_context_length": 18000,  # Increased for comprehensive answers
            "temperature": 0.1,  # Slightly more focused
            "model": "gpt-4-turbo",
            "max_tokens": 4096,  # Increased for comprehensive responses
        }
        if config:
            self.config.update(config)
        
        # Initialize components with enhanced settings
        self.vector_store = vector_store or VectorStore()
        self.query_processor = query_processor or QueryProcessor(self.vector_store)
        self.context_assembler = context_assembler or ContextAssembler(
            max_context_length=self.config["max_context_length"]
        )
        self.llm_generator = llm_generator or LLMGenerator(
            model=self.config["model"],
            temperature=self.config["temperature"],
            max_tokens=self.config["max_tokens"]
        )
    
    def analyze_query_complexity(self, query: str) -> Dict[str, Any]:
        """
        Analyze query to determine if it needs comprehensive or focused retrieval.
        """
        query_lower = query.lower()
        
        # Indicators of comprehensive queries
        comprehensive_indicators = [
            "tell me about", "explain", "describe", "what is our", "how does our",
            "overview", "structure", "process", "policy", "all", "everything",
            "complete", "full", "entire", "comprehensive"
        ]
        
        # Indicators of specific queries  
        specific_indicators = [
            "when", "where", "who", "which", "what time", "what day",
            "how much", "how many", "name", "contact", "phone", "email"
        ]
        
        # Count indicators
        comprehensive_score = sum(1 for indicator in comprehensive_indicators 
                                if indicator in query_lower)
        specific_score = sum(1 for indicator in specific_indicators 
                           if indicator in query_lower)
        
        # Determine query type
        if comprehensive_score > specific_score and comprehensive_score > 0:
            query_type = "comprehensive"
            suggested_k = min(15, self.config["retrieval_k"] + 3)
            suggested_threshold = max(0.1, self.config["min_similarity"] - 0.05)
        elif specific_score > 0:
            query_type = "specific"
            suggested_k = max(3, self.config["retrieval_k"] - 5)
            suggested_threshold = min(0.3, self.config["min_similarity"] + 0.1)
        else:
            query_type = "general"
            suggested_k = self.config["retrieval_k"]
            suggested_threshold = self.config["min_similarity"]
        
        return {
            "type": query_type,
            "suggested_k": suggested_k,
            "suggested_threshold": suggested_threshold,
            "comprehensive_score": comprehensive_score,
            "specific_score": specific_score
        }
    
    def answer_question(
        self,
        query: str,
        filter_metadata: Optional[Dict[str, Any]] = None,
        stream: bool = False
    ) -> Dict[str, Any]:
        """
        Process a user query with adaptive retrieval based on query complexity.
        """
        # Analyze the query to determine optimal retrieval parameters
        query_analysis = self.analyze_query_complexity(query)
        
        # Use adaptive parameters based on query analysis
        retrieval_k = query_analysis["suggested_k"]
        min_similarity = query_analysis["suggested_threshold"]
        
        # Retrieve relevant chunks with adaptive parameters
        chunks = self.query_processor.retrieve_relevant_chunks(
            query=query,
            k=retrieval_k,
            min_similarity=min_similarity,
            filter_metadata=filter_metadata
        )
        
        # If we got very few chunks and it's a comprehensive query, try again with lower threshold
        if len(chunks) < 3 and query_analysis["type"] == "comprehensive":
            chunks = self.query_processor.retrieve_relevant_chunks(
                query=query,
                k=retrieval_k + 5,
                min_similarity=max(0.05, min_similarity - 0.1),
                filter_metadata=filter_metadata
            )
        
        # Assemble context
        context = self.context_assembler.assemble_context(chunks)
        
        # Generate response
        answer = self.llm_generator.generate_response(query, context, stream)
        
        # Extract sources from chunks for citation
        sources = []
        for chunk in chunks:
            metadata = chunk.get("metadata", {})
            source = {
                "title": metadata.get("source_title", "Unknown Source"),
                "url": metadata.get("source_url", ""),
                "section": metadata.get("header_path", ""),
                "similarity": chunk.get("similarity", 0)
            }
            if source not in sources:
                sources.append(source)
        
        # Return the complete result with analysis info
        return {
            "query": query,
            "answer": answer,
            "sources": sources,
            "context": context,
            "chunks_retrieved": len(chunks),
            "query_analysis": query_analysis,  # Include analysis for debugging
            "retrieval_params": {
                "k": retrieval_k,
                "min_similarity": min_similarity
            }
        }

# For backward compatibility, create an alias
RAGEngine = EnhancedRAGEngine

