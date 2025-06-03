# src/generation/llm_generator.py

import os
import time
import openai
from typing import Dict, Any, Optional
from tenacity import retry, stop_after_attempt, wait_exponential

from src.generation.prompt_templates import PromptTemplates

class LLMGenerator:
    """
    Handles interaction with OpenAI's API to generate responses based on
    user queries and retrieved context.
    """
    
    def __init__(
        self,
        model: str = "gpt-4-turbo",
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ):
        """
        Initialize the LLM Generator.
        
        Args:
            model: The OpenAI model to use
            temperature: Controls randomness (0 to 1)
            max_tokens: Maximum response length
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.prompt_templates = PromptTemplates()
        
        # Load API key from environment
        openai.api_key = os.getenv("OPENAI_API_KEY")
        if not openai.api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def generate_response(
        self, 
        query: str, 
        context: Optional[str] = None, 
        stream: bool = False
    ) -> str:
        """
        Generate a response for the given query and context.
        
        Args:
            query: The user's question
            context: The retrieved context
            stream: Whether to stream the response
            
        Returns:
            The generated response
        """
        try:
            # If no context is provided or it's empty, return insufficient info response
            if not context or context == "No relevant information found.":
                return self.prompt_templates.format_insufficient_info(query)
            
            # Format the system and user prompts
            system_prompt = self.prompt_templates.format_system_prompt(context)
            user_prompt = self.prompt_templates.format_user_prompt(query)
            
            # Prepare the message array
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            # Call the OpenAI API
            response = openai.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=stream
            )
            
            # Process the response
            if stream:
                # Return a generator for streaming
                def response_generator():
                    collected_chunks = []
                    for chunk in response:
                        if chunk.choices and chunk.choices[0].delta.content:
                            content = chunk.choices[0].delta.content
                            collected_chunks.append(content)
                            yield content
                    return "".join(collected_chunks)
                return response_generator()
            else:
                # Return the complete response
                return response.choices[0].message.content
                
        except Exception as e:
            print(f"Error generating response: {e}")
            return f"I encountered an error while trying to answer your question. Please try again. Error: {e}"