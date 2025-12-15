"""LLM client for interacting with various language models.

Supports OpenAI, Anthropic, and local models.
"""

import os
import json
from typing import Dict, Any, Optional, List
import openai
from openai import OpenAI
import logging

logger = logging.getLogger(__name__)


class LLMClient:
    """Client for interacting with Large Language Models."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize LLM client with configuration.
        
        Args:
            config: Configuration dictionary with LLM settings
        """
        self.config = config
        self.provider = config.get("provider", "openai")
        self.model = config.get("model", "gpt-4")
        self.temperature = config.get("temperature", 0.7)
        self.max_tokens = config.get("max_tokens", 4096)
        
        # Initialize API key from environment
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key and "api_key" in config:
            api_key = config["api_key"].replace("${OPENAI_API_KEY}", os.getenv("OPENAI_API_KEY", ""))
        
        if self.provider == "openai":
            self.client = OpenAI(api_key=api_key)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    def generate(
        self, 
        prompt: str, 
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        json_mode: bool = False
    ) -> str:
        """Generate text completion from prompt.
        
        Args:
            prompt: Input prompt
            temperature: Override default temperature
            max_tokens: Override default max tokens
            json_mode: Whether to use JSON response format
            
        Returns:
            Generated text response
        """
        temp = temperature if temperature is not None else self.temperature
        max_tok = max_tokens if max_tokens is not None else self.max_tokens
        
        try:
            if self.provider == "openai":
                messages = [{"role": "user", "content": prompt}]
                
                kwargs = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": temp,
                    "max_tokens": max_tok
                }
                
                if json_mode:
                    kwargs["response_format"] = {"type": "json_object"}
                
                response = self.client.chat.completions.create(**kwargs)
                return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise
    
    def generate_json(
        self, 
        prompt: str, 
        temperature: Optional[float] = None
    ) -> Dict[str, Any]:
        """Generate JSON response from prompt.
        
        Args:
            prompt: Input prompt
            temperature: Override default temperature
            
        Returns:
            Parsed JSON response
        """
        response_text = self.generate(
            prompt=prompt,
            temperature=temperature,
            json_mode=True
        )
        
        try:
            return json.loads(response_text)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.debug(f"Response text: {response_text}")
            # Try to extract JSON from response
            start = response_text.find("{")
            end = response_text.rfind("}") + 1
            if start >= 0 and end > start:
                try:
                    return json.loads(response_text[start:end])
                except:
                    pass
            raise
    
    def generate_batch(
        self, 
        prompts: List[str], 
        temperature: Optional[float] = None
    ) -> List[str]:
        """Generate responses for multiple prompts.
        
        Args:
            prompts: List of input prompts
            temperature: Override default temperature
            
        Returns:
            List of generated responses
        """
        return [self.generate(prompt, temperature=temperature) for prompt in prompts]
    
    def get_embedding(self, text: str, model: str = "text-embedding-ada-002") -> List[float]:
        """Get embedding vector for text.
        
        Args:
            text: Input text
            model: Embedding model name
            
        Returns:
            Embedding vector
        """
        try:
            if self.provider == "openai":
                response = self.client.embeddings.create(
                    model=model,
                    input=text
                )
                return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error getting embedding: {e}")
            raise
    
    def get_embeddings_batch(self, texts: List[str], model: str = "text-embedding-ada-002") -> List[List[float]]:
        """Get embedding vectors for multiple texts.
        
        Args:
            texts: List of input texts
            model: Embedding model name
            
        Returns:
            List of embedding vectors
        """
        try:
            if self.provider == "openai":
                response = self.client.embeddings.create(
                    model=model,
                    input=texts
                )
                return [data.embedding for data in response.data]
        except Exception as e:
            logger.error(f"Error getting embeddings: {e}")
            raise
