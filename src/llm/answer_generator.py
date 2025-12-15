"""Answer generation module for SemRAG system.

Handles local, global, and hybrid search answer generation.
"""

from typing import Dict, Any, List, Optional
import logging
from .llm_client import LLMClient
from .prompt_templates import PromptTemplates

logger = logging.getLogger(__name__)


class AnswerGenerator:
    """Generate answers using different search strategies."""
    
    def __init__(self, llm_client: LLMClient):
        """Initialize answer generator.
        
        Args:
            llm_client: LLM client instance
        """
        self.llm_client = llm_client
        self.prompts = PromptTemplates()
    
    def generate_local_answer(
        self, 
        question: str, 
        context_chunks: List[str],
        entities: List[str]
    ) -> Dict[str, Any]:
        """Generate answer using local search (entity-based) context.
        
        Args:
            question: User question
            context_chunks: Retrieved context chunks
            entities: Relevant entities
            
        Returns:
            Dictionary with answer and metadata
        """
        logger.info(f"Generating local search answer for: {question}")
        
        prompt = self.prompts.format_local_search(
            question=question,
            context=context_chunks,
            entities=entities
        )
        
        answer = self.llm_client.generate(prompt, temperature=0.7)
        
        return {
            "answer": answer,
            "search_type": "local",
            "num_chunks": len(context_chunks),
            "entities": entities,
            "context": context_chunks
        }
    
    def generate_global_answer(
        self, 
        question: str, 
        community_summaries: List[str]
    ) -> Dict[str, Any]:
        """Generate answer using global search (community-based) context.
        
        Args:
            question: User question
            community_summaries: Community summaries
            
        Returns:
            Dictionary with answer and metadata
        """
        logger.info(f"Generating global search answer for: {question}")
        
        prompt = self.prompts.format_global_search(
            question=question,
            community_summaries=community_summaries
        )
        
        answer = self.llm_client.generate(prompt, temperature=0.7)
        
        return {
            "answer": answer,
            "search_type": "global",
            "num_communities": len(community_summaries),
            "community_summaries": community_summaries
        }
    
    def generate_hybrid_answer(
        self,
        question: str,
        local_context: List[str],
        global_context: List[str],
        entities: List[str]
    ) -> Dict[str, Any]:
        """Generate answer using hybrid search context.
        
        Args:
            question: User question
            local_context: Local search context chunks
            global_context: Global search community summaries
            entities: Relevant entities
            
        Returns:
            Dictionary with answer and metadata
        """
        logger.info(f"Generating hybrid search answer for: {question}")
        
        prompt = self.prompts.format_hybrid_search(
            question=question,
            local_context=local_context,
            global_context=global_context,
            entities=entities
        )
        
        answer = self.llm_client.generate(prompt, temperature=0.7)
        
        return {
            "answer": answer,
            "search_type": "hybrid",
            "num_local_chunks": len(local_context),
            "num_global_communities": len(global_context),
            "entities": entities,
            "local_context": local_context,
            "global_context": global_context
        }
    
    def generate_answer(
        self,
        question: str,
        retrieval_results: Dict[str, Any],
        search_type: str = "hybrid"
    ) -> Dict[str, Any]:
        """Generate answer based on retrieval results and search type.
        
        Args:
            question: User question
            retrieval_results: Results from retrieval system
            search_type: Type of search (local, global, or hybrid)
            
        Returns:
            Dictionary with answer and metadata
        """
        if search_type == "local":
            return self.generate_local_answer(
                question=question,
                context_chunks=retrieval_results.get("chunks", []),
                entities=retrieval_results.get("entities", [])
            )
        elif search_type == "global":
            return self.generate_global_answer(
                question=question,
                community_summaries=retrieval_results.get("community_summaries", [])
            )
        else:  # hybrid
            return self.generate_hybrid_answer(
                question=question,
                local_context=retrieval_results.get("local_chunks", []),
                global_context=retrieval_results.get("global_summaries", []),
                entities=retrieval_results.get("entities", [])
            )
