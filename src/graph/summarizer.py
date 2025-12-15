"""Summarization module for chunks and communities.

Generates summaries for individual chunks and community-level summaries.
"""

from typing import List, Dict, Any
import logging
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


class Summarizer:
    """Generate summaries for chunks and communities."""
    
    def __init__(self, llm_client, show_progress: bool = True):
        """Initialize summarizer.
        
        Args:
            llm_client: LLM client for generating summaries
        """
        self.llm_client = llm_client
        self.show_progress = show_progress
    
    def summarize_chunk(self, chunk_text: str) -> str:
        """Generate summary for a single chunk.
        
        Args:
            chunk_text: Text content of the chunk
            
        Returns:
            Summary text
        """
        from ..llm.prompt_templates import PromptTemplates
        
        prompt = PromptTemplates.format_chunk_summary(chunk_text)
        
        try:
            summary = self.llm_client.generate(prompt, temperature=0.5, max_tokens=200)
            return summary.strip()
        except Exception as e:
            logger.error(f"Error generating chunk summary: {e}")
            # Return truncated chunk as fallback
            return chunk_text[:200] + "..."
    
    def summarize_chunks(
        self, 
        chunks: List[Dict[str, Any]]
    ) -> Dict[int, str]:
        """Generate summaries for multiple chunks.
        
        Args:
            chunks: List of chunk dictionaries
            
        Returns:
            Dictionary mapping chunk_id to summary
        """
        logger.info(f"Generating summaries for {len(chunks)} chunks")
        
        summaries = {}
        iterable = tqdm(chunks, desc="Summarizing chunks", unit="chunk") if self.show_progress else chunks
        for chunk in iterable:
            chunk_id = chunk["chunk_id"]
            chunk_text = chunk.get("text", "")
            
            summary = self.summarize_chunk(chunk_text)
            summaries[chunk_id] = summary
        
        return summaries
    
    def summarize_community(
        self,
        community_chunks: List[str],
        community_entities: List[str],
        chunk_summaries: Dict[int, str] = None
    ) -> str:
        """Generate summary for a community.
        
        Args:
            community_chunks: List of chunk texts in the community
            community_entities: List of entity names in the community
            chunk_summaries: Optional pre-computed chunk summaries
            
        Returns:
            Community summary text
        """
        from ..llm.prompt_templates import PromptTemplates
        
        # Use chunk summaries if available, otherwise use full text (truncated)
        if chunk_summaries:
            summaries_list = [chunk_summaries.get(i, "") for i in range(len(community_chunks))]
        else:
            # Generate on-the-fly summaries
            summaries_list = [self.summarize_chunk(chunk) for chunk in community_chunks]
        
        prompt = PromptTemplates.format_community_summary(
            entities=community_entities,
            chunk_summaries=summaries_list
        )
        
        try:
            summary = self.llm_client.generate(prompt, temperature=0.6, max_tokens=300)
            return summary.strip()
        except Exception as e:
            logger.error(f"Error generating community summary: {e}")
            # Fallback: combine first few chunk summaries
            return " ".join(summaries_list[:3])
    
    def summarize_communities(
        self,
        communities: Dict[int, List[str]],
        chunks: List[Dict[str, Any]],
        entities_by_community: Dict[int, List[str]],
        chunk_summaries: Dict[int, str] = None
    ) -> Dict[int, str]:
        """Generate summaries for all communities.
        
        Args:
            communities: Dictionary mapping community_id to chunk_ids
            chunks: List of all chunks
            entities_by_community: Dictionary mapping community_id to entity names
            chunk_summaries: Optional pre-computed chunk summaries
            
        Returns:
            Dictionary mapping community_id to summary
        """
        logger.info(f"Generating summaries for {len(communities)} communities")
        
        # Create chunk lookup
        chunk_map = {chunk["chunk_id"]: chunk for chunk in chunks}
        
        community_summaries = {}
        iterable = tqdm(communities.items(), total=len(communities), desc="Summarizing communities", unit="community") if self.show_progress else communities.items()
        
        for comm_id, chunk_ids in iterable:
            # Get chunk texts
            community_chunk_texts = []
            for chunk_id in chunk_ids:
                if chunk_id in chunk_map:
                    community_chunk_texts.append(chunk_map[chunk_id]["text"])
            
            # Get entities for this community
            community_entities = entities_by_community.get(comm_id, [])
            
            # Generate community summary
            if community_chunk_texts:
                summary = self.summarize_community(
                    community_chunks=community_chunk_texts,
                    community_entities=community_entities,
                    chunk_summaries=chunk_summaries
                )
                community_summaries[comm_id] = summary
        
        logger.info(f"Generated {len(community_summaries)} community summaries")
        
        return community_summaries
