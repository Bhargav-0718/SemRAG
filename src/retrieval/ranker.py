"""Ranker module for re-ranking and hybrid retrieval.

Combines local and global search results with optional re-ranking.
"""

import numpy as np
from typing import List, Dict, Any, Tuple
import logging
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


class Ranker:
    """Rank and combine retrieval results."""
    
    def __init__(
        self,
        embedding_function,
        local_weight: float = 0.5,
        global_weight: float = 0.5,
        top_k: int = 5
    ):
        """Initialize ranker.
        
        Args:
            embedding_function: Function to get embeddings
            local_weight: Weight for local search results
            global_weight: Weight for global search results
            top_k: Number of final results to return
        """
        self.embedding_function = embedding_function
        self.local_weight = local_weight
        self.global_weight = global_weight
        self.top_k = top_k
    
    def combine_results(
        self,
        local_results: Dict[str, Any],
        global_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Combine local and global search results.
        
        Args:
            local_results: Results from local search
            global_results: Results from global search
            
        Returns:
            Combined results dictionary
        """
        logger.info("Combining local and global search results")
        
        # Extract chunks and summaries
        local_chunks = local_results.get("chunks", [])
        global_summaries = global_results.get("community_summaries", [])
        entities = local_results.get("entities", [])
        
        # Take top results based on weights
        num_local = int(self.top_k * self.local_weight / (self.local_weight + self.global_weight))
        num_global = self.top_k - num_local
        
        combined = {
            "local_chunks": local_chunks[:num_local],
            "global_summaries": global_summaries[:num_global],
            "entities": entities,
            "local_results": local_results,
            "global_results": global_results
        }
        
        return combined
    
    def rerank_chunks(
        self,
        query: str,
        chunks: List[str]
    ) -> List[Tuple[str, float]]:
        """Re-rank chunks based on query relevance.
        
        Args:
            query: User query
            chunks: List of chunk texts
            
        Returns:
            List of (chunk_text, score) tuples sorted by score
        """
        if len(chunks) == 0:
            return []
        
        logger.info(f"Re-ranking {len(chunks)} chunks")
        
        query_embedding = self.embedding_function(query)
        query_embedding = np.array(query_embedding).reshape(1, -1)
        
        chunk_scores = []
        
        for chunk in chunks:
            try:
                chunk_embedding = self.embedding_function(chunk)
                chunk_embedding = np.array(chunk_embedding).reshape(1, -1)
                
                similarity = cosine_similarity(query_embedding, chunk_embedding)[0][0]
                chunk_scores.append((chunk, similarity))
            except Exception as e:
                logger.warning(f"Error computing similarity for chunk: {e}")
                chunk_scores.append((chunk, 0.0))
        
        # Sort by score
        chunk_scores.sort(key=lambda x: x[1], reverse=True)
        
        return chunk_scores
    
    def hybrid_search(
        self,
        query: str,
        local_results: Dict[str, Any],
        global_results: Dict[str, Any],
        rerank: bool = True
    ) -> Dict[str, Any]:
        """Perform hybrid search combining local and global results.
        
        Args:
            query: User query
            local_results: Results from local search
            global_results: Results from global search
            rerank: Whether to re-rank results
            
        Returns:
            Hybrid search results
        """
        logger.info("Performing hybrid search")
        
        # Combine results
        combined = self.combine_results(local_results, global_results)
        
        # Optional re-ranking
        if rerank:
            local_chunks = combined["local_chunks"]
            if local_chunks:
                reranked = self.rerank_chunks(query, local_chunks)
                combined["local_chunks"] = [chunk for chunk, score in reranked[:self.top_k]]
                combined["chunk_scores"] = [(score) for chunk, score in reranked[:self.top_k]]
        
        return combined
