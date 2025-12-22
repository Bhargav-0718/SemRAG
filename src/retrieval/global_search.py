import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import logging
import pickle
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


class GlobalSearch:
    """Global search retriever using community summaries."""
    
    def __init__(
        self,
        embedding_function,
        top_k_communities: int = 5,
        cache_path: Optional[str] = None,
        use_cache: bool = True
    ):
        """Initialize global search.
        
        Args:
            embedding_function: Function to get embeddings
            top_k_communities: Number of communities to retrieve
            cache_path: Path to save/load cached embeddings
            use_cache: Whether to use cached embeddings if available
        """
        self.embedding_function = embedding_function
        self.top_k_communities = top_k_communities
        self.cache_path = cache_path
        self.use_cache = use_cache
        
        # Cache for community summary embeddings
        self.community_embeddings = {}
    
    def load_cached_embeddings(self) -> bool:
        """Load embeddings from cache file if available.
        
        Returns:
            True if embeddings were loaded successfully, False otherwise
        """
        if not self.use_cache or not self.cache_path:
            return False
        
        cache_file = Path(self.cache_path)
        if not cache_file.exists():
            logger.info(f"No cached community embeddings found at {self.cache_path}")
            return False
        
        try:
            with open(cache_file, 'rb') as f:
                self.community_embeddings = pickle.load(f)
            logger.info(f"Loaded {len(self.community_embeddings)} cached community embeddings from {self.cache_path}")
            return True
        except Exception as e:
            logger.warning(f"Failed to load cached community embeddings: {e}")
            return False
    
    def save_cached_embeddings(self):
        """Save embeddings to cache file."""
        if not self.use_cache or not self.cache_path:
            return
        
        try:
            cache_file = Path(self.cache_path)
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(cache_file, 'wb') as f:
                pickle.dump(self.community_embeddings, f)
            logger.info(f"Saved {len(self.community_embeddings)} community embeddings to {self.cache_path}")
        except Exception as e:
            logger.error(f"Failed to save cached community embeddings: {e}")
    
    def compute_community_embeddings(
        self, 
        community_summaries: Dict[int, str],
        force_recompute: bool = False
    ):
        """Precompute embeddings for community summaries.
        
        Args:
            community_summaries: Dictionary mapping community_id to summary
            force_recompute: Force recomputation even if cache exists
        """
        # Try to load from cache first
        if not force_recompute and self.load_cached_embeddings():
            # Verify all communities have embeddings
            missing_communities = [cid for cid in community_summaries.keys() 
                                 if cid not in self.community_embeddings]
            if not missing_communities:
                logger.info("All community embeddings loaded from cache")
                return
            else:
                logger.info(f"Found {len(missing_communities)} communities without cached embeddings, computing...")
                community_summaries = {cid: community_summaries[cid] for cid in missing_communities}
        
        logger.info(f"Computing embeddings for {len(community_summaries)} communities")
        
        for comm_id, summary in community_summaries.items():
            try:
                embedding = self.embedding_function(summary)
                self.community_embeddings[comm_id] = embedding
            except Exception as e:
                logger.error(f"Error computing embedding for community {comm_id}: {e}")
        
        # Save to cache
        self.save_cached_embeddings()
    
    def rank_communities(
        self, 
        query: str, 
        community_summaries: Dict[int, str]
    ) -> List[Tuple[int, float]]:
        """Rank communities by relevance to query.
        
        Args:
            query: User query
            community_summaries: Dictionary of community summaries
            
        Returns:
            List of (community_id, score) tuples sorted by score
        """
        query_embedding = self.embedding_function(query)
        query_embedding = np.array(query_embedding).reshape(1, -1)
        
        community_scores = []
        
        for comm_id in community_summaries.keys():
            if comm_id in self.community_embeddings:
                comm_embedding = np.array(self.community_embeddings[comm_id]).reshape(1, -1)
                
                # Compute cosine similarity
                similarity = cosine_similarity(query_embedding, comm_embedding)[0][0]
                
                community_scores.append((comm_id, similarity))
        
        # Sort by score
        community_scores.sort(key=lambda x: x[1], reverse=True)
        
        return community_scores
    
    def search(
        self, 
        query: str, 
        community_summaries: Dict[int, str]
    ) -> Dict[str, Any]:
        """Perform global search to retrieve relevant community summaries.
        
        Args:
            query: User query
            community_summaries: Dictionary of community summaries
            
        Returns:
            Dictionary with retrieved community summaries
        """
        logger.info(f"Performing global search for query: {query}")
        
        # Rank communities
        ranked_communities = self.rank_communities(query, community_summaries)
        
        # Get top-k communities
        top_community_ids = [comm_id for comm_id, score in ranked_communities[:self.top_k_communities]]
        
        # Retrieve community summaries
        retrieved_summaries = [
            community_summaries[comm_id] 
            for comm_id in top_community_ids 
            if comm_id in community_summaries
        ]
        
        return {
            "community_summaries": retrieved_summaries,
            "community_ids": top_community_ids,
            "num_communities": len(community_summaries),
            "community_scores": ranked_communities[:self.top_k_communities]
        }
