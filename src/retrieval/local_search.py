"""Local search module for entity-based retrieval.

Retrieves relevant chunks based on entity matching and semantic similarity.
"""

import numpy as np
from typing import List, Dict, Any, Tuple
import logging
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


class LocalSearch:
    """Local search retriever using entities and semantic similarity."""
    
    def __init__(
        self,
        graph,
        embedding_function,
        top_k_entities: int = 10,
        top_k_chunks: int = 5,
        similarity_weight: float = 0.6,
        graph_weight: float = 0.4,
        show_progress: bool = True
    ):
        """Initialize local search.
        
        Args:
            graph: Knowledge graph (NetworkX)
            embedding_function: Function to get embeddings
            top_k_entities: Number of top entities to consider
            top_k_chunks: Number of chunks to retrieve
            similarity_weight: Weight for semantic similarity score
            graph_weight: Weight for graph-based score
        """
        self.graph = graph
        self.embedding_function = embedding_function
        self.top_k_entities = top_k_entities
        self.top_k_chunks = top_k_chunks
        self.similarity_weight = similarity_weight
        self.graph_weight = graph_weight
        self.show_progress = show_progress
        
        # Cache for embeddings
        self.chunk_embeddings = {}
    
    def compute_chunk_embeddings(self, chunks: List[Dict[str, Any]]):
        """Precompute embeddings for all chunks.
        
        Args:
            chunks: List of chunk dictionaries
        """
        logger.info(f"Computing embeddings for {len(chunks)} chunks")
        iterable = tqdm(chunks, desc="Embedding chunks", unit="chunk") if self.show_progress else chunks
        
        for chunk in iterable:
            chunk_id = chunk["chunk_id"]
            chunk_text = chunk.get("text", "")
            
            try:
                embedding = self.embedding_function(chunk_text)
                self.chunk_embeddings[chunk_id] = embedding
            except Exception as e:
                logger.error(f"Error computing embedding for chunk {chunk_id}: {e}")
    
    def find_relevant_entities(self, query: str) -> List[Tuple[str, float]]:
        """Find entities relevant to the query.
        
        Args:
            query: User query
            
        Returns:
            List of (entity_name, relevance_score) tuples
        """
        # Get query embedding
        query_embedding = self.embedding_function(query)
        query_embedding = np.array(query_embedding).reshape(1, -1)
        
        # Get all entity nodes
        entity_nodes = [n for n in self.graph.nodes() if n.startswith("entity_")]
        
        entity_scores = []
        
        for entity_node in entity_nodes:
            entity_data = self.graph.nodes[entity_node]
            entity_name = entity_data.get("entity_name", "")
            entity_desc = entity_data.get("description", "")
            
            # Compute text for embedding
            entity_text = f"{entity_name} {entity_desc}"
            
            try:
                entity_embedding = self.embedding_function(entity_text)
                entity_embedding = np.array(entity_embedding).reshape(1, -1)
                
                # Compute similarity
                similarity = cosine_similarity(query_embedding, entity_embedding)[0][0]
                
                # Boost by entity frequency
                frequency = entity_data.get("frequency", 1)
                boosted_score = similarity * (1 + np.log1p(frequency) * 0.1)
                
                entity_scores.append((entity_name, entity_node, boosted_score))
            except Exception as e:
                logger.warning(f"Error computing similarity for entity {entity_name}: {e}")
        
        # Sort by score and return top-k
        entity_scores.sort(key=lambda x: x[2], reverse=True)
        return [(name, score) for name, node, score in entity_scores[:self.top_k_entities]]
    
    def get_entity_chunks(self, entity_names: List[str]) -> List[int]:
        """Get chunk IDs that mention the given entities.
        
        Args:
            entity_names: List of entity names
            
        Returns:
            List of chunk IDs
        """
        chunk_ids = set()
        
        for entity_name in entity_names:
            # Find entity node
            entity_nodes = [n for n in self.graph.nodes() 
                          if n.startswith("entity_") and entity_name.lower() in n.lower()]
            
            for entity_node in entity_nodes:
                # Get connected chunk nodes
                neighbors = self.graph.neighbors(entity_node)
                for neighbor in neighbors:
                    if neighbor.startswith("chunk_"):
                        chunk_id = int(neighbor.split("_")[1])
                        chunk_ids.add(chunk_id)
        
        return list(chunk_ids)
    
    def rank_chunks(
        self, 
        query: str, 
        chunk_ids: List[int],
        chunks: List[Dict[str, Any]]
    ) -> List[Tuple[int, float]]:
        """Rank chunks by relevance to query.
        
        Args:
            query: User query
            chunk_ids: List of candidate chunk IDs
            chunks: List of all chunks
            
        Returns:
            List of (chunk_id, score) tuples sorted by score
        """
        query_embedding = self.embedding_function(query)
        query_embedding = np.array(query_embedding).reshape(1, -1)
        
        chunk_scores = []
        
        for chunk_id in chunk_ids:
            if chunk_id in self.chunk_embeddings:
                chunk_embedding = np.array(self.chunk_embeddings[chunk_id]).reshape(1, -1)
                
                # Semantic similarity score
                similarity = cosine_similarity(query_embedding, chunk_embedding)[0][0]
                
                # Graph-based score (number of relevant entities in chunk)
                chunk_node = f"chunk_{chunk_id}"
                if self.graph.has_node(chunk_node):
                    num_entities = len([n for n in self.graph.neighbors(chunk_node) 
                                      if n.startswith("entity_")])
                    graph_score = min(num_entities / 10.0, 1.0)  # Normalize
                else:
                    graph_score = 0.0
                
                # Combined score
                combined_score = (
                    self.similarity_weight * similarity + 
                    self.graph_weight * graph_score
                )
                
                chunk_scores.append((chunk_id, combined_score))
        
        # Sort by score
        chunk_scores.sort(key=lambda x: x[1], reverse=True)
        
        return chunk_scores
    
    def search(
        self, 
        query: str, 
        chunks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Perform local search to retrieve relevant chunks.
        
        Args:
            query: User query
            chunks: List of all chunks
            
        Returns:
            Dictionary with retrieved chunks and entities
        """
        logger.info(f"Performing local search for query: {query}")
        
        # Find relevant entities
        relevant_entities = self.find_relevant_entities(query)
        entity_names = [name for name, score in relevant_entities]
        
        logger.info(f"Found {len(entity_names)} relevant entities")
        
        # Get chunks that mention these entities
        candidate_chunk_ids = self.get_entity_chunks(entity_names)
        
        logger.info(f"Found {len(candidate_chunk_ids)} candidate chunks")
        
        # Rank chunks
        ranked_chunks = self.rank_chunks(query, candidate_chunk_ids, chunks)
        
        # Get top-k chunks
        top_chunk_ids = [chunk_id for chunk_id, score in ranked_chunks[:self.top_k_chunks]]
        
        # Retrieve chunk texts
        chunk_map = {c["chunk_id"]: c for c in chunks}
        retrieved_chunks = [chunk_map[cid]["text"] for cid in top_chunk_ids if cid in chunk_map]
        
        return {
            "chunks": retrieved_chunks,
            "chunk_ids": top_chunk_ids,
            "entities": entity_names[:5],  # Top 5 entities
            "num_candidates": len(candidate_chunk_ids),
            "entity_scores": relevant_entities[:5]
        }
