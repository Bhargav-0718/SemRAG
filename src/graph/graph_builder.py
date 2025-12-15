"""Graph builder module for constructing knowledge graph.

Builds a networkx graph from entities, chunks, and their relationships.
"""

import networkx as nx
from typing import List, Dict, Any, Optional
import logging
import json

logger = logging.getLogger(__name__)


class GraphBuilder:
    """Build knowledge graph from entities and chunks."""
    
    def __init__(self):
        """Initialize graph builder."""
        self.graph = nx.Graph()
    
    def build_graph(
        self,
        chunks: List[Dict[str, Any]],
        entities: List[Dict[str, Any]],
        relationships: List[Dict[str, Any]]
    ) -> nx.Graph:
        """Build complete knowledge graph.
        
        Args:
            chunks: List of text chunks
            entities: List of extracted entities
            relationships: List of relationships between entities
            
        Returns:
            NetworkX graph
        """
        logger.info("Building knowledge graph")
        
        self.graph.clear()
        
        # Add chunk nodes
        self._add_chunk_nodes(chunks)
        
        # Add entity nodes
        self._add_entity_nodes(entities)
        
        # Add chunk-entity edges (mentions)
        self._add_chunk_entity_edges(entities)
        
        # Add entity-entity edges (relationships)
        self._add_entity_relationships(relationships)
        
        logger.info(f"Graph built: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
        
        return self.graph
    
    def _add_chunk_nodes(self, chunks: List[Dict[str, Any]]):
        """Add chunk nodes to graph."""
        for chunk in chunks:
            chunk_id = f"chunk_{chunk['chunk_id']}"
            self.graph.add_node(
                chunk_id,
                node_type="chunk",
                text=chunk["text"],
                chunk_id=chunk["chunk_id"],
                num_chars=chunk.get("num_chars", len(chunk["text"])),
                num_words=chunk.get("num_words", len(chunk["text"].split()))
            )
        
        logger.info(f"Added {len(chunks)} chunk nodes")
    
    def _add_entity_nodes(self, entities: List[Dict[str, Any]]):
        """Add entity nodes to graph."""
        for i, entity in enumerate(entities):
            entity_id = f"entity_{entity['name']}_{entity['type']}"
            self.graph.add_node(
                entity_id,
                node_type="entity",
                entity_name=entity["name"],
                entity_type=entity["type"],
                description=entity.get("description", ""),
                frequency=entity.get("frequency", 1),
                source_chunks=entity.get("source_chunks", [])
            )
        
        logger.info(f"Added {len(entities)} entity nodes")
    
    def _add_chunk_entity_edges(self, entities: List[Dict[str, Any]]):
        """Add edges between chunks and entities they mention."""
        edge_count = 0
        
        for entity in entities:
            entity_id = f"entity_{entity['name']}_{entity['type']}"
            source_chunks = entity.get("source_chunks", [])
            
            for chunk_id in source_chunks:
                chunk_node_id = f"chunk_{chunk_id}"
                if self.graph.has_node(chunk_node_id) and self.graph.has_node(entity_id):
                    self.graph.add_edge(
                        chunk_node_id,
                        entity_id,
                        edge_type="mentions",
                        weight=1.0
                    )
                    edge_count += 1
        
        logger.info(f"Added {edge_count} chunk-entity edges")
    
    def _add_entity_relationships(self, relationships: List[Dict[str, Any]]):
        """Add edges between related entities."""
        edge_count = 0
        
        for rel in relationships:
            source = rel.get("source", "")
            target = rel.get("target", "")
            rel_type = rel.get("relationship", "relates_to")
            
            # Find entity nodes (may need fuzzy matching)
            source_nodes = [n for n in self.graph.nodes() 
                          if n.startswith("entity_") and source.lower() in n.lower()]
            target_nodes = [n for n in self.graph.nodes() 
                          if n.startswith("entity_") and target.lower() in n.lower()]
            
            # Add edges for all matching pairs
            for src_node in source_nodes:
                for tgt_node in target_nodes:
                    if src_node != tgt_node:
                        self.graph.add_edge(
                            src_node,
                            tgt_node,
                            edge_type="relates_to",
                            relationship=rel_type,
                            description=rel.get("description", ""),
                            weight=1.0
                        )
                        edge_count += 1
        
        logger.info(f"Added {edge_count} entity relationship edges")
    
    def get_entity_neighbors(self, entity_name: str) -> List[str]:
        """Get neighboring entities of a given entity.
        
        Args:
            entity_name: Name of the entity
            
        Returns:
            List of neighboring entity names
        """
        # Find entity node
        entity_nodes = [n for n in self.graph.nodes() 
                       if n.startswith("entity_") and entity_name.lower() in n.lower()]
        
        if not entity_nodes:
            return []
        
        entity_node = entity_nodes[0]
        neighbors = list(self.graph.neighbors(entity_node))
        
        # Filter to only entity neighbors
        entity_neighbors = [n for n in neighbors if n.startswith("entity_")]
        
        # Extract entity names
        return [self.graph.nodes[n]["entity_name"] for n in entity_neighbors]
    
    def get_chunk_entities(self, chunk_id: int) -> List[Dict[str, Any]]:
        """Get all entities mentioned in a chunk.
        
        Args:
            chunk_id: ID of the chunk
            
        Returns:
            List of entity dictionaries
        """
        chunk_node = f"chunk_{chunk_id}"
        
        if not self.graph.has_node(chunk_node):
            return []
        
        neighbors = list(self.graph.neighbors(chunk_node))
        entity_neighbors = [n for n in neighbors if n.startswith("entity_")]
        
        entities = []
        for entity_node in entity_neighbors:
            entity_data = self.graph.nodes[entity_node]
            entities.append({
                "name": entity_data["entity_name"],
                "type": entity_data["entity_type"],
                "description": entity_data.get("description", "")
            })
        
        return entities
    
    def save_graph(self, filepath: str):
        """Save graph to file.
        
        Args:
            filepath: Path to save graph
        """
        data = nx.node_link_data(self.graph)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Graph saved to {filepath}")
    
    def load_graph(self, filepath: str):
        """Load graph from file.
        
        Args:
            filepath: Path to load graph from
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.graph = nx.node_link_graph(data)
        logger.info(f"Graph loaded from {filepath}")
        return self.graph
