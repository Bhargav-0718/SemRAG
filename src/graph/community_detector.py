"""Community detection module for knowledge graph.

Detects communities in the graph using Leiden or Louvain algorithm.
"""

import networkx as nx
from typing import List, Dict, Any, Set
import logging

try:
    import community as community_louvain
except ImportError:
    community_louvain = None

try:
    import leidenalg
    import igraph as ig
except ImportError:
    leidenalg = None
    ig = None

logger = logging.getLogger(__name__)


class CommunityDetector:
    """Detect communities in knowledge graph."""
    
    def __init__(
        self, 
        algorithm: str = "leiden",
        resolution: float = 1.0,
        min_community_size: int = 2
    ):
        """Initialize community detector.
        
        Args:
            algorithm: Algorithm to use (leiden or louvain)
            resolution: Resolution parameter for community detection
            min_community_size: Minimum number of nodes in a community
        """
        self.algorithm = algorithm
        self.resolution = resolution
        self.min_community_size = min_community_size
    
    def detect_communities(self, graph: nx.Graph) -> Dict[int, List[str]]:
        """Detect communities in graph.
        
        Args:
            graph: NetworkX graph
            
        Returns:
            Dictionary mapping community_id to list of node_ids
        """
        logger.info(f"Detecting communities using {self.algorithm} algorithm")
        
        if self.algorithm == "leiden" and leidenalg is not None:
            communities = self._detect_leiden(graph)
        elif self.algorithm == "louvain" or community_louvain is not None:
            communities = self._detect_louvain(graph)
        else:
            # Fallback to simple connected components
            logger.warning("Required libraries not installed, using connected components")
            communities = self._detect_connected_components(graph)
        
        # Filter small communities
        filtered_communities = {
            cid: nodes for cid, nodes in communities.items()
            if len(nodes) >= self.min_community_size
        }
        
        logger.info(f"Detected {len(filtered_communities)} communities (min size: {self.min_community_size})")
        
        return filtered_communities
    
    def _detect_leiden(self, graph: nx.Graph) -> Dict[int, List[str]]:
        """Detect communities using Leiden algorithm."""
        if ig is None or leidenalg is None:
            logger.warning("Leiden algorithm not available, falling back to Louvain")
            return self._detect_louvain(graph)
        
        # Convert NetworkX graph to igraph
        node_list = list(graph.nodes())
        node_to_idx = {node: idx for idx, node in enumerate(node_list)}
        
        edges = [(node_to_idx[u], node_to_idx[v]) for u, v in graph.edges()]
        
        ig_graph = ig.Graph()
        ig_graph.add_vertices(len(node_list))
        ig_graph.add_edges(edges)
        
        # Run Leiden algorithm
        # Some versions of leidenalg do not support 'resolution_parameter' for ModularityVertexPartition.
        # Try without the parameter; if a non-default resolution was requested, warn that it's ignored.
        try:
            if self.resolution != 1.0:
                logger.warning("Leiden ModularityPartition: 'resolution' ignored due to library constraints; using default.")
            partition = leidenalg.find_partition(
                ig_graph,
                leidenalg.ModularityVertexPartition
            )
        except TypeError:
            logger.warning("Leiden call failed with ModularityPartition; falling back to Louvain")
            return self._detect_louvain(graph)
        
        # Convert to dictionary format
        communities = {}
        for comm_id, community in enumerate(partition):
            communities[comm_id] = [node_list[idx] for idx in community]
        
        return communities
    
    def _detect_louvain(self, graph: nx.Graph) -> Dict[int, List[str]]:
        """Detect communities using Louvain algorithm."""
        if community_louvain is None:
            logger.warning("Louvain algorithm not available, using connected components")
            return self._detect_connected_components(graph)
        
        # Run Louvain algorithm
        partition = community_louvain.best_partition(
            graph,
            resolution=self.resolution
        )
        
        # Convert to dictionary format
        communities = {}
        for node, comm_id in partition.items():
            if comm_id not in communities:
                communities[comm_id] = []
            communities[comm_id].append(node)
        
        return communities
    
    def _detect_connected_components(self, graph: nx.Graph) -> Dict[int, List[str]]:
        """Fallback: use connected components as communities."""
        components = nx.connected_components(graph)
        communities = {i: list(comp) for i, comp in enumerate(components)}
        return communities
    
    def get_community_chunks(self, communities: Dict[int, List[str]]) -> Dict[int, List[int]]:
        """Extract chunk IDs for each community.
        
        Args:
            communities: Dictionary mapping community_id to node_ids
            
        Returns:
            Dictionary mapping community_id to chunk_ids
        """
        community_chunks = {}
        
        for comm_id, nodes in communities.items():
            chunk_ids = []
            for node in nodes:
                if node.startswith("chunk_"):
                    chunk_id = int(node.split("_")[1])
                    chunk_ids.append(chunk_id)
            community_chunks[comm_id] = chunk_ids
        
        return community_chunks
    
    def get_community_entities(self, communities: Dict[int, List[str]]) -> Dict[int, List[str]]:
        """Extract entity names for each community.
        
        Args:
            communities: Dictionary mapping community_id to node_ids
            
        Returns:
            Dictionary mapping community_id to entity_names
        """
        community_entities = {}
        
        for comm_id, nodes in communities.items():
            entities = []
            for node in nodes:
                if node.startswith("entity_"):
                    # Extract entity name from node ID
                    parts = node.split("_", 2)
                    if len(parts) >= 3:
                        entity_name = parts[1]
                        entities.append(entity_name)
            community_entities[comm_id] = entities
        
        return community_entities
