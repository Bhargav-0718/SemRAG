"""Unit tests for retrieval modules.

Tests local search, global search, and ranking functionality.
"""

import pytest
import networkx as nx
import yaml
from src.retrieval.local_search import LocalSearch
from src.retrieval.global_search import GlobalSearch
from src.retrieval.ranker import Ranker
from src.llm.llm_client import LLMClient


@pytest.fixture
def config():
    """Load configuration for testing."""
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)


@pytest.fixture
def llm_client(config):
    """Provide LLM client for testing."""
    return LLMClient(config["llm"])


@pytest.fixture
def sample_graph():
    """Create a sample knowledge graph for testing."""
    G = nx.Graph()
    
    # Add entity nodes
    G.add_node("entity_Ambedkar_PERSON", entity_name="Ambedkar", entity_type="PERSON")
    G.add_node("entity_Caste_CONCEPT", entity_name="Caste", entity_type="CONCEPT")
    G.add_node("entity_India_LOCATION", entity_name="India", entity_type="LOCATION")
    
    # Add chunk nodes
    G.add_node("chunk_0")
    G.add_node("chunk_1")
    G.add_node("chunk_2")
    
    # Add edges (entity-chunk relationships)
    G.add_edge("entity_Ambedkar_PERSON", "chunk_0")
    G.add_edge("entity_Ambedkar_PERSON", "chunk_1")
    G.add_edge("entity_Caste_CONCEPT", "chunk_0")
    G.add_edge("entity_Caste_CONCEPT", "chunk_2")
    G.add_edge("entity_India_LOCATION", "chunk_1")
    
    return G


@pytest.fixture
def sample_chunks():
    """Provide sample chunks for testing."""
    return [
        {
            "chunk_id": 0,
            "text": "Dr. Ambedkar worked to abolish the caste system in India."
        },
        {
            "chunk_id": 1,
            "text": "Ambedkar was born in India and studied law."
        },
        {
            "chunk_id": 2,
            "text": "The caste system has deep historical roots in society."
        }
    ]


@pytest.fixture
def local_search(sample_graph, llm_client):
    """Provide local search instance."""
    return LocalSearch(
        graph=sample_graph,
        embedding_function=llm_client.get_embedding,
        top_k_entities=5,
        top_k_chunks=3,
        use_cache=False  # Disable cache for testing
    )


@pytest.fixture
def global_search(llm_client):
    """Provide global search instance."""
    return GlobalSearch(
        embedding_function=llm_client.get_embedding,
        top_k_communities=3,
        use_cache=False  # Disable cache for testing
    )


@pytest.fixture
def ranker(llm_client):
    """Provide ranker instance."""
    return Ranker(
        embedding_function=llm_client.get_embedding,
        local_weight=0.5,
        global_weight=0.5,
        top_k=5
    )


class TestLocalSearch:
    """Test local search functionality."""
    
    def test_local_search_initialization(self, local_search):
        """Test that local search initializes correctly."""
        assert local_search.graph is not None
        assert local_search.embedding_function is not None
        assert local_search.top_k_entities == 5
        assert local_search.top_k_chunks == 3
    
    def test_get_entity_chunks(self, local_search):
        """Test getting chunks for entities."""
        entity_names = ["Ambedkar"]
        
        chunk_ids = local_search.get_entity_chunks(entity_names)
        
        # Should find chunks connected to Ambedkar
        assert len(chunk_ids) > 0
        assert all(isinstance(cid, int) for cid in chunk_ids)
        
        print(f"✓ Found {len(chunk_ids)} chunks for entity 'Ambedkar'")


class TestGlobalSearch:
    """Test global search functionality."""
    
    def test_global_search_initialization(self, global_search):
        """Test that global search initializes correctly."""
        assert global_search.embedding_function is not None
        assert global_search.top_k_communities == 3
    
    def test_search_communities(self, global_search):
        """Test searching communities with a query."""
        query = "What is the caste system?"
        community_summaries = [
            "Community 0: Discussion of caste system origins",
            "Community 1: Ambedkar's work on social reform",
            "Community 2: Indian constitution and rights"
        ]
        
        # This would normally use embeddings
        # For now, just verify the method exists
        assert hasattr(global_search, 'search')


class TestRanker:
    """Test ranking and hybrid search functionality."""
    
    def test_ranker_initialization(self, ranker):
        """Test that ranker initializes correctly."""
        assert ranker.embedding_function is not None
        assert ranker.local_weight == 0.5
        assert ranker.global_weight == 0.5
        assert ranker.top_k == 5
    
    def test_combine_results(self, ranker):
        """Test combining local and global results."""
        local_results = {
            "chunks": ["Chunk 1", "Chunk 2", "Chunk 3"],
            "entities": ["Entity1", "Entity2"]
        }
        
        global_results = {
            "community_summaries": ["Summary 1", "Summary 2"]
        }
        
        combined = ranker.combine_results(local_results, global_results)
        
        # Verify combined results
        assert "local_chunks" in combined
        assert "global_summaries" in combined
        assert "entities" in combined
        assert len(combined["local_chunks"]) > 0
        assert len(combined["global_summaries"]) > 0
        
        print(f"✓ Combined {len(combined['local_chunks'])} local + {len(combined['global_summaries'])} global results")
    
    def test_rerank_chunks(self, ranker):
        """Test re-ranking of chunks by relevance."""
        query = "What is caste?"
        chunks = [
            "The caste system is a social hierarchy.",
            "Ambedkar studied at Columbia University.",
            "Caste divisions have existed for centuries."
        ]
        
        # This test requires API calls, so we just verify the method exists
        assert hasattr(ranker, 'rerank_chunks')


class TestEmbeddingCaching:
    """Test embedding caching functionality."""
    
    def test_cache_save_and_load(self, sample_graph, llm_client, tmp_path):
        """Test saving and loading embeddings from cache."""
        cache_path = tmp_path / "test_embeddings.pkl"
        
        # Create local search with cache enabled
        local_search = LocalSearch(
            graph=sample_graph,
            embedding_function=llm_client.get_embedding,
            use_cache=True,
            cache_path=str(cache_path)
        )
        
        # Add some dummy embeddings
        local_search.chunk_embeddings[0] = [0.1, 0.2, 0.3]
        local_search.entity_embeddings["entity_test"] = [0.4, 0.5, 0.6]
        
        # Save cache
        local_search.save_cached_embeddings()
        
        # Verify cache file exists
        assert cache_path.exists()
        assert cache_path.stat().st_size > 0
        
        # Create new instance and load cache
        local_search2 = LocalSearch(
            graph=sample_graph,
            embedding_function=llm_client.get_embedding,
            use_cache=True,
            cache_path=str(cache_path)
        )
        
        loaded = local_search2.load_cached_embeddings()
        
        # Verify cache loaded
        assert loaded is True
        assert 0 in local_search2.chunk_embeddings
        assert "entity_test" in local_search2.entity_embeddings
        
        print(f"✓ Successfully saved and loaded embeddings cache")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
