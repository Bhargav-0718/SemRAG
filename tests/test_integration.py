"""Integration tests for AmbedkarGPT pipeline.

Tests the complete end-to-end workflow of the SemRAG system.
"""

import pytest
import os
import json
import tempfile
import shutil
from pathlib import Path

from src.pipeline.ambedkargpt import AmbedkarGPT


@pytest.fixture
def config_file():
    """Provide config file path."""
    return "config.yaml"


@pytest.fixture
def test_pipeline(config_file):
    """Initialize pipeline for testing."""
    return AmbedkarGPT(config_file)


class TestDataLoading:
    """Test data loading functionality."""
    
    def test_load_processed_data(self, test_pipeline):
        """Test loading previously processed data."""
        # Load data
        test_pipeline.load_processed_data()
        
        # Verify data is loaded
        assert test_pipeline.chunks is not None
        assert len(test_pipeline.chunks) > 0
        assert test_pipeline.entities is not None
        assert len(test_pipeline.entities) > 0
        assert test_pipeline.graph is not None
        assert test_pipeline.graph.number_of_nodes() > 0
        assert test_pipeline.communities is not None
        assert len(test_pipeline.communities) > 0
        
        print(f"✓ Loaded {len(test_pipeline.chunks)} chunks")
        print(f"✓ Loaded {len(test_pipeline.entities)} entities")
        print(f"✓ Loaded graph with {test_pipeline.graph.number_of_nodes()} nodes")
        print(f"✓ Loaded {len(test_pipeline.communities)} communities")
    
    def test_retrieval_initialization(self, test_pipeline):
        """Test retrieval system initialization."""
        # Load data first
        test_pipeline.load_processed_data()
        
        # Initialize retrieval
        test_pipeline._initialize_retrieval()
        
        # Verify retrieval components
        assert test_pipeline.local_search is not None
        assert test_pipeline.global_search is not None
        assert test_pipeline.ranker is not None
        
        # Verify embeddings loaded
        assert len(test_pipeline.local_search.chunk_embeddings) > 0
        assert len(test_pipeline.local_search.entity_embeddings) > 0
        
        print(f"✓ Loaded {len(test_pipeline.local_search.chunk_embeddings)} chunk embeddings")
        print(f"✓ Loaded {len(test_pipeline.local_search.entity_embeddings)} entity embeddings")


class TestLocalSearch:
    """Test local search functionality."""
    
    @pytest.fixture
    def initialized_pipeline(self, test_pipeline):
        """Provide fully initialized pipeline."""
        test_pipeline.load_processed_data()
        test_pipeline._initialize_retrieval()
        return test_pipeline
    
    def test_local_search_query(self, initialized_pipeline):
        """Test local search with a sample query."""
        query = "What is the caste system?"
        
        # Perform local search
        result = initialized_pipeline.local_search.search(query, initialized_pipeline.chunks)
        
        # Verify results
        assert "chunks" in result
        assert "entities" in result
        assert len(result["chunks"]) > 0
        assert len(result["entities"]) > 0
        
        print(f"✓ Local search returned {len(result['chunks'])} chunks")
        print(f"✓ Found entities: {result['entities'][:3]}")
    
    def test_entity_finding(self, initialized_pipeline):
        """Test entity finding functionality."""
        query = "Dr. Ambedkar"
        
        # Find relevant entities
        entities = initialized_pipeline.local_search.find_relevant_entities(query)
        
        # Verify entities found
        assert len(entities) > 0
        assert all(isinstance(e, tuple) and len(e) == 2 for e in entities)
        
        # Check that scores are reasonable (can be > 1 for combined scores)
        for entity_name, score in entities:
            assert score >= 0  # Scores should be non-negative
            assert isinstance(score, (int, float))
        
        print(f"✓ Found {len(entities)} relevant entities")
        print(f"✓ Top entity: {entities[0][0]} (score: {entities[0][1]:.3f})")


class TestGlobalSearch:
    """Test global search functionality."""
    
    @pytest.fixture
    def initialized_pipeline(self, test_pipeline):
        """Provide fully initialized pipeline."""
        test_pipeline.load_processed_data()
        test_pipeline._initialize_retrieval()
        return test_pipeline
    
    def test_global_search_query(self, initialized_pipeline):
        """Test global search with a sample query."""
        query = "What are the main themes in Ambedkar's philosophy?"
        
        # Perform global search
        result = initialized_pipeline.global_search.search(
            query, 
            initialized_pipeline.community_summaries
        )
        
        # Verify results
        assert "community_summaries" in result
        assert len(result["community_summaries"]) > 0
        
        print(f"✓ Global search returned {len(result['community_summaries'])} summaries")


class TestHybridSearch:
    """Test hybrid search functionality."""
    
    @pytest.fixture
    def initialized_pipeline(self, test_pipeline):
        """Provide fully initialized pipeline."""
        test_pipeline.load_processed_data()
        test_pipeline._initialize_retrieval()
        return test_pipeline
    
    def test_hybrid_search_query(self, initialized_pipeline):
        """Test hybrid search combining local and global."""
        query = "How did Ambedkar view the caste system?"
        
        # Perform local and global searches
        local_results = initialized_pipeline.local_search.search(
            query, 
            initialized_pipeline.chunks
        )
        global_results = initialized_pipeline.global_search.search(
            query, 
            initialized_pipeline.community_summaries
        )
        
        # Combine with ranker
        hybrid_results = initialized_pipeline.ranker.hybrid_search(
            query, 
            local_results, 
            global_results,
            rerank=True
        )
        
        # Verify results
        assert "local_chunks" in hybrid_results
        assert "global_summaries" in hybrid_results
        assert len(hybrid_results["local_chunks"]) > 0
        assert len(hybrid_results["global_summaries"]) > 0
        
        print(f"✓ Hybrid search returned {len(hybrid_results['local_chunks'])} chunks")
        print(f"✓ Hybrid search returned {len(hybrid_results['global_summaries'])} summaries")


class TestQueryInterface:
    """Test the main query interface."""
    
    @pytest.fixture
    def initialized_pipeline(self, test_pipeline):
        """Provide fully initialized pipeline."""
        test_pipeline.load_processed_data()
        test_pipeline._initialize_retrieval()
        return test_pipeline
    
    def test_local_query(self, initialized_pipeline):
        """Test query with local search."""
        question = "What is endogamy?"
        
        result = initialized_pipeline.query(question, search_type="local")
        
        # Verify answer structure
        assert "answer" in result
        assert "search_type" in result
        assert result["search_type"] == "local"
        assert "context" in result
        assert len(result["answer"]) > 0
        assert len(result["context"]) > 0
        
        print(f"✓ Local query returned answer of {len(result['answer'])} characters")
        print(f"✓ Used {len(result['context'])} context chunks")
    
    def test_global_query(self, initialized_pipeline):
        """Test query with global search."""
        question = "What are Ambedkar's main contributions?"
        
        result = initialized_pipeline.query(question, search_type="global")
        
        # Verify answer structure
        assert "answer" in result
        assert "search_type" in result
        assert result["search_type"] == "global"
        assert "context" in result
        assert len(result["answer"]) > 0
        
        print(f"✓ Global query returned answer of {len(result['answer'])} characters")
        print(f"✓ Used {result.get('num_communities', 0)} communities")
    
    def test_hybrid_query(self, initialized_pipeline):
        """Test query with hybrid search."""
        question = "How did Ambedkar critique the caste system?"
        
        result = initialized_pipeline.query(question, search_type="hybrid")
        
        # Verify answer structure
        assert "answer" in result
        assert "search_type" in result
        assert result["search_type"] == "hybrid"
        assert "context" in result
        assert len(result["answer"]) > 0
        
        print(f"✓ Hybrid query returned answer of {len(result['answer'])} characters")
        print(f"✓ Used {len(result['context'])} context chunks")
    
    def test_multiple_queries(self, initialized_pipeline):
        """Test multiple consecutive queries."""
        questions = [
            ("What is caste?", "local"),
            ("What are Ambedkar's main ideas?", "global"),
            ("How does caste affect society?", "hybrid")
        ]
        
        for question, search_type in questions:
            result = initialized_pipeline.query(question, search_type=search_type)
            
            assert "answer" in result
            assert len(result["answer"]) > 0
            assert result["search_type"] == search_type
        
        print(f"✓ Successfully processed {len(questions)} consecutive queries")


class TestAnswerQuality:
    """Test answer quality and relevance."""
    
    @pytest.fixture
    def initialized_pipeline(self, test_pipeline):
        """Provide fully initialized pipeline."""
        test_pipeline.load_processed_data()
        test_pipeline._initialize_retrieval()
        return test_pipeline
    
    def test_answer_contains_context(self, initialized_pipeline):
        """Test that answers are based on retrieved context."""
        question = "What is Dr. Ambedkar's view on caste?"
        
        result = initialized_pipeline.query(question, search_type="local")
        
        # Answer should be substantial
        assert len(result["answer"]) > 100
        
        # Should have context
        assert len(result["context"]) > 0
        
        # Context should be relevant (non-empty strings)
        for ctx in result["context"]:
            assert isinstance(ctx, str)
            assert len(ctx) > 0
        
        print(f"✓ Answer is {len(result['answer'])} characters long")
        print(f"✓ Based on {len(result['context'])} context pieces")
    
    def test_entities_in_answer(self, initialized_pipeline):
        """Test that relevant entities are identified."""
        question = "What did Ambedkar say about Brahmins?"
        
        result = initialized_pipeline.query(question, search_type="local")
        
        # Should have entities
        assert "entities" in result
        assert len(result.get("entities", [])) > 0
        
        print(f"✓ Identified {len(result['entities'])} relevant entities")
        print(f"✓ Top entities: {result['entities'][:3]}")


class TestCaching:
    """Test caching functionality."""
    
    def test_embedding_cache_exists(self):
        """Test that embedding caches are created and persist."""
        chunk_cache = Path("./data/processed/chunk_embeddings.pkl")
        community_cache = Path("./data/processed/community_embeddings.pkl")
        
        # Caches should exist after processing
        assert chunk_cache.exists()
        assert community_cache.exists()
        
        # Caches should have reasonable size
        assert chunk_cache.stat().st_size > 0
        assert community_cache.stat().st_size > 0
        
        print(f"✓ Chunk embeddings cache: {chunk_cache.stat().st_size / 1024:.1f} KB")
        print(f"✓ Community embeddings cache: {community_cache.stat().st_size / 1024:.1f} KB")


def run_tests():
    """Run all tests with detailed output."""
    print("=" * 80)
    print("AMBEDKARGPT INTEGRATION TESTS")
    print("=" * 80)
    
    # Run pytest with verbose output
    pytest.main([
        __file__,
        "-v",
        "-s",  # Show print statements
        "--tb=short",  # Short traceback format
        "--color=yes"
    ])


if __name__ == "__main__":
    run_tests()
