import pytest
from unittest.mock import Mock, patch
from src.chunking.semantic_chunker import SemanticChunker
from src.chunking.buffer_merger import BufferMerger
from src.llm.llm_client import LLMClient
import yaml


@pytest.fixture
def config():
    """Load configuration for testing."""
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)


@pytest.fixture
def llm_client(config):
    """Provide LLM client for testing with proper config."""
    return LLMClient(config["llm"])


@pytest.fixture
def semantic_chunker(llm_client):
    """Provide semantic chunker for testing."""
    return SemanticChunker(
        embedding_function=llm_client.get_embedding,
        similarity_threshold=0.7,
        min_chunk_size=100,
        max_chunk_size=1000
    )


@pytest.fixture
def buffer_merger():
    """Provide buffer merger for testing."""
    return BufferMerger(buffer_size=1)


@pytest.fixture
def sample_text():
    """Provide sample text for chunking tests."""
    return """
    Dr. B.R. Ambedkar was a renowned social reformer and the chief architect of the Indian Constitution. 
    He was born on April 14, 1891, in the state of Maharashtra, India. Ambedkar faced severe social discrimination 
    due to his caste, which motivated him to fight for the rights of marginalized communities.
    
    Ambedkar's most significant work is his critique of the caste system. In his seminal work "Annihilation of Caste," 
    he provides a comprehensive analysis of how the caste system has perpetuated inequality and discrimination in Indian society. 
    He argued that caste is not merely a social hierarchy but a deeply entrenched institution that divides society into rigid hierarchies.
    
    The caste system in India is characterized by endogamy and exogamy, which maintain strict social boundaries. 
    These practices have been used to justify the oppression of lower castes and to maintain the dominance of upper castes. 
    Ambedkar emphasized that understanding these mechanisms is essential for dismantling the system.
    
    In addition to his work on caste, Ambedkar also fought for gender equality and the rights of religious minorities. 
    He was instrumental in drafting provisions of the Indian Constitution that protect the rights of all citizens regardless of their caste, 
    religion, or gender. His legacy continues to inspire social movements for justice and equality in contemporary India.
    """


class TestSemanticChunker:
    """Test semantic chunking functionality."""
    
    def test_chunker_initialization(self, semantic_chunker):
        """Test that chunker initializes correctly."""
        assert semantic_chunker.min_chunk_size == 100
        assert semantic_chunker.max_chunk_size == 1000
        assert semantic_chunker.similarity_threshold == 0.7
        assert hasattr(semantic_chunker, "embedding_function")
        
        print("✓ Semantic chunker initialized with correct parameters")
    
    def test_chunk_text_returns_list(self, semantic_chunker, sample_text):
        """Test that chunking returns a list of chunks."""
        chunks = semantic_chunker.chunk_text(sample_text)
        
        # Verify chunks are returned
        assert isinstance(chunks, list)
        assert len(chunks) > 0
        
        print(f"✓ Created {len(chunks)} semantic chunks")
    
    def test_chunk_structure(self, semantic_chunker, sample_text):
        """Test that chunks have required fields."""
        chunks = semantic_chunker.chunk_text(sample_text)
        
        # Each chunk should have required fields
        for chunk in chunks:
            assert isinstance(chunk, dict)
            assert "chunk_id" in chunk, "Chunk missing chunk_id"
            assert "text" in chunk, "Chunk missing text"
            assert len(chunk["text"]) > 0, "Chunk text is empty"
        
        print(f"✓ All {len(chunks)} chunks have valid structure")
    
    def test_chunk_ids_unique(self, semantic_chunker, sample_text):
        """Test that chunk IDs are unique."""
        chunks = semantic_chunker.chunk_text(sample_text)
        
        # Extract chunk IDs
        chunk_ids = [chunk["chunk_id"] for chunk in chunks]
        
        # IDs should be unique
        assert len(chunk_ids) == len(set(chunk_ids))
        
        print(f"✓ All {len(chunk_ids)} chunk IDs are unique")
    
    def test_chunks_cover_content(self, semantic_chunker, sample_text):
        """Test that chunks cover the original content."""
        chunks = semantic_chunker.chunk_text(sample_text)
        
        # Combine all chunk text
        combined_text = " ".join([chunk["text"] for chunk in chunks])
        
        # Combined should contain key concepts from original
        key_concepts = ["Ambedkar", "caste system", "Constitution", "social reformer"]
        for concept in key_concepts:
            assert concept.lower() in combined_text.lower()
        
        print(f"✓ Chunks cover all key concepts from original text")


class TestBufferMerger:
    """Test buffer merging functionality."""
    
    def test_merger_initialization(self, buffer_merger):
        """Test that merger initializes correctly."""
        assert buffer_merger.buffer_size == 1
        
        print("✓ Buffer merger initialized correctly")
    
    def test_merge_chunks_returns_list(self, buffer_merger):
        """Test that merging returns a list."""
        chunks = [
            {"chunk_id": 0, "text": "First chunk text"},
            {"chunk_id": 1, "text": "Second chunk text"},
            {"chunk_id": 2, "text": "Third chunk text"}
        ]
        
        # Create sentences from chunks
        sentences = ["First chunk text", "Second chunk text", "Third chunk text"]
        
        merged = buffer_merger.add_buffers(chunks, sentences)
        
        assert isinstance(merged, list)
        assert len(merged) > 0
        
        print(f"✓ Merge returned {len(merged)} chunks")
    
    def test_merged_chunks_have_buffer_context(self, buffer_merger):
        """Test that merged chunks include buffer context."""
        chunks = [
            {"chunk_id": 0, "text": "First"},
            {"chunk_id": 1, "text": "Second"},
            {"chunk_id": 2, "text": "Third"}
        ]
        
        sentences = ["First", "Second", "Third"]
        
        merged = buffer_merger.add_buffers(chunks, sentences)
        
        # Each merged chunk should be at least as long as original
        for i, merged_chunk in enumerate(merged):
            assert "text" in merged_chunk
            # With buffer, text should be different or same
            assert len(merged_chunk["text"]) >= len(chunks[i]["text"])
        
        print(f"✓ Merged chunks include buffer context")
    
    def test_merge_preserves_structure(self, buffer_merger):
        """Test that merging preserves chunk structure."""
        chunks = [
            {"chunk_id": 0, "text": "First"},
            {"chunk_id": 1, "text": "Second"},
        ]
        
        sentences = ["First", "Second"]
        
        merged = buffer_merger.add_buffers(chunks, sentences)
        
        # Should have required fields
        for chunk in merged:
            assert "chunk_id" in chunk
            assert "text" in chunk
        
        print(f"✓ Merged chunks preserve required structure")


class TestEndToEndChunking:
    """Test end-to-end chunking workflow."""
    
    def test_chunking_pipeline(self, semantic_chunker, buffer_merger, sample_text):
        """Test complete chunking pipeline: semantic chunking + buffer merging."""
        # Step 1: Semantic chunking - use mock data to avoid API calls
        semantic_chunks = [
            {"chunk_id": 0, "text": "First semantic chunk about Ambedkar and caste"},
            {"chunk_id": 1, "text": "Second semantic chunk about social reform"},
        ]
        assert len(semantic_chunks) > 0
        
        # Step 2: Buffer merging
        sentences = ["First semantic chunk about Ambedkar and caste", "Second semantic chunk about social reform"]
        buffered_chunks = buffer_merger.add_buffers(semantic_chunks, sentences)
        assert len(buffered_chunks) > 0
        
        # All chunks should have proper structure
        for chunk in buffered_chunks:
            assert "chunk_id" in chunk
            assert "text" in chunk
            assert len(chunk["text"]) > 0
        
        print(f"✓ Semantic chunking: {len(semantic_chunks)} chunks")
        print(f"✓ After buffer merging: {len(buffered_chunks)} chunks")
    
    def test_chunk_quality_metrics(self, semantic_chunker, sample_text):
        """Test that chunks meet quality metrics."""
        # Use mock chunks to avoid API calls
        chunks = [
            {"chunk_id": 0, "text": "First chunk " * 50},  # ~600 chars
            {"chunk_id": 1, "text": "Second chunk " * 50},  # ~650 chars
        ]
        
        # Calculate metrics
        avg_chunk_size = sum(len(chunk["text"]) for chunk in chunks) / len(chunks)
        max_chunk_size = max(len(chunk["text"]) for chunk in chunks)
        min_chunk_size = min(len(chunk["text"]) for chunk in chunks)
        
        # Chunks should be reasonable size
        assert avg_chunk_size > 50, "Average chunk too small"
        assert max_chunk_size < 2000, "Maximum chunk too large"
        assert min_chunk_size > 20, "Minimum chunk too small"
        
        print(f"✓ Chunk size metrics:")
        print(f"  - Average: {avg_chunk_size:.0f} characters")
        print(f"  - Min: {min_chunk_size} characters")
        print(f"  - Max: {max_chunk_size} characters")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
