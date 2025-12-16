from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class BufferMerger:
    """Add buffer sentences around chunks for additional context."""
    
    def __init__(self, buffer_size: int = 1):
        """Initialize buffer merger.
        
        Args:
            buffer_size: Number of sentences to add before/after each chunk
                        (0 = no buffer, 1 = one sentence, 3 = three sentences, etc.)
        """
        self.buffer_size = buffer_size
    
    def add_buffers(
        self, 
        chunks: List[Dict[str, Any]], 
        sentences: List[str]
    ) -> List[Dict[str, Any]]:
        """Add buffer sentences to chunks.
        
        Args:
            chunks: List of chunk dictionaries
            sentences: Original list of all sentences
            
        Returns:
            List of chunks with added buffer sentences
        """
        if self.buffer_size == 0 or len(chunks) == 0:
            return chunks
        
        logger.info(f"Adding buffers of size {self.buffer_size} to {len(chunks)} chunks")
        
        # Map chunks to their sentence ranges
        buffered_chunks = []
        
        for i, chunk in enumerate(chunks):
            chunk_text = chunk["text"]
            
            # Find this chunk's sentences in the original text
            # This is approximate - we'll use the chunk order
            start_sent_idx = chunk.get("start_sentence", i * 10)  # Approximate
            end_sent_idx = chunk.get("end_sentence", min((i + 1) * 10, len(sentences)))
            
            # Calculate buffer boundaries
            buffer_start = max(0, start_sent_idx - self.buffer_size)
            buffer_end = min(len(sentences), end_sent_idx + self.buffer_size)
            
            # Get sentences with buffer
            buffered_sentences = sentences[buffer_start:buffer_end]
            buffered_text = " ".join(buffered_sentences)
            
            # Create new chunk with buffer
            buffered_chunk = {
                **chunk,
                "text_with_buffer": buffered_text,
                "original_text": chunk_text,
                "buffer_size": self.buffer_size,
                "buffer_start_idx": buffer_start,
                "buffer_end_idx": buffer_end,
                "num_buffer_sentences": len(buffered_sentences)
            }
            
            buffered_chunks.append(buffered_chunk)
        
        logger.info(f"Added buffers to all chunks")
        return buffered_chunks
    
    def add_buffers_simple(
        self, 
        chunks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Add overlapping buffers to chunks (simpler method).
        
        This method adds overlap by including parts of adjacent chunks.
        
        Args:
            chunks: List of chunk dictionaries
            
        Returns:
            List of chunks with buffer overlap
        """
        if self.buffer_size == 0 or len(chunks) <= 1:
            # Return as-is with original text marked
            for chunk in chunks:
                chunk["text_with_buffer"] = chunk["text"]
                chunk["original_text"] = chunk["text"]
            return chunks
        
        logger.info(f"Adding simple buffers (overlap) to {len(chunks)} chunks")
        
        buffered_chunks = []
        
        for i, chunk in enumerate(chunks):
            # Get sentences from this chunk
            chunk_sentences = chunk["text"].split(". ")
            
            # Add buffer from previous chunk
            buffer_before = []
            if i > 0:
                prev_sentences = chunks[i-1]["text"].split(". ")
                buffer_before = prev_sentences[-self.buffer_size:] if len(prev_sentences) >= self.buffer_size else prev_sentences
            
            # Add buffer from next chunk
            buffer_after = []
            if i < len(chunks) - 1:
                next_sentences = chunks[i+1]["text"].split(". ")
                buffer_after = next_sentences[:self.buffer_size] if len(next_sentences) >= self.buffer_size else next_sentences
            
            # Combine all parts
            all_parts = buffer_before + chunk_sentences + buffer_after
            buffered_text = ". ".join([s.strip() for s in all_parts if s.strip()])
            
            # Ensure proper sentence endings
            if not buffered_text.endswith("."):
                buffered_text += "."
            
            buffered_chunk = {
                **chunk,
                "text_with_buffer": buffered_text,
                "original_text": chunk["text"],
                "buffer_size": self.buffer_size,
                "num_buffer_before": len(buffer_before),
                "num_buffer_after": len(buffer_after)
            }
            
            buffered_chunks.append(buffered_chunk)
        
        return buffered_chunks
