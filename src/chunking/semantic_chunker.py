"""Semantic chunking module using embedding-based similarity.

Implements the SemRAG semantic chunking algorithm that splits text
into chunks based on semantic similarity between consecutive sentences.
"""

import nltk
from typing import List, Dict, Any, Tuple
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import logging
from tqdm import tqdm

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)

logger = logging.getLogger(__name__)


class SemanticChunker:
    """Semantic chunker that uses embeddings to detect topical boundaries."""

    def __init__(
        self,
        embedding_function,
        similarity_threshold: float = 0.7,
        min_chunk_size: int = 100,
        max_chunk_size: int = 1000,
    ):
        """Initialize semantic chunker.

        Args:
            embedding_function: Function to get embeddings for text
            similarity_threshold: Threshold for semantic similarity (0-1)
            min_chunk_size: Minimum characters per chunk
            max_chunk_size: Maximum characters per chunk
        """
        self.embedding_function = embedding_function
        self.similarity_threshold = similarity_threshold
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size

    def split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using NLTK."""
        try:
            sentences = nltk.sent_tokenize(text)
            sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
            return sentences
        except Exception as e:
            logger.error(f"Error splitting sentences: {e}")
            return [
                s.strip() + "."
                for s in text.split(".")
                if len(s.strip()) > 10
            ]

    def compute_sentence_similarities(self, sentences: List[str]) -> List[float]:
        """Compute cosine similarity between consecutive sentences."""

        if len(sentences) < 2:
            return []

        # Step 1: Embed sentences (MOST EXPENSIVE STEP)
        embeddings = []

        for sentence in tqdm(
            sentences,
            desc="Embedding sentences (Semantic Chunking)",
            unit="sentence",
        ):
            try:
                emb = self.embedding_function(sentence)
                embeddings.append(emb)
            except Exception as e:
                logger.warning(f"Error getting embedding for sentence: {e}")
                embeddings.append(
                    [0.0] * len(embeddings[0])
                    if embeddings
                    else [0.0] * 1536
                )

        embeddings = np.array(embeddings)

        # Step 2: Compute cosine similarities
        similarities = []

        for i in tqdm(
            range(len(embeddings) - 1),
            desc="Computing cosine similarities",
            unit="pair",
        ):
            sim = cosine_similarity(
                embeddings[i].reshape(1, -1),
                embeddings[i + 1].reshape(1, -1),
            )[0][0]
            similarities.append(sim)

        return similarities

    def find_chunk_boundaries(self, similarities: List[float]) -> List[int]:
        """Find chunk boundaries based on similarity drops."""
        boundaries = [0]

        for i, sim in enumerate(similarities):
            if sim < self.similarity_threshold:
                boundaries.append(i + 1)

        boundaries.append(len(similarities) + 1)
        return boundaries

    def merge_small_chunks(
        self,
        chunks: List[str],
        boundaries: List[int],
    ) -> Tuple[List[str], List[int]]:
        """Merge chunks that are too small."""
        merged_chunks = []
        merged_boundaries = [boundaries[0]]
        current_chunk = ""

        for i, chunk in enumerate(chunks):
            if not current_chunk:
                current_chunk = chunk
            elif len(current_chunk) + len(chunk) < self.min_chunk_size:
                current_chunk += " " + chunk
            else:
                merged_chunks.append(current_chunk)
                merged_boundaries.append(boundaries[i])
                current_chunk = chunk

        if current_chunk:
            merged_chunks.append(current_chunk)
            merged_boundaries.append(boundaries[-1])

        return merged_chunks, merged_boundaries

    def split_large_chunks(self, chunks: List[str]) -> List[str]:
        """Split chunks that exceed max size."""
        result_chunks = []

        for chunk in chunks:
            if len(chunk) <= self.max_chunk_size:
                result_chunks.append(chunk)
            else:
                sentences = self.split_into_sentences(chunk)
                current = ""

                for sent in sentences:
                    if len(current) + len(sent) <= self.max_chunk_size:
                        current += " " + sent if current else sent
                    else:
                        if current:
                            result_chunks.append(current)
                        current = sent

                if current:
                    result_chunks.append(current)

        return result_chunks

    def chunk_text(self, text: str) -> List[Dict[str, Any]]:
        """Chunk text into semantically coherent segments."""
        logger.info("Starting semantic chunking")

        sentences = self.split_into_sentences(text)
        logger.info(f"Split into {len(sentences)} sentences")

        if not sentences:
            return []

        if len(sentences) == 1:
            return [
                {
                    "chunk_id": 0,
                    "text": sentences[0],
                    "start_sentence": 0,
                    "end_sentence": 1,
                }
            ]

        similarities = self.compute_sentence_similarities(sentences)
        logger.info(f"Computed {len(similarities)} similarity scores")

        boundaries = self.find_chunk_boundaries(similarities)
        logger.info(f"Found {len(boundaries) - 1} initial chunks")

        chunks = []
        for i in range(len(boundaries) - 1):
            start_idx = boundaries[i]
            end_idx = boundaries[i + 1]
            chunk_text = " ".join(sentences[start_idx:end_idx])
            chunks.append(chunk_text)

        chunks, boundaries = self.merge_small_chunks(chunks, boundaries)
        logger.info(f"After merging small chunks: {len(chunks)} chunks")

        chunks = self.split_large_chunks(chunks)
        logger.info(f"After splitting large chunks: {len(chunks)} chunks")

        result = []
        for i, chunk_text in enumerate(chunks):
            result.append(
                {
                    "chunk_id": i,
                    "text": chunk_text,
                    "num_chars": len(chunk_text),
                    "num_words": len(chunk_text.split()),
                }
            )

        return result
