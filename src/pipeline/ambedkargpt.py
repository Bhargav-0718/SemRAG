"""Main SemRAG pipeline for AmbedkarGPT.

Orchestrates the entire SemRAG workflow:
1. Document loading and semantic chunking
2. Entity extraction and graph building
3. Community detection and summarization
4. Retrieval and answer generation
"""

import os
import json
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

import PyPDF2

from ..chunking.semantic_chunker import SemanticChunker
from ..chunking.buffer_merger import BufferMerger
from ..graph.entity_extractor import EntityExtractor
from ..graph.graph_builder import GraphBuilder
from ..graph.community_detector import CommunityDetector
from ..graph.summarizer import Summarizer
from ..llm.llm_client import LLMClient
from ..llm.answer_generator import AnswerGenerator
from ..retrieval.local_search import LocalSearch
from ..retrieval.global_search import GlobalSearch
from ..retrieval.ranker import Ranker

logger = logging.getLogger(__name__)


class AmbedkarGPT:
    """Main SemRAG pipeline for AmbedkarGPT system."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize AmbedkarGPT pipeline.
        
        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Setup logging
        self._setup_logging()
        
        logger.info("Initializing AmbedkarGPT pipeline")
        
        # Initialize LLM client
        self.llm_client = LLMClient(self.config["llm"])
        
        # Initialize components
        self.chunker = None
        self.buffer_merger = None
        self.entity_extractor = None
        self.graph_builder = None
        self.community_detector = None
        self.summarizer = None
        self.answer_generator = None
        self.local_search = None
        self.global_search = None
        self.ranker = None
        
        # Data storage
        self.chunks = []
        self.entities = []
        self.relationships = []
        self.graph = None
        self.communities = {}
        self.chunk_summaries = {}
        self.community_summaries = {}
    
    def _setup_logging(self):
        """Setup logging configuration."""
        log_config = self.config.get("logging", {})
        log_level = log_config.get("level", "INFO")
        log_file = log_config.get("file", "ambedkargpt.log")
        
        # Create logs directory if needed
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            level=getattr(logging, log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )

        # Silence noisy HTTP client logs (e.g., httpx from OpenAI)
        for noisy_logger in ["httpx", "httpcore", "openai"]:
            logging.getLogger(noisy_logger).setLevel(logging.WARNING)
    
    def load_pdf(self, pdf_path: str) -> str:
        """Load text from PDF file.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Extracted text
        """
        logger.info(f"Loading PDF: {pdf_path}")
        
        text = ""
        
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
            
            logger.info(f"Extracted {len(text)} characters from PDF")
            return text
        
        except Exception as e:
            logger.error(f"Error loading PDF: {e}")
            raise
    
    def process_document(self, pdf_path: Optional[str] = None, text: Optional[str] = None):
        """Process document through the full SemRAG pipeline.
        
        Args:
            pdf_path: Path to PDF file (optional)
            text: Raw text (optional, if pdf_path not provided)
        """
        logger.info("Starting document processing pipeline (with checkpoints)")
        data_config = self.config["data"]
        
        def _file_nonempty(path_str: str) -> bool:
            p = Path(path_str)
            return p.exists() and p.stat().st_size > 0
        
        def _save_json(path_str: str, obj: Any):
            p = Path(path_str)
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, 'w', encoding='utf-8') as f:
                json.dump(obj, f, indent=2, ensure_ascii=False)
        
        def _load_json(path_str: str) -> Any:
            with open(path_str, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        # Load document
        if text is None:
            if pdf_path is None:
                pdf_path = self.config["data"]["input_pdf"]
            text = self.load_pdf(pdf_path)
        
        # Step 1 & 2: Semantic Chunking + Buffer Merging (checkpoint: processed_chunks)
        if _file_nonempty(data_config["processed_chunks"]):
            logger.info("Checkpoint found: chunks -> loading from disk")
            self.chunks = _load_json(data_config["processed_chunks"])
            logger.info(f"Loaded {len(self.chunks)} chunks from checkpoint")
        else:
            logger.info("Step 1: Semantic Chunking")
            self.chunker = SemanticChunker(
                embedding_function=self.llm_client.get_embedding,
                similarity_threshold=self.config["chunking"]["similarity_threshold"],
                min_chunk_size=self.config["chunking"]["min_chunk_size"],
                max_chunk_size=self.config["chunking"]["max_chunk_size"]
            )
            self.chunks = self.chunker.chunk_text(text)
            logger.info(f"Created {len(self.chunks)} chunks")
            
            logger.info("Step 2: Buffer Merging")
            buffer_size = self.config["chunking"]["buffer_size"]
            self.buffer_merger = BufferMerger(buffer_size=buffer_size)
            self.chunks = self.buffer_merger.add_buffers_simple(self.chunks)
            
            # Save checkpoint
            _save_json(data_config["processed_chunks"], self.chunks)
            logger.info("Saved checkpoint: processed_chunks")
        
        # Step 3: Entity Extraction (checkpoint: entities)
        if _file_nonempty(data_config["entities"]):
            logger.info("Checkpoint found: entities -> loading from disk")
            entities_data = _load_json(data_config["entities"])
            self.entities = entities_data.get("entities", [])
            self.relationships = entities_data.get("relationships", [])
            logger.info(f"Loaded {len(self.entities)} entities and {len(self.relationships)} relationships")
        else:
            logger.info("Step 3: Entity Extraction")
            self.entity_extractor = EntityExtractor(
                llm_client=self.llm_client,
                entity_types=self.config["entity_extraction"]["entity_types"]
            )
            self.entities, self.relationships = self.entity_extractor.extract_from_chunks(self.chunks)
            logger.info(f"Extracted {len(self.entities)} entities and {len(self.relationships)} relationships")
            _save_json(data_config["entities"], {
                "entities": self.entities,
                "relationships": self.relationships
            })
            logger.info("Saved checkpoint: entities")
        
        # Step 4: Graph Building (checkpoint: graph)
        self.graph_builder = GraphBuilder()
        if _file_nonempty(data_config["graph"]):
            logger.info("Checkpoint found: graph -> loading from disk")
            self.graph = self.graph_builder.load_graph(data_config["graph"])
        else:
            logger.info("Step 4: Graph Building")
            self.graph = self.graph_builder.build_graph(self.chunks, self.entities, self.relationships)
            self.graph_builder.save_graph(data_config["graph"])
            logger.info("Saved checkpoint: graph")
        
        # Step 5: Community Detection (checkpoint: communities)
        self.community_detector = CommunityDetector(
            algorithm=self.config["community_detection"]["algorithm"],
            resolution=self.config["community_detection"]["resolution"],
            min_community_size=self.config["community_detection"]["min_community_size"]
        )
        communities_raw = None
        if _file_nonempty(data_config["communities"]):
            logger.info("Checkpoint found: communities -> loading from disk")
            self.communities = _load_json(data_config["communities"])
        else:
            logger.info("Step 5: Community Detection")
            communities_raw = self.community_detector.detect_communities(self.graph)
            self.communities = self.community_detector.get_community_chunks(communities_raw)
            _save_json(data_config["communities"], self.communities)
            logger.info("Saved checkpoint: communities")
        
        # Step 6: Summarization (checkpoint: summaries)
        def _compute_entities_by_community_from_chunks() -> Dict[int, List[str]]:
            # Derive entities per community using graph neighbors of chunk nodes
            mapping = {}
            for comm_id, chunk_ids in self.communities.items():
                names = set()
                for cid in chunk_ids:
                    node = f"chunk_{cid}"
                    if self.graph is not None and self.graph.has_node(node):
                        for neigh in self.graph.neighbors(node):
                            if neigh.startswith("entity_"):
                                parts = neigh.split("_", 2)
                                if len(parts) >= 3:
                                    names.add(parts[1])
                mapping[comm_id] = sorted(names)
            return mapping
        
        if _file_nonempty(data_config["summaries"]):
            logger.info("Checkpoint found: summaries -> loading from disk")
            summaries_data = _load_json(data_config["summaries"])
            self.chunk_summaries = summaries_data.get("chunk_summaries", {})
            self.community_summaries = summaries_data.get("community_summaries", {})
        else:
            logger.info("Step 6: Summarization")
            self.summarizer = Summarizer(
                llm_client=self.llm_client,
                show_progress=self.config["summarization"].get("progress_bar", True)
            )
            # Generate chunk summaries
            self.chunk_summaries = self.summarizer.summarize_chunks(self.chunks)
            # Generate community summaries
            if communities_raw is None:
                entities_by_community = _compute_entities_by_community_from_chunks()
            else:
                entities_by_community = self.community_detector.get_community_entities(communities_raw)
            self.community_summaries = self.summarizer.summarize_communities(
                communities=self.communities,
                chunks=self.chunks,
                entities_by_community=entities_by_community,
                chunk_summaries=self.chunk_summaries
            )
            _save_json(data_config["summaries"], {
                "chunk_summaries": self.chunk_summaries,
                "community_summaries": self.community_summaries
            })
            logger.info("Saved checkpoint: summaries")
        
        # Step 7: Initialize Retrieval Components
        logger.info("Step 7: Initializing Retrieval Components")
        
        # Local search
        self.local_search = LocalSearch(
            graph=self.graph,
            embedding_function=self.llm_client.get_embedding,
            top_k_entities=self.config["retrieval"]["local_search"]["top_k_entities"],
            top_k_chunks=self.config["retrieval"]["local_search"]["top_k_chunks"],
            similarity_weight=self.config["retrieval"]["local_search"]["similarity_weight"],
            graph_weight=self.config["retrieval"]["local_search"]["graph_weight"],
            show_progress=self.config["retrieval"]["local_search"].get("progress_bar", True)
        )
        self.local_search.compute_chunk_embeddings(self.chunks)
        
        # Global search
        self.global_search = GlobalSearch(
            embedding_function=self.llm_client.get_embedding,
            top_k_communities=self.config["retrieval"]["global_search"]["top_k_communities"]
        )
        self.global_search.compute_community_embeddings(self.community_summaries)
        
        # Ranker
        self.ranker = Ranker(
            embedding_function=self.llm_client.get_embedding,
            local_weight=self.config["retrieval"]["hybrid"]["local_weight"],
            global_weight=self.config["retrieval"]["hybrid"]["global_weight"],
            top_k=self.config["ranking"]["top_k_final"]
        )
        
        # Answer generator
        self.answer_generator = AnswerGenerator(llm_client=self.llm_client)
        
        logger.info("Document processing complete (checkpoints written)")
    
    def query(self, question: str, search_type: str = "hybrid") -> Dict[str, Any]:
        """Query the system with a question.
        
        Args:
            question: User question
            search_type: Type of search (local, global, or hybrid)
            
        Returns:
            Dictionary with answer and metadata
        """
        logger.info(f"Processing query: {question}")
        
        if self.local_search is None:
            raise ValueError("System not initialized. Call process_document() first.")
        
        # Perform retrieval based on search type
        if search_type == "local":
            retrieval_results = self.local_search.search(question, self.chunks)
        elif search_type == "global":
            retrieval_results = self.global_search.search(question, self.community_summaries)
        else:  # hybrid
            local_results = self.local_search.search(question, self.chunks)
            global_results = self.global_search.search(question, self.community_summaries)
            retrieval_results = self.ranker.hybrid_search(
                question, 
                local_results, 
                global_results,
                rerank=self.config["ranking"]["rerank"]
            )
        
        # Generate answer
        answer_data = self.answer_generator.generate_answer(
            question=question,
            retrieval_results=retrieval_results,
            search_type=search_type
        )
        
        logger.info("Query processing complete")
        
        return answer_data
    
    def save_processed_data(self):
        """Save processed data to files."""
        logger.info("Saving processed data")
        
        data_config = self.config["data"]
        
        # Create directories
        for path_key in ["processed_chunks", "entities", "graph", "communities", "summaries"]:
            path = Path(data_config[path_key])
            path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save chunks
        with open(data_config["processed_chunks"], 'w', encoding='utf-8') as f:
            json.dump(self.chunks, f, indent=2, ensure_ascii=False)
        
        # Save entities
        with open(data_config["entities"], 'w', encoding='utf-8') as f:
            json.dump({
                "entities": self.entities,
                "relationships": self.relationships
            }, f, indent=2, ensure_ascii=False)
        
        # Save graph
        if self.graph:
            self.graph_builder.save_graph(data_config["graph"])
        
        # Save communities
        with open(data_config["communities"], 'w', encoding='utf-8') as f:
            json.dump(self.communities, f, indent=2, ensure_ascii=False)
        
        # Save summaries
        with open(data_config["summaries"], 'w', encoding='utf-8') as f:
            json.dump({
                "chunk_summaries": self.chunk_summaries,
                "community_summaries": self.community_summaries
            }, f, indent=2, ensure_ascii=False)
        
        logger.info("Data saved successfully")
    
    def load_processed_data(self):
        """Load previously processed data."""
        logger.info("Loading processed data")
        
        data_config = self.config["data"]

        def _ensure_file(path_key: str):
            path = Path(data_config[path_key])
            if not path.exists():
                raise FileNotFoundError(f"Required processed file missing: {path}")
            if path.stat().st_size == 0:
                raise ValueError(f"Processed file is empty: {path}")
            return path
        
        # Validate files exist and are non-empty
        required_files = ["processed_chunks", "entities", "graph", "communities", "summaries"]
        paths = {key: _ensure_file(key) for key in required_files}
        
        # Load chunks
        try:
            with open(paths["processed_chunks"], 'r', encoding='utf-8') as f:
                self.chunks = json.load(f)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Unable to parse processed chunks file: {paths['processed_chunks']}") from exc
        
        # Load entities
        try:
            with open(paths["entities"], 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.entities = data["entities"]
                self.relationships = data["relationships"]
        except json.JSONDecodeError as exc:
            raise ValueError(f"Unable to parse entities file: {paths['entities']}") from exc
        
        # Load graph
        self.graph_builder = GraphBuilder()
        self.graph = self.graph_builder.load_graph(str(paths["graph"]))
        
        # Load communities
        try:
            with open(paths["communities"], 'r', encoding='utf-8') as f:
                self.communities = json.load(f)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Unable to parse communities file: {paths['communities']}") from exc
        
        # Load summaries
        try:
            with open(paths["summaries"], 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.chunk_summaries = data["chunk_summaries"]
                self.community_summaries = data["community_summaries"]
        except json.JSONDecodeError as exc:
            raise ValueError(f"Unable to parse summaries file: {paths['summaries']}") from exc
        
        # Reinitialize retrieval components
        self._initialize_retrieval()
        
        logger.info("Data loaded successfully")
    
    def _initialize_retrieval(self):
        """Initialize retrieval components with loaded data."""
        # Local search
        self.local_search = LocalSearch(
            graph=self.graph,
            embedding_function=self.llm_client.get_embedding,
            top_k_entities=self.config["retrieval"]["local_search"]["top_k_entities"],
            top_k_chunks=self.config["retrieval"]["local_search"]["top_k_chunks"],
            similarity_weight=self.config["retrieval"]["local_search"]["similarity_weight"],
            graph_weight=self.config["retrieval"]["local_search"]["graph_weight"]
        )
        self.local_search.compute_chunk_embeddings(self.chunks)
        
        # Global search
        self.global_search = GlobalSearch(
            embedding_function=self.llm_client.get_embedding,
            top_k_communities=self.config["retrieval"]["global_search"]["top_k_communities"]
        )
        self.global_search.compute_community_embeddings(self.community_summaries)
        
        # Ranker
        self.ranker = Ranker(
            embedding_function=self.llm_client.get_embedding,
            local_weight=self.config["retrieval"]["hybrid"]["local_weight"],
            global_weight=self.config["retrieval"]["hybrid"]["global_weight"],
            top_k=self.config["ranking"]["top_k_final"]
        )
        
        # Answer generator
        self.answer_generator = AnswerGenerator(llm_client=self.llm_client)
