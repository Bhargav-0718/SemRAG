# AmbedkarGPT: SemRAG Implementation

A Semantic Knowledge-Augmented RAG (Retrieval-Augmented Generation) system for answering questions about Dr. B.R. Ambedkar, based on the [SemRAG research paper](https://arxiv.org/abs/2507.21110).

## Overview

AmbedkarGPT implements the SemRAG architecture that combines:
- **Semantic Chunking**: Intelligent text segmentation based on semantic similarity
- **Entity Extraction**: LLM-based extraction of entities and relationships
- **Knowledge Graph**: Construction of entity-chunk-community graph structure
- **Community Detection**: Identification of thematic communities using Leiden/Louvain algorithms
- **Hierarchical Summarization**: Chunk-level and community-level summaries
- **Hybrid Retrieval**: Local (entity-based) and global (community-based) search

## Features

✅ **Semantic Chunking with Buffer Merging**: Contextual text segmentation
✅ **Entity-Relationship Extraction**: Automated knowledge graph construction
✅ **Multi-level Summarization**: Chunk and community summaries
✅ **Hybrid Search**: Combines local and global retrieval strategies
✅ **Configurable Pipeline**: YAML-based configuration
✅ **Persistent Storage**: Save and reload processed data

## Installation

### Prerequisites

- Python 3.9+
- OpenAI API key (or compatible LLM provider)

### Setup

1. **Clone the repository**:
```bash
git clone https://github.com/yourusername/ambedkargpt.git
cd ambedkargpt
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

Or install in development mode:
```bash
pip install -e .
```

3. **Set up environment variables**:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

Or create a `.env` file:
```
OPENAI_API_KEY=your-api-key-here
```

4. **Download NLTK data**:
```python
import nltk
nltk.download('punkt')
```

## Quick Start

### Basic Usage

```python
from src.pipeline.ambedkargpt import AmbedkarGPT

# Initialize the system
rag_system = AmbedkarGPT(config_path="config.yaml")

# Process a document
rag_system.process_document(pdf_path="data/Ambedkar_book.pdf")

# Query the system
result = rag_system.query(
    "What were Dr. Ambedkar's views on social justice?",
    search_type="hybrid"  # Options: local, global, hybrid
)

print(result["answer"])
```

### Using Pre-processed Data

```python
# Load previously processed data
rag_system = AmbedkarGPT()
rag_system.load_processed_data()

# Query immediately
result = rag_system.query("What is Ambedkar's contribution to the Indian Constitution?")
print(result["answer"])
```

## Configuration

Edit `config.yaml` to customize the system:

```yaml
# Chunking parameters
chunking:
  similarity_threshold: 0.7  # Semantic boundary detection
  buffer_size: 1  # Context sentences (0, 1, 3, 5)
  min_chunk_size: 100
  max_chunk_size: 1000

# Entity extraction
entity_extraction:
  entity_types:
    - PERSON
    - ORGANIZATION
    - LOCATION
    - EVENT
    - CONCEPT

# Community detection
community_detection:
  algorithm: leiden  # leiden or louvain
  resolution: 1.0
  min_community_size: 2

# Retrieval
retrieval:
  local_search:
    top_k_entities: 10
    top_k_chunks: 5
  global_search:
    top_k_communities: 5
  hybrid:
    local_weight: 0.5
    global_weight: 0.5
```

## Architecture

The SemRAG pipeline consists of:

### 1. Document Processing
- **PDF Loading**: Extract text from PDF documents
- **Semantic Chunking**: Split text based on semantic similarity
- **Buffer Merging**: Add context sentences around chunks

### 2. Knowledge Graph Construction
- **Entity Extraction**: Extract entities and relationships using LLM
- **Graph Building**: Create entity-chunk graph with NetworkX
- **Community Detection**: Identify thematic communities

### 3. Summarization
- **Chunk Summaries**: Concise summaries of each chunk
- **Community Summaries**: High-level thematic summaries

### 4. Retrieval & Generation
- **Local Search**: Entity-based retrieval (specific details)
- **Global Search**: Community-based retrieval (broad themes)
- **Hybrid Search**: Combined local and global retrieval
- **Answer Generation**: LLM-based answer synthesis

## Search Types

### Local Search (Entity-based)
Best for: **Specific, detailed questions**
- Retrieves chunks mentioning relevant entities
- Provides detailed, context-specific answers
- Example: "Who were Ambedkar's contemporaries?"

### Global Search (Community-based)
Best for: **High-level, thematic questions**
- Retrieves community summaries
- Provides broad, synthesized answers
- Example: "What were the main themes in Ambedkar's philosophy?"

### Hybrid Search
Best for: **Complex questions requiring both detail and breadth**
- Combines local and global retrieval
- Balances specific details with overall themes
- Example: "How did Ambedkar's personal experiences shape his political philosophy?"

## Project Structure

```
ambedkargpt/
├── config.yaml              # Configuration file
├── requirements.txt         # Dependencies
├── setup.py                # Package setup
├── README.md               # This file
├── data/
│   ├── Ambedkar_book.pdf   # Input document
│   └── processed/          # Processed data
│       ├── chunks.json
│       ├── entities.json
│       ├── graph.json
│       ├── communities.json
│       └── summaries.json
├── src/
│   ├── chunking/           # Semantic chunking modules
│   │   ├── semantic_chunker.py
│   │   └── buffer_merger.py
│   ├── graph/              # Graph construction
│   │   ├── entity_extractor.py
│   │   ├── graph_builder.py
│   │   ├── community_detector.py
│   │   └── summarizer.py
│   ├── llm/                # LLM interaction
│   │   ├── llm_client.py
│   │   ├── prompt_templates.py
│   │   └── answer_generator.py
│   ├── retrieval/          # Retrieval modules
│   │   ├── local_search.py
│   │   ├── global_search.py
│   │   └── ranker.py
│   └── pipeline/           # Main pipeline
│       └── ambedkargpt.py
└── tests/                  # Unit tests
    ├── test_chunking.py
    ├── test_retrieval.py
    └── test_integration.py
```

## API Reference

### AmbedkarGPT

Main pipeline class.

#### Methods

- `process_document(pdf_path=None, text=None)`: Process a document through the pipeline
- `query(question, search_type='hybrid')`: Query the system
- `save_processed_data()`: Save processed data to disk
- `load_processed_data()`: Load previously processed data

### Search Types

- `local`: Entity-based search for specific details
- `global`: Community-based search for broad themes
- `hybrid`: Combined local and global search (recommended)

## Performance Tips

1. **Buffer Size**: 
   - 0: No context (fastest, less accurate)
   - 1: One sentence context (balanced)
   - 3-5: More context (slower, more accurate)

2. **Community Detection**:
   - Use Leiden for better quality (requires igraph)
   - Use Louvain for faster processing

3. **Caching**:
   - Process documents once, then use `load_processed_data()`
   - Embeddings are cached automatically

## Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=src tests/

# Run specific test file
pytest tests/test_chunking.py
```

## Research Paper

This implementation is based on:

**SemRAG: Semantic Knowledge-Augmented RAG for Improved Question-Answering**
- Authors: Kezhen Zhong, Basem Suleiman, Abdelkarim Erradi, Shijing Chen
- arXiv: [2507.21110](https://arxiv.org/abs/2507.21110)

Key contributions:
- Semantic chunking with buffer merging
- Entity-based knowledge graph construction
- Hierarchical community-based summarization
- Hybrid local-global retrieval strategy

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

MIT License - see LICENSE file for details

## Citation

If you use this implementation, please cite the original SemRAG paper:

```bibtex
@article{zhong2025semrag,
  title={SemRAG: Semantic Knowledge-Augmented RAG for Improved Question-Answering},
  author={Zhong, Kezhen and Suleiman, Basem and Erradi, Abdelkarim and Chen, Shijing},
  journal={arXiv preprint arXiv:2507.21110},
  year={2025}
}
```

## Acknowledgments

- SemRAG research paper authors
- Dr. B.R. Ambedkar's contributions to social justice
- OpenAI for GPT models and embeddings
- NetworkX, NLTK, and other open-source libraries

## Support

For issues, questions, or contributions:
- GitHub Issues: [issues](https://github.com/yourusername/ambedkargpt/issues)
- Email: your.email@example.com

---

**Built with ❤️ for preserving and disseminating knowledge about Dr. B.R. Ambedkar**
