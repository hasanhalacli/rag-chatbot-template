# RAG Chatbot Template

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Qdrant](https://img.shields.io/badge/Qdrant-vector%20DB-red.svg)](https://qdrant.tech/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-green.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

Production-ready RAG (Retrieval-Augmented Generation) chatbot template. **No LangChain, no LlamaIndex** - just clean, deployment-friendly Python code.

## Why No LangChain/LlamaIndex?

This template is intentionally built without heavy frameworks:

| Aspect | With Frameworks | This Template |
|--------|-----------------|---------------|
| Dependencies | 50+ packages | ~15 packages |
| Docker image | 2-3 GB | < 500 MB |
| Cold start | 10-30s | 2-5s |
| Debugging | Abstraction layers | Direct code |
| Customization | Override patterns | Modify directly |
| Production | Framework updates break things | You control everything |

**Perfect for**: Production deployments, custom RAG pipelines, learning RAG internals.

## Features

- **Multi-Provider LLM Support**: OpenAI, Azure OpenAI, Anthropic Claude, xAI Grok
- **Qdrant Vector Store**: Fast, production-ready vector similarity search
- **Flexible Chunking**: Recursive, semantic, or sentence-based strategies
- **Reranking**: Cross-encoder reranking for improved relevance
- **Conversation Memory**: Configurable context window management
- **RAG Evaluation**: Built-in faithfulness and relevance metrics
- **FastAPI Server**: Production-ready API with health checks
- **Docker Ready**: Multi-stage Dockerfile for minimal images

## Quick Start

### Installation

```bash
git clone https://github.com/hasanhalacli/rag-chatbot-template.git
cd rag-chatbot-template

# Using uv (recommended)
uv sync

# Or pip
pip install -e .
```

### Environment Setup

```bash
cp .env.example .env
# Edit .env with your API keys
```

### Start Qdrant

```bash
docker-compose up -d qdrant
```

### Ingest Documents

```bash
python scripts/ingest.py --input_dir data/documents --collection my_docs
```

### Chat

```bash
# CLI chat
python scripts/chat.py --collection my_docs

# Or start API server
python scripts/serve.py
```

## Project Structure

```
rag-chatbot-template/
├── src/rag_chatbot/
│   ├── core/
│   │   ├── config.py           # Configuration management
│   │   ├── embeddings.py       # Embedding models (HuggingFace, OpenAI)
│   │   └── llm.py              # Multi-provider LLM wrapper
│   ├── ingestion/
│   │   ├── loader.py           # Document loaders (PDF, text, web)
│   │   ├── chunker.py          # Text chunking strategies
│   │   └── pipeline.py         # Ingestion orchestration
│   ├── retrieval/
│   │   ├── qdrant_store.py     # Qdrant vector store
│   │   ├── retrievers.py       # Retrieval strategies
│   │   └── reranker.py         # Cross-encoder reranking
│   ├── generation/
│   │   ├── rag_chain.py        # RAG pipeline
│   │   ├── prompts.py          # Prompt templates
│   │   └── memory.py           # Conversation memory
│   ├── evaluation/
│   │   └── metrics.py          # RAG evaluation metrics
│   └── api/
│       ├── app.py              # FastAPI application
│       ├── routes.py           # API endpoints
│       └── models.py           # Pydantic schemas
├── scripts/
│   ├── ingest.py               # Document ingestion CLI
│   ├── chat.py                 # Interactive chat CLI
│   └── serve.py                # API server
├── configs/
│   ├── config.yaml             # Main configuration
│   └── prompts.yaml            # Prompt templates
├── notebooks/
│   ├── 01_ingestion.ipynb      # Ingestion walkthrough
│   └── 02_retrieval.ipynb      # Retrieval tuning
├── tests/
├── Dockerfile
├── docker-compose.yml
└── pyproject.toml
```

## LLM Providers

```python
from rag_chatbot.core.llm import LLMClient

# OpenAI
client = LLMClient(provider="openai", model="gpt-4o")

# Azure OpenAI
client = LLMClient(provider="azure", model="gpt-4o", api_version="2024-02-01")

# Anthropic Claude
client = LLMClient(provider="anthropic", model="claude-3-5-sonnet-20241022")

# xAI Grok
client = LLMClient(provider="xai", model="grok-beta")
```

## Configuration

```yaml
# configs/config.yaml
embedding:
  model: sentence-transformers/all-MiniLM-L6-v2
  device: auto

qdrant:
  host: localhost
  port: 6333
  collection: documents

retrieval:
  top_k: 5
  score_threshold: 0.7
  rerank: true
  rerank_model: cross-encoder/ms-marco-MiniLM-L-6-v2

generation:
  provider: openai
  model: gpt-4o
  temperature: 0.7
  max_tokens: 1000

chunking:
  strategy: recursive
  chunk_size: 512
  chunk_overlap: 50
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/chat` | Send message and get response |
| POST | `/ingest` | Ingest documents |
| GET | `/collections` | List collections |
| DELETE | `/collections/{name}` | Delete collection |
| GET | `/health` | Health check |

### Chat Request

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What is RAG?",
    "collection": "my_docs",
    "conversation_id": "abc123"
  }'
```

## Evaluation

Built-in metrics for RAG quality:

```python
from rag_chatbot.evaluation import RAGEvaluator

evaluator = RAGEvaluator()
results = evaluator.evaluate(
    questions=["What is X?", "How does Y work?"],
    ground_truth=["X is...", "Y works by..."],
    collection="my_docs"
)

print(f"Faithfulness: {results['faithfulness']:.2f}")
print(f"Relevance: {results['relevance']:.2f}")
print(f"Answer Correctness: {results['correctness']:.2f}")
```

## Chunking Strategies

This template supports multiple chunking strategies for different document types:

| Strategy | Best For | Description |
|----------|----------|-------------|
| **Recursive** | General text | Splits on separators (paragraphs → sentences → words) |
| **Sentence** | Structured docs | Preserves sentence boundaries using NLTK |
| **Semantic** | Mixed content | Splits where embedding similarity drops |
| **LLM-based** | Complex docs | Uses LLM to identify logical boundaries |

```python
from rag_chatbot.ingestion import get_chunker

# Recursive (default, fast)
chunker = get_chunker("recursive", chunk_size=512, chunk_overlap=50)

# Sentence-based (preserves meaning)
chunker = get_chunker("sentence", chunk_size=512, chunk_overlap=1)

# Semantic (embedding-aware)
chunker = get_chunker("semantic", embedding_model=embed_model, threshold=0.7)

# LLM-based (most intelligent, slowest)
chunker = get_chunker("llm", llm_client=llm, max_chunk_size=1000)
```

### Chunking Best Practices

- **Technical docs**: Use recursive with 512-1024 chunk size
- **Legal/medical**: Use sentence chunker to preserve context
- **Mixed content**: Use semantic chunker with similarity threshold 0.6-0.8
- **Long documents**: Combine LLM chunker for structure + recursive for sections

---

## Reranking

Cross-encoder reranking significantly improves retrieval quality by re-scoring retrieved documents:

```python
from rag_chatbot.retrieval import CrossEncoderReranker

# Initialize reranker
reranker = CrossEncoderReranker(
    model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",  # Fast, good quality
    # model_name="BAAI/bge-reranker-large",  # Slower, better quality
    top_k=3,
)

# Retrieve more, rerank to top 3
docs = retriever.retrieve(query, top_k=10)
reranked_docs = reranker.rerank(query, docs, top_k=3)
```

### Reranker Models

| Model | Speed | Quality | Use Case |
|-------|-------|---------|----------|
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | Fast | Good | Production, low latency |
| `cross-encoder/ms-marco-MiniLM-L-12-v2` | Medium | Better | Balanced |
| `BAAI/bge-reranker-base` | Medium | Better | Multilingual |
| `BAAI/bge-reranker-large` | Slow | Best | Quality-critical |

---

## RAG Best Practices

### 1. Retrieval Quality

- **Hybrid search**: Combine dense (embeddings) + sparse (BM25) retrieval
- **Query expansion**: Generate query variations with LLM
- **Metadata filtering**: Pre-filter by date, source, category
- **Over-retrieve + rerank**: Fetch 3-5x candidates, rerank to final set

### 2. Chunking Optimization

- **Chunk size**: 256-512 for precise retrieval, 512-1024 for context
- **Overlap**: 10-20% overlap prevents losing context at boundaries
- **Metadata enrichment**: Add source, page number, section headers
- **Document structure**: Preserve headings, lists, tables as metadata

### 3. Prompt Engineering

- **System prompts**: Define assistant persona and constraints
- **Few-shot examples**: Include 1-2 examples of desired output format
- **Source citation**: Instruct LLM to cite [1], [2] from context
- **Fallback handling**: Define behavior when context is insufficient

### 4. Production Considerations

- **Caching**: Cache embeddings, cache frequent queries
- **Rate limiting**: Implement backoff for LLM API calls
- **Monitoring**: Track retrieval quality, answer faithfulness
- **A/B testing**: Compare chunking strategies, prompt variations

### 5. Common Pitfalls

- ❌ Chunks too large → retrieves irrelevant content
- ❌ Chunks too small → loses context
- ❌ No reranking → noisy retrieval hurts generation
- ❌ Ignoring metadata → misses filtering opportunities
- ❌ Single retrieval strategy → misses edge cases

---

## Docker Deployment

```bash
# Build
docker build -t rag-chatbot:latest .

# Run with Qdrant
docker-compose up -d
```

## Requirements

- Python 3.10+
- Qdrant (local or cloud)
- API key for at least one LLM provider

## License

MIT License - see [LICENSE](LICENSE)

## Author

**Hasan Halacli** - [Website](https://www.halacli.com/) · [GitHub](https://github.com/hasanhalacli)
