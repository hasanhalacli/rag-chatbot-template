# RAG Chatbot Template

[![gates](https://github.com/hasanhalacli/rag-chatbot-template/actions/workflows/gates.yml/badge.svg)](https://github.com/hasanhalacli/rag-chatbot-template/actions/workflows/gates.yml)
[![Python 3.10–3.13](https://img.shields.io/badge/python-3.10%E2%80%933.13-blue.svg)](https://www.python.org/downloads/)
[![Qdrant](https://img.shields.io/badge/Qdrant-vector%20DB-red.svg)](https://qdrant.tech/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

Production-ready RAG (Retrieval-Augmented Generation) chatbot template. **No LangChain, no LlamaIndex** - just clean, deployment-friendly Python code.

## Why No LangChain/LlamaIndex?

This template is intentionally built without heavy frameworks:

| Aspect | With Frameworks | This Template |
|--------|-----------------|---------------|
| Dependencies | 50+ packages | ~15 packages |
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
- **Gated by default**: every dependency pinned and locked; secret scan, vulnerability audit and tests run on every pull request

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
docker compose up -d qdrant
```

### Ingest, retrieve, answer

```python
from rag_chatbot.core import EmbeddingModel, LLMClient
from rag_chatbot.generation import RAGChain
from rag_chatbot.ingestion import get_chunker
from rag_chatbot.retrieval import Document, QdrantStore, Retriever

embedder = EmbeddingModel("sentence-transformers/all-MiniLM-L6-v2")
store = QdrantStore(host="localhost", port=6333, embedding_dim=embedder.embedding_dim)
store.create_collection("my_docs")

text = open("data/handbook.txt").read()
chunks = get_chunker("recursive", chunk_size=512, chunk_overlap=50).chunk(text, {"source": "handbook"})
docs = [Document(content=c.text, metadata=c.metadata) for c in chunks]
store.add_documents("my_docs", docs, embedder.embed([d.content for d in docs]).tolist())

chain = RAGChain(
    retriever=Retriever(store, embedder, collection="my_docs", top_k=5),
    llm_client=LLMClient(provider="openai", model="gpt-4o"),   # key from OPENAI_API_KEY
)
response = chain.query("What is the refund policy?", conversation_id="demo")
print(response.answer)
for source in response.sources:
    print("-", source.metadata, source.score)
```

Follow-up questions on the same `conversation_id` are condensed against the history before retrieval.

## Project Structure

```
rag-chatbot-template/
├── src/rag_chatbot/
│   ├── core/
│   │   ├── config.py           # Settings from env vars or YAML
│   │   ├── embeddings.py       # Sentence-transformer embeddings
│   │   └── llm.py              # OpenAI, Azure OpenAI, Anthropic, xAI behind one client
│   ├── ingestion/
│   │   └── chunker.py          # Recursive, sentence, semantic and LLM chunking
│   ├── retrieval/
│   │   ├── qdrant_store.py     # Qdrant collections and upserts
│   │   ├── retrievers.py       # Vector, multi-query and hybrid retrieval
│   │   └── reranker.py         # Cross-encoder reranking
│   └── generation/
│       ├── rag_chain.py        # Retrieve → (rerank) → generate, with memory
│       ├── prompts.py          # Prompt templates
│       └── memory.py           # Buffer, sliding-window and summary memory
├── tests/                      # Unit tests — no model or database needed
├── .github/workflows/gates.yml # Secret scan, pinned-deps check, audit, tests
├── docker-compose.yml          # Local Qdrant
├── pyproject.toml              # Exact versions only
└── uv.lock
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
# config.yaml — load with Settings.from_yaml("config.yaml")
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

llm:
  provider: openai
  model: gpt-4o
  temperature: 0.7
  max_tokens: 1000

chunking:
  strategy: recursive
  chunk_size: 512
  chunk_overlap: 50
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

## Development

```bash
uv sync --extra dev
uv run pytest -q
```

Every pull request runs the same four gates as CI: a full-history secret scan, a check that no
dependency uses a version range, `uv lock --check`, and a vulnerability audit of the locked set —
then the tests. A change lands only through a reviewed pull request. See [AGENTS.md](AGENTS.md)
for the rules that apply to people and coding agents alike.

## Running Qdrant locally

```bash
docker compose up -d qdrant
```

## Requirements

- Python 3.10 – 3.13
- Qdrant (local or cloud)
- API key for at least one LLM provider

## License

MIT License - see [LICENSE](LICENSE)

## Author

**Hasan Halacli** — AI Solution Architect & Technical Lead

[Portfolio](https://www.halacli.com/) · [Blog](https://www.halacli.com/blog.html) · [LinkedIn](https://www.linkedin.com/in/hasan-h-326b5a171/) · [GitHub](https://github.com/hasanhalacli)

> 📖 Related deep-dive: [**Enterprise RAG Chatbot — Architecture, Evaluation & Delivery**](https://www.halacli.com/case-rag-chatbot.html)
