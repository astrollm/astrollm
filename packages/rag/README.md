# RAG Package

Retrieval-Augmented Generation system for AstroLLM.

## Architecture

```
Query → Embed → Hybrid Search (semantic + keyword) → Rerank → Top-k → Inject into prompt
```

## Components

- **Embedding**: BGE-large-en-v1.5 via HF Text Embeddings Inference
- **Vector Store**: PostgreSQL + pgvector (cosine similarity)
- **Full-Text**: PostgreSQL ts_vector (keyword search)
- **Hybrid**: 70/30 semantic/keyword weighting (tunable)
- **Reranking**: Cross-encoder on top-k for precision (optional, Phase 3+)
- **Chunking**: Section-aware (respects paper structure)

## Key Design Decisions

1. **pgvector over Pinecone/Weaviate**: PostgreSQL is already in our stack, pgvector avoids another service, and we need relational queries on paper metadata alongside vector search.

2. **Section-aware chunking**: Don't split across abstract/intro/methods/results/conclusion boundaries. Each chunk knows which section it came from, enabling targeted retrieval.

3. **Metadata filtering**: Filter by date, topic, citation count before vector search to reduce noise.

## Setup

```bash
# Start PostgreSQL with pgvector
docker compose -f docker/docker-compose.yml up -d db embedder

# Ingest papers
python src/ingest.py --input data/processed/ --batch-size 100

# Test retrieval
python src/search.py --query "stellar mass black holes in binary systems"
```
