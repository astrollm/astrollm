---
globs: ["packages/data-pipeline/**", "packages/rag/**"]
---

# Data Pipeline & RAG Rules

- Raw data goes in `data/raw/` — never modified after download
- Processed data goes in `data/processed/` — versioned by pipeline hash
- SFT datasets in `data/sft/` — JSONL format, validated against `data/sft/schema.json`
- Every dataset must have a `manifest.json` with provenance metadata (source, date, pipeline version)
- Astronomy tool access via `astroquery` (SIMBAD, ADS, NED, VizieR) — already in deps
- NASA ADS requires API key from `.env` (`ADS_API_KEY`)
- Embedding model must match the dimension in `docker/init.sql` (currently 1024 for BGE-large)
- pgvector uses HNSW indexing — no IVFFlat
- Chunk boundaries must respect paper section structure (don't split mid-section)
- Max chunk size: 512 tokens with 50-token overlap
