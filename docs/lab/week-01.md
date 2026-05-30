# Week 01 Lab Report

**Date**: 2026-05-18 to 2026-05-31
**Phase**: 1 (weeks 1-2 — pilot retrieval prototype)
**Hours spent**: ~10 hrs

## What I Built

The end-to-end pilot retrieval pipeline (`feat/pilot-retrieval`), deliberately minimal:

- `packages/data-pipeline/src/ingest_ads.py` — NASA ADS abstract ingestion (httpx against
  the ADS search API), writes `data/raw/.../abstracts.jsonl` + a provenance `manifest.json`.
- `packages/rag/src/index_corpus.py` — embeds abstracts with **BGE-small-en-v1.5 (384-dim)**,
  loads dense vectors into **pgvector (HNSW)** and builds a **SQLite FTS5** BM25 index.
- `packages/rag/src/pilot_retrieval.py` — hybrid retrieval: dense + lexical fused with
  **reciprocal-rank fusion (RRF, k=60)**. Also the retrieval CLI (query → ranked bibcodes).
- `packages/evaluation/src/eval_retrieval.py` — **Recall@10** and **MRR** over a query YAML.
- Infra: a separate `docker/docker-compose.pilot.yml` (pgvector on **:5433**), `init.sql`
  updated to **vector(384)** + ADS `bibcode`/`year` columns (STEP 0).
- `tests/fixtures/pilot/` — a 14-doc synthetic corpus + 13-query set for an offline smoke test.
- `packages/evaluation/queries/pilot_exoplanet_atmospheres.yaml` — **30 candidate queries**
  (named targets + molecules + processes) drafted for review.

## The Number

Offline machinery smoke test (14 synthetic abstracts, 13 queries):

> **Recall@10 = 1.000, MRR = 1.000**

This is **saturated by design** — every fixture query names its target, so it only proves the
plumbing (embed → pgvector → FTS5 → RRF → metrics) is correct, including the multi-relevant
case. **It is not a retrieval-quality result.** The real number — 500 ADS abstracts vs. the
30-query set — is blocked on an ADS key (below).

## What I Learned

- BGE retrieval wants an **asymmetric** setup: queries get the "Represent this sentence…"
  instruction prefix, passages don't. Wired that into the embedder helpers.
- FTS5 chokes on hyphenated identifiers (`WASP-39b`, `TRAPPIST-1`); sanitizing the query into
  an OR of double-quoted tokens avoids syntax errors and maximizes lexical recall — fusion
  restores precision.
- RRF is refreshingly dependency-free: rank-only fusion means dense cosine and BM25 scores
  never have to be normalized against each other.

## Observations

- **Docker bit me hard.** Both this repo's pilot compose and an unrelated local project's
  compose live in a `docker/` dir, so Compose derived the **same project name `docker`** for
  both. My first `up` recreated the *other* project's DB container onto my volume. Recovered
  by rebinding their real data volume (`docker_pg_data`, 33 tables) and isolating mine under
  `-p astrollm-pilot`. Lesson baked into the compose header: **always pass an explicit `-p`.**
- The fixture-saturates-metrics effect is a good reminder that a number without a real,
  confusable corpus is theater. The 30-query set is built to have genuine near-misses.

## Blockers / Questions

- **No `ADS_API_KEY`** (no `.env`, nothing exported) → cannot ingest the real 500 abstracts →
  cannot ground `expected_bibcodes` → the real Recall@10/MRR is pending. The code path is
  ready; drop a token in `.env` and `ingest_ads.py` → `index_corpus.py` → `eval_retrieval.py`
  produces the real number. Get a free token at ui.adsabs.harvard.edu/user/settings/token.
- Open: is `abs:"exoplanet atmosphere"` (2018+) the right slice, or should I widen the query?

## Next Week

- Obtain ADS token, ingest ~500 abstracts, **review/edit the 30-query set** and fill
  `expected_bibcodes`, then report the real Recall@10/MRR.
- Sanity-check dense vs. lexical contribution (ablate one arm of the RRF).
- Still **out of scope** until weeks 5-10: SPECTER2, cross-encoder reranking, SIMBAD alias
  expansion, full-text fetch, section-aware chunking.

## Reading Log

| Material | Sections | Notes |
|----------|----------|-------|
| BAAI BGE model card | Usage / retrieval | Query instruction prefix; normalize embeddings for cosine |
| NASA ADS API docs | search/query | `fl`, `fq=year:[…]`, ≤200 rows/page, Bearer token |
| Cormack et al. 2009 | RRF | Rank-only fusion, k=60 default — no score normalization needed |
| pgvector README | HNSW | `vector_cosine_ops`, `<=>` cosine distance operator |
