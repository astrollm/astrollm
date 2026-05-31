# Week 01 Lab Report

**Date**: 2026-05-30 to 2026-05-31
**Phase**: 1 (weeks 1-2 — pilot retrieval prototype)

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

Real run — 500 ADS abstracts (`abs:"exoplanet atmosphere"`, 2018+, of 4,845 matches), scored
over 16 grounded queries:

> **Recall@10 = 0.812 · MRR = 0.776**

Not saturated — there are real misses and low ranks, which is what makes it useful:

- **q12 (WASP-96b cloud-free water) = 0.00.** The relevant JWST paper's title is the generic
  "The JWST Early Release Observations" (no "WASP-96b"), so neither BM25 nor dense matched the
  query well — a real recall miss, and possibly also a labeling nuance to revisit.
- **q03/q05/q06/q18 = 0.50** — found one of two expected papers.
- **q07/q08 = 1.00 recall but RR 0.17/0.25** — the relevant paper is retrieved but ranked deep
  (HD 189733b/HD 209458b queries pull many same-target papers; the labeled one isn't top).

Caveats: `expected_bibcodes` are my **first-pass, title-level labels** (entity-grounded,
rank-independent) and need review; 16 of 30 queries are scored (q11/q15 had no clear corpus
target, q19-q30 are broad-topic and left for manual labeling). The number will move after review.

For reference, the offline synthetic fixture (14 docs) still scores 1.000/1.000 — saturated by
design, used only as a machinery smoke test (embed → pgvector → FTS5 → RRF → metrics).

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

- ~~No `ADS_API_KEY`~~ — **resolved.** Token added; ingested 500 abstracts and produced the
  real number above. End-to-end path confirmed: `ingest_ads.py` → `index_corpus.py` →
  `eval_retrieval.py`.
- **Needs review:** the first-pass `expected_bibcodes` labels (esp. q12's 0.00 — is the generic
  "JWST Early Release Observations" paper the right WASP-96b target, or is there a better one?),
  and labeling the 14 unscored queries (q11, q15, q19-q30).
- Open: is `abs:"exoplanet atmosphere"` (2018+) the right slice, or should I widen the query?
  Several famous targets' canonical papers (HD 189733b sodium, WASP-121b inversion) predate 2018
  and so aren't in the corpus.

## Next Week

- **Domain review of the first-pass `expected_bibcodes` labels** — start with the outliers
  (q12's 0.00 miss, then q07/q08 which are retrieved but ranked deep), then the four 0.50
  queries (q03, q05, q06, q18), then label the currently-empty q11, q15, and q19–q30.
  Tracked as a follow-up PR.
- **Ablation**: dense-only vs. lexical-only vs. hybrid (RRF) on the 500-doc corpus, to
  quantify each arm's contribution to the number.
- **Decide whether to widen the ADS slice** / drop the 2018 cutoff to recover pre-2018
  canonical papers (HD 189733b sodium, WASP-121b inversion, etc.).
- Still **out of scope** until weeks 5-10: SPECTER2, cross-encoder reranking, SIMBAD alias
  expansion, full-text fetch, section-aware chunking.

## Reading Log

| Material | Sections | Notes |
|----------|----------|-------|
| BAAI BGE model card | Usage / retrieval | Query instruction prefix; normalize embeddings for cosine |
| NASA ADS API docs | search/query | `fl`, `fq=year:[…]`, ≤200 rows/page, Bearer token |
| Cormack et al. 2009 | RRF | Rank-only fusion, k=60 default — no score normalization needed |
| pgvector README | HNSW | `vector_cosine_ops`, `<=>` cosine distance operator |
