# Pilot Retrieval — Dense vs Lexical vs Hybrid Ablation

Follow-up to [PR #4 label review](pilot-label-review.md). The pilot ships a hybrid retriever
(dense pgvector/BGE + lexical FTS5/BM25, fused with reciprocal-rank fusion). The label review
surfaced recurring **single-arm-strong misses** (q12, q11) where one arm finds the relevant paper
and fusion buries it. This experiment isolates the contribution of each arm: it runs the **same**
gold set over the **same** frozen index three ways — dense-only, lexical-only, hybrid — and reports
Recall@10 and MRR for each.

This is the dense-vs-lexical-vs-hybrid ablation listed in
[week-01's Next Week section](../lab/week-01.md).

## Setup — what is fixed and what varies

**Frozen corpus snapshot (not re-ingested, re-embedded, or re-chunked for this experiment):**

| property | value |
|---|---|
| corpus | 500 ADS abstracts, query `abs:"exoplanet atmosphere"`, `year_start=2018` |
| pipeline | `pilot-retrieval-0.1.0` (naive chunking: one chunk = one abstract) |
| embeddings | BGE-small-en-v1.5, 384-dim, normalized (dense) |
| dense store | pgvector HNSW (`m=32`, `ef_construction=256`), cosine |
| lexical store | SQLite FTS5, BM25, OR-of-tokens match |
| gold set | `packages/evaluation/queries/pilot_exoplanet_atmospheres.yaml` (reviewed in PR #4) |
| scored queries | 29 (q15 has no in-corpus target → unscored) |

**Single-variable discipline.** Exactly one thing changes across the three runs: which arm(s) feed
the final ranking. The corpus snapshot, the embeddings, the labels, `k=10`, the candidate `pool`,
and the RRF constant are identical for all three arms. Concretely, every arm draws candidates from
the same `dense_search` / `lexical_search` over the same index; only the selection/fusion step
differs (see `retrieve(query, arm, k, pool)` in `packages/rag/src/pilot_retrieval.py`):

| arm | ranking |
|---|---|
| **dense** | pgvector/BGE cosine, top-10 |
| **lexical** | FTS5/BM25, top-10 |
| **hybrid** | dense+lexical fused with RRF over the pool, top-10 (the pilot config) |

**Held fixed at the pilot values:** `k = 10`, `pool = 50` (per-arm candidate depth fed to RRF),
`RRF_K = 60`. The `pool` knob is inert for the single arms — their top-10 is the first 10 of the
pool regardless — and is held at 50 so the only moving part is the arm.

**Reported three ways**, because the two query kinds mean different things (per the gold-set header
and the label review):

- **named-target** (q01–q18, n=17): relevance judged by entity match + abstract; approximates the
  in-corpus relevant set, so recall misses are meaningful.
- **broad known-item** (q19–q30, n=12): labeled with 1–2 landmark papers per topic, **not** an
  exhaustive relevant set. Recall@10 here means "did the landmark surface", not true topical recall.
- **all scored** (n=29).

**q12 caveat — arm, not pool.** This ablation tests the *arm*, not pool size. It does **not** sweep
the pool. For q12 specifically a larger pool is **not** expected to help: the WASP-96b ERO sits at
**dense #338**, so reaching it would add only a negligible RRF term — q12 is a dense-retrieval
failure that lexical catches and hybrid dilutes, independent of pool depth. A pool sweep is a
separate, optional, later experiment and is deliberately out of scope here. The per-query tables
below report a *diagnostic* full-corpus rank of the gold paper for the single arms (depth =
|corpus|); this is pool-independent annotation only and does not change any scored hybrid number.

## Predictions (pre-registered)

Registered **before** running the experiment (these are the commission's hypotheses):

- **H1.** Lexical-only beats hybrid on single-arm-strong misses. Concretely, q12's ERO
  (`2022ApJ...936L..14P`) sits at lexical ~#4 but hybrid buries it (dense #338, so RRF dilutes the
  lexical signal below the top-10 cutoff).
- **H2.** Dense-only beats lexical-only on the broad/conceptual set (q19–q30), where there is no
  entity string for BM25 to anchor on.
- **H3.** Hybrid is best on average, but is strictly beaten by a single arm on specific tail
  queries (i.e. fusion wins the aggregate while losing individual single-arm-strong cases).

## Results

<!-- PENDING RUN — filled in the results commit. -->

## Interpretation

<!-- PENDING RUN — filled in the results commit, tied back to H1/H2/H3. -->

## Reproduce

```bash
# Pilot DB up + indexed (frozen 500-abstract corpus), then:
uv run python packages/evaluation/src/ablation_retrieval.py \
    --queries packages/evaluation/queries/pilot_exoplanet_atmospheres.yaml \
    --json-out docs/research/pilot-ablation-results.json

# Or evaluate a single arm directly:
uv run python packages/evaluation/src/eval_retrieval.py \
    --queries packages/evaluation/queries/pilot_exoplanet_atmospheres.yaml --arm dense
```

Raw per-query results (every arm, every query, plus the diagnostic full-corpus ranks) are written
to `docs/research/pilot-ablation-results.json`.
