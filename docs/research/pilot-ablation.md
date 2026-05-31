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

### Aggregate — Recall@10 / MRR by arm and split

Best per column in **bold**.

| arm | named R@10 | named MRR | broad R@10 | broad MRR | all R@10 | all MRR |
|---|---|---|---|---|---|---|
| dense | 0.794 | 0.785 | **0.625** | 0.374 | **0.724** | 0.615 |
| lexical | **0.804** | 0.765 | 0.583 | 0.317 | 0.713 | 0.580 |
| hybrid | 0.794 | **0.794** | 0.542 | **0.381** | 0.690 | **0.623** |

**Harness validation:** the hybrid row reproduces the PR #4
[label-review](pilot-label-review.md) baseline *exactly* — named 0.794/0.794, broad 0.542/0.381,
all 0.690/0.623 — confirming nothing but the arm changed.

**Two clean patterns:** hybrid wins **every MRR split** (it ranks the first relevant doc highest);
dense wins **every recall split** (all-scored and broad), with lexical's lone recall win on the
named set driven entirely by q12. Hybrid has the **lowest** all-scored Recall@10 (0.690 vs dense
0.724).

### Per-query (all scored, raw)

`R@10 (rank)` = recall@10 and the rank of the first gold hit within the top-10 (`—` = miss, so
RR = 0). `gold` = size of the labeled relevant set.

| id | target | kind | gold | dense R@10 (rank) | lexical R@10 (rank) | hybrid R@10 (rank) |
|---|---|---|---|---|---|---|
| q01 | WASP-39b | named | 3 | 1.00 (#1) | 1.00 (#1) | 1.00 (#1) |
| q02 | WASP-39b | named | 1 | 1.00 (#1) | 1.00 (#1) | 1.00 (#1) |
| q03 | K2-18b | named | 2 | 0.50 (#1) | 0.50 (#1) | 0.50 (#1) |
| q04 | K2-18b | named | 2 | 1.00 (#1) | 1.00 (#1) | 1.00 (#1) |
| q05 | TRAPPIST-1e | named | 2 | 0.50 (#7) | 0.50 (#4) | 0.50 (#2) |
| q06 | TRAPPIST-1b | named | 1 | 1.00 (#1) | 1.00 (#1) | 1.00 (#1) |
| q07 | HD 189733b | named | 2 | 1.00 (#1) | 0.50 (#2) | 1.00 (#1) |
| q08 | HD 209458b | named | 3 | 1.00 (#1) | 0.67 (#1) | 1.00 (#2) |
| q09 | 55 Cancri e | named | 2 | 1.00 (#1) | 1.00 (#1) | 1.00 (#1) |
| q10 | GJ 1214b | named | 2 | 1.00 (#1) | 1.00 (#2) | 1.00 (#1) |
| q11 | WASP-121b | named | 1 | 0.00 (—) | 0.00 (—) | 0.00 (—) |
| q12 | WASP-96b | named | 1 | 0.00 (—) | **1.00 (#4)** | 0.00 (—) |
| q13 | LTT 9779b | named | 1 | 1.00 (#1) | 1.00 (#1) | 1.00 (#1) |
| q14 | GJ 486b | named | 2 | 1.00 (#1) | 1.00 (#1) | 1.00 (#1) |
| q16 | TOI-270d | named | 1 | 1.00 (#1) | 1.00 (#2) | 1.00 (#1) |
| q17 | WASP-17b | named | 1 | 1.00 (#1) | 1.00 (#1) | 1.00 (#1) |
| q18 | WASP-43b | named | 2 | 0.50 (#5) | 0.50 (#1) | 0.50 (#2) |
| q19 | CO2 | broad | 1 | 1.00 (#1) | 1.00 (#2) | 1.00 (#1) |
| q20 | SO2 | broad | 1 | 1.00 (#1) | 1.00 (#1) | 1.00 (#1) |
| q21 | H2O | broad | 2 | 0.00 (—) | 0.00 (—) | 0.00 (—) |
| q22 | CH4 | broad | 1 | 0.00 (—) | 1.00 (#8) | 1.00 (#10) |
| q23 | CO | broad | 1 | 1.00 (#2) | 1.00 (#1) | 1.00 (#2) |
| q24 | Na | broad | 2 | **0.50 (#7)** | **0.50 (#4)** | 0.00 (—) |
| q25 | escape | broad | 2 | 1.00 (#1) | 0.50 (#7) | 0.50 (#2) |
| q26 | clouds | broad | 2 | 0.50 (#6) | 0.50 (#3) | 0.50 (#3) |
| q27 | retrieval | broad | 2 | 1.00 (#4) | 1.00 (#3) | 1.00 (#1) |
| q28 | C/O ratio | broad | 2 | **0.50 (#7)** | **0.50 (#8)** | 0.00 (—) |
| q29 | emission | broad | 2 | **0.50 (#7)** | 0.00 (—) | 0.00 (—) |
| q30 | phase curve | broad | 2 | 0.50 (#7) | 0.00 (—) | 0.50 (#7) |

### Decisive queries — arms disagree on top-10 hit/miss

`topN` = first-gold rank in that arm's scored top-10; `full` = the gold paper's rank in the arm's
*entire* 500-doc ranking (pool-independent diagnostic — shows where the signal actually lives).

| id | target | dense (top10 / full) | lexical (top10 / full) | hybrid (top10) |
|---|---|---|---|---|
| q12 | WASP-96b | miss / #338 of 500 | #4 / #4 of 500 | miss |
| q22 | CH4 | miss / #21 of 500 | #8 / #8 of 500 | #10 |
| q24 | Na | #7 / #7 of 500 | #4 / #4 of 500 | miss |
| q28 | C/O ratio | #7 / #7 of 500 | #8 / #8 of 500 | miss |
| q29 | emission | #7 / #7 of 500 | miss / #15 of 500 | miss |
| q30 | phase curve | #7 / #7 of 500 | miss / #15 of 500 | #7 |

Per-paper detail behind the two-arm losses (gold paper → dense rank / lexical rank in the full
500-doc ranking; **bold** = inside the pool of 50, plain = outside it):

- **q24 (Na):** `2020A&A...635A.206C` → dense **#7** / lexical #87; `2018A&A...612A..53P` → dense #89
  / lexical **#4**. Each relevant paper is strong in *one* arm and outside the *other* arm's pool,
  so each earns a single weak RRF term and hybrid buries both.
- **q28 (C/O):** `2021ApJ...914...12L` → dense **#7** / lexical **#41**; `2024ApJ...963L...5X` →
  dense #54 / lexical **#8**. Same split: the relevance is divided single-arm-strong across arms.

## Interpretation

### H1 — confirmed

Lexical-only beats hybrid on single-arm-strong misses. **q12 (WASP-96b):** lexical retrieves the
ERO at **#4** (recall 1.00) while hybrid misses it entirely. The mechanism is exactly the one
predicted and documented in the label review: the ERO's abstract is generic outreach text, so it
embeds far from the query (**dense #338**) while matching lexically; RRF with a pool of 50 gives the
paper a single weak term and ranks both-arms-mediocre papers above it. **q24 (Na)** is a second
clean case — lexical #4, hybrid miss. Lexical-only strictly recovers what hybrid dilutes.

### H2 — confirmed (modest margin)

Dense-only beats lexical-only on the broad/conceptual set in **both** metrics: Recall@10 0.625 vs
0.583, MRR 0.374 vs 0.317. The margin is small (≈one query of recall), but the direction is as
predicted and consistent. Conceptual queries have no entity string for BM25 to anchor on, so lexical
ranks the landmark lower or misses it — e.g. q25 (escape) dense #1 vs lexical #7, q29 (emission)
dense #7 vs lexical miss, q30 (phase curve) dense #7 vs lexical miss. Dense's semantic match anchors
better. Caveat: broad labels are known-item (optimistic), so treat the broad numbers as directional.

### H3 — partially confirmed; this is the headline

The "**strictly beaten by a single arm on tail queries**" clause is confirmed, strongly — and more
strongly than predicted. Hybrid is beaten by lexical on q12, by dense on q29, and on **q24 and q28
by _both_ single arms at once**: each relevant paper is single-arm-strong in a *different* arm (and
outside the other arm's pool), so RRF hands each a lone weak term and ranks generic
both-arms-mediocre papers above them. Each single arm recovers one of the two; fusion recovers
neither.

The "**hybrid is best on average**" clause is **refuted on Recall@10 and confirmed only on MRR.**
Hybrid wins every MRR split (named 0.794, broad 0.381, all 0.623 — all firsts) but has the **lowest**
all-scored Recall@10 (0.690), below dense (0.724) and lexical (0.713). So the honest result: on this
frozen pilot, hybrid RRF buys **ranking quality** (it places a relevant doc highest) at a measurable
**recall cost** (−0.034 vs dense overall). The folk intuition "fusion is best on average" holds for
*where it ranks the first hit*, not for *whether it finds everything*.

**Why:** RRF's failure mode is split single-arm-strong relevance. When a relevant paper is strong in
only one arm and outside the other arm's pool, fusion reduces it to a single ~`1/(60+rank)` term —
easily out-scored by papers that rank merely *decently in both* arms (two terms). The pilot corpus
has enough such cases (q12, q24, q28, plus partials) to pull hybrid's recall below dense's. Note
this is a **fusion** problem, not a **pool** problem: q12's ERO at dense #338 is unreachable by any
sane pool, so widening the pool would not fix it (the q12 caveat above, now demonstrated).

### Takeaways

- **Diagnosis over verdict.** The useful output is not "arm X wins" (the aggregate gaps are ≤0.034)
  but the reproducible failure mode: hybrid RRF loses recall precisely on split single-arm-strong
  relevance. The per-query and decisive-query tables localize every instance.
- **If forced to pick one default on this corpus:** dense-only maximizes recall (find the paper at
  all), hybrid maximizes MRR (rank it first). The choice follows the product's recall-vs-ranking
  weighting; the difference is small either way.
- **Follow-ups (out of scope here — single-variable ablation):** (1) score-aware fusion that does
  not discard a strong single-arm signal (weighted RRF, or max-of-normalized-scores); (2) a reranker
  over the union of both arms' top-k; (3) section-aware / full-text chunking so multi-target release
  papers (q12) embed near their per-target science and dense recall improves — the same chunking
  work the label review flagged for weeks 5–10. A **pool sweep is explicitly not** a fix for the
  q12-class miss and is a separate, optional experiment.

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
