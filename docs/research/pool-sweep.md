# Pool-Depth Sweep — Fusion Bottleneck vs Candidate Depth (frozen index)

Follow-up to the [corpus-widening ablation](corpus-widening.md). That PR found H5: widening the
corpus to 2,500 docs at a fixed pool=50 hurts recall on the original queries, while the candidate
**union** at pool=50 still held 0.948 of the gold — yet the fused top-k surfaced much less. So the
degradation is at least partly a **fusion-demotion** problem, not candidate absence. This experiment
isolates that: sweep **only** the per-arm candidate **pool depth** and watch whether deepening the
pool closes the gap between what the candidate set contains and what fusion surfaces.

This is a **frozen-index ablation**, like the original [pilot ablation](pilot-ablation.md) and the
opposite of the widening PR: **no re-ingest, no re-embed, no re-index.** The 2,500-abstract index
built in PR #6 (corpus frozen at `main` `d84f3ff`) is queried as-is at four pool depths.

## Setup — what is fixed and what varies

### The swept variable: per-arm candidate pool depth

`pool ∈ {50, 100, 200, 500}`. The pool is the per-arm candidate depth handed to RRF; at each depth
the fusion/selection step re-runs over the deeper candidate sets.

### Single-variable rule — the controlled variable is POOL

Held fixed and identical across all depths:

| held fixed | value |
|---|---|
| corpus | the **frozen** 2,500-abstract index from PR #6 (`main` `d84f3ff`) — not re-touched |
| dense model | BGE-small-en-v1.5, 384-dim, normalized |
| stores | pgvector HNSW (`m=32`, `ef_construction=256`) cosine; SQLite FTS5 BM25 |
| chunking | naive — one chunk = title + abstract |
| arms | dense / lexical / hybrid (RRF) — same fusion/selection code |
| scored cuts | `k = 10` and a deep cut of `50`, **fixed** (decoupled from pool — see below) |
| RRF constant | `RRF_K = 60` |
| labels | the original 29-label gold, **unchanged** |

The fusion primitives (`dense_search`, `lexical_search`, `_rrf_fuse`, `RRF_K`) are **imported** by
the sweep harness from `pilot_retrieval`, and the metric/bootstrap helpers from
`ablation_retrieval`, so neither the controlled variable nor the readouts can drift. The harness
**reproduces the PR #6 pool=50 all-slice numbers exactly** as a built-in cross-check (dense R@10
0.529, lexical 0.437, hybrid 0.592; hybrid R@50 0.793; union 0.948).

**Why a new harness and not `ablation_retrieval --pool`.** The H8 test needs the **fused recall at
fixed cuts** (top-10, top-50) while the **candidate pool varies**. `ablation_retrieval` ties its
deep cut to `pool` (it reports recall@pool), so it cannot express "fused top-50 recall at pool=200".
`pool_sweep.py` decouples the scored cut (10/50, fixed) from the candidate depth (swept), which is
exactly what makes the candidate-union-vs-fused-top-k gap measurable.

### Pre-registered confound — complementarity is depth-dependent by construction

Union recall and the dense-only / lexical-only / both / neither counts **grow with pool by
construction** (a deeper pool admits more candidates into each arm's set). Compare these
**within a depth** and read the **trend**; do **not** cross-compare raw union sizes across depths as
if the method had changed. The controlled comparison is fused-top-k vs the union *at the same depth*.

### Cost proxy appropriate to the current pipeline

The current stage-1 pipeline is `hybrid recall → RRF fusion → top-k`, with **no cross-encoder**. At
this stage pool cost is modest, so the cost proxy reported here is the **mean fused candidate-set
size** (|union of both arms' top-pool|) — the number of candidates fusion ranks, and the input size
a future stage-2 reranker would face. It is **not** a wall-clock absolute, which would be misleading
at this stage. See the [stage-2 boundary note](#h10-operating-point-lever-call).

## Known-item relabel protocol (committed a priori, before scoring)

This is the deferred scored stratified report (b) from PR #6, done without post-hoc bias. The 29
scored queries are partitioned by an **a priori, ranking-blind rule** — fixed in `pool_sweep_known_item.yaml`
**before** the sweep was scored:

> **KNOWN-ITEM (scored):** the named-target queries **q01–q18**. The gold is the faithful canonical
> landmark set for a specific planet + result — closed and stable under corpus growth (newer papers
> are follow-ups, not a new "canonical landmark for THIS question"), so Recall@k is true recall of
> the canonical result and `|relevant| ≪ k`. (q15 has no in-corpus target → unscored.)
>
> **BROAD-ENTITY (measured):** the topical molecule/process queries **q19–q30**. The gold is a
> **non-exhaustive** landmark proxy for an open relevant set that scales with the corpus (the pilot's
> own caveat: "Recall@10 means did the landmark surface, not true topical recall"). On a 5× corpus
> the proxy is even less representative, so Recall@k is reported as a **coverage measurement**, not
> recall, under the criterion: **Recall@k is reported only where `|relevant| ≪ k`.**

The split **is** the pilot gold-set header's named-target vs broad/topical distinction (fixed in
PR #3/#4), applied verbatim — a pre-existing, ranking-blind, dated source. The committed landmark
bibcodes are the **frozen gold** (the harness asserts they equal `pilot_exoplanet_atmospheres.yaml`,
so this is an audit record, not a relabel; the original 29 labels are unchanged). Provenance: rule
from the pilot gold-set header (PR #3/#4); bibcodes from the PR #4 abstract review; committed by
Claude on 2026-05-31 before running the sweep; source ADS abstracts. Partition: **17 known-item
scored** (q01–q14, q16–q18), **12 broad-entity measured** (q19–q30), **1 unscored** (q15).

*Terminology note: the pilot labeled the topical queries with known-item LANDMARKS as a coping
strategy for their open relevant sets; under the `|relevant| ≪ k` criterion those are exactly the
BROAD-ENTITY (measured) queries here. The KNOWN-ITEM scored partition is the named-target set.*

## Predictions (pre-registered)

Registered **before** the retrieval readouts below (continuing the H1–H7 audit trail;
[corpus-widening.md](corpus-widening.md) ended at H7).

- **H8 — pool depth vs the fusion bottleneck (the real test).** At each pool depth, report BOTH
  candidate-union recall AND fused top-10 / top-50 recall, plus the **gap** (the fusion leak).
  Falsifiable prediction: deeper pool **raises candidate-union recall** (near-certain — this leg is
  expected by construction) but fused-top-k recall improves **much less**, i.e. the fusion gap does
  **not** close with depth, because the bottleneck is **RRF demotion**, not candidate presence. If
  confirmed → the next lever is **fusion / stage-2 reranking**, not pool depth. If falsified
  (fused-top-k tracks the candidate gain) → **pool depth is a cheap fix.** Either way is
  decision-relevant.
- **H9 — fragile vs robust edge under depth.** As pool deepens, fusion matters less and per-query
  edges compress. Prediction: the **fragile** dense−hybrid R@10 edge (4-query, LOO-collapsing in #6)
  **vanishes entirely** at depth; the **robust** lexical−hybrid edge (7-query, LOO-surviving)
  **persists longer**. Report dense−hybrid AND lexical−hybrid paired-bootstrap CIs at **each** depth,
  **each with leave-one-out** — never a bare CI (n=29 makes single-point CIs fragile, as #6 showed).
- **H10 — operating point + lever call.** Given H8, recommend either a production pool depth (if
  pool helps) or flag fusion / reranking as the required next lever (if the gap doesn't close).
  **Stated boundary:** at the current stage-1 pipeline (hybrid recall → fusion → top-k, no
  cross-encoder) pool cost is modest; the binding latency cost arrives when **stage-2 reranking
  exists** and pool becomes the reranker's input size. So PR #7 fixes the **recall headroom**; the
  full latency tradeoff is revisited when reranking lands. Cost is reported as a proxy
  (candidates fused), not a wall-clock absolute.

## Results

> **Pending the retrieval readouts** (recorded in the next commit, after this pre-registration is
> committed — mirroring the H1–H7 audit trail). The headline is report (a): all 29 method-fixed,
> R@10/R@50 + candidate-union + fused-top-k + complementarity across depths, with the union-vs-fused
> gap curve as the H8 centerpiece. Report (b) shows the stratified known-item (scored) and
> broad-entity (measured) partitions alongside. Per-depth paired CIs with LOO test H9.

## Reproduce

```bash
# Frozen 2,500 index from PR #6 must be up (no re-index in this PR). Then:
uv run python packages/evaluation/src/pool_sweep.py \
    --queries packages/evaluation/queries/pilot_exoplanet_atmospheres.yaml \
    --known-item packages/evaluation/queries/pool_sweep_known_item.yaml \
    --json-out docs/research/pool-sweep-results.json
```

Raw per-pool, per-query results (every arm, every depth, complementarity, bootstrap+LOO) are written
to `docs/research/pool-sweep-results.json`.
