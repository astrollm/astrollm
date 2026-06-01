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

Harness cross-check passed: **pool=50 reproduces the PR #6 all-slice numbers exactly** (dense R@10
0.529, lexical 0.437, hybrid 0.592; hybrid R@50 0.793; union 0.948), so only the pool varies.

### (a) All 29 — candidate-union vs fused-top-k gap across depths (H8 centerpiece)

| pool | union R@pool | fused R@10 | fused R@50 | gap@10 (union−fused10) | gap@50 (union−fused50) |
|---|---|---|---|---|---|
| 50 | 0.948 | 0.592 | 0.793 | +0.356 | +0.155 |
| 100 | 0.966 | 0.592 | 0.862 | +0.374 | +0.103 |
| 200 | 0.966 | 0.592 | 0.828 | +0.374 | +0.138 |
| 500 | 1.000 | 0.592 | 0.845 | +0.408 | +0.155 |

The candidate union climbs with depth (0.948 → **1.000**), but **fused R@10 is exactly flat at
0.592 — not a single query's top-10 changes from pool 50 to 500.** Fused R@50 barely moves and is
**non-monotonic** (0.793 → 0.862 → 0.828 → 0.845): deeper pool sometimes *hurts* it. So **gap@10
widens** (the union rises while fused stays put) and **gap@50 does not close** — at pool=500 the
candidate set holds *everything* (union 1.000) yet fused top-50 leaves the *same* +0.155 gap it had
at pool=50. The leak is fusion, not candidate absence.

**Pure fusion-demotion is visible at pool=500** (union recall 1.000 — the gold is in the candidate
set — but fused top-50 misses it): q12 WASP-96b, q21 H₂O, q24 Na, q25 escape, q26 clouds, q28 C/O,
q29 emission. Several are **both-arm** docs (present in *both* arms' top-500) that RRF still ranks
below fused #50 because they sit mid-rank in both lists — the corpus-widening fusion-dilution
mechanism, now unmovable by any depth.

### (a) All 29 — R@10 / R@50 by arm across depths

| pool | dense R@10 | lexical R@10 | hybrid R@10 | dense R@50 | lexical R@50 | hybrid R@50 |
|---|---|---|---|---|---|---|
| 50 | 0.529 | 0.437 | 0.592 | 0.776 | 0.810 | 0.793 |
| 100 | 0.529 | 0.437 | 0.592 | 0.776 | 0.810 | 0.862 |
| 200 | 0.529 | 0.437 | 0.592 | 0.776 | 0.810 | 0.828 |
| 500 | 0.529 | 0.437 | 0.592 | 0.776 | 0.810 | 0.845 |

Single-arm R@10/R@50 are **pool-independent by construction** (a single arm's ranking doesn't depend
on the pool) — flat across every row, shown for reference. Only **hybrid** moves with pool, and only
at R@50, and only by a query or two (non-monotonically). Hybrid R@10 is the immovable 0.592.

### (a) All 29 — complementarity across depths (depth-dependent by construction)

| pool | union R@pool | D-only | L-only | both | neither | mean cand-set size |
|---|---|---|---|---|---|---|
| 50 | 0.948 | 8 | 7 | 31 | 3 | 79 |
| 100 | 0.966 | 3 | 3 | 41 | 2 | 157 |
| 200 | 0.966 | 2 | 3 | 42 | 2 | 310 |
| 500 | 1.000 | 0 | 2 | 47 | 0 | 734 |

As pre-registered, these grow/shift **by construction**: a deeper pool turns "exclusive to one arm's
top-50" into "in both arms' top-500", so single-arm-only docs migrate to **both** (D-only 8 → 0) and
**neither** drains to 0 at pool=500. This is candidate-set *redistribution*, **not** evidence that
fusion needs the candidates less — the fused-top-k leak (above) persists regardless. The
candidate-set property and the ranking property are decoupled, and the ranking property is the one
that binds. (`mean cand-set size` is the H10 cost proxy: 79 → 734 candidates fused.)

### (b) KNOWN-ITEM (named q01–q18, scored) — gap and arms across depths

| pool | union R@pool | fused R@10 | fused R@50 | gap@10 | gap@50 |
|---|---|---|---|---|---|
| 50 | 1.000 | 0.716 | 0.853 | +0.284 | +0.147 |
| 100 | 1.000 | 0.716 | 0.941 | +0.284 | +0.059 |
| 200 | 1.000 | 0.716 | 0.941 | +0.284 | +0.059 |
| 500 | 1.000 | 0.716 | 0.941 | +0.284 | +0.059 |

| pool | dense R@10 | lexical R@10 | hybrid R@10 | dense R@50 | lexical R@50 | hybrid R@50 |
|---|---|---|---|---|---|---|
| 50 | 0.637 | 0.539 | 0.716 | 0.853 | 0.941 | 0.853 |
| 100–500 | 0.637 | 0.539 | 0.716 | 0.853 | 0.941 | 0.941 |

For the scored partition the candidate union is **1.000 at every depth** (the named gold is always in
the top-50 union). Fused R@10 is **flat at 0.716** — the gap@10 of **+0.284** is *immovable by pool*.
Fused R@50 rises once (0.853 → 0.941 at pool=100, ≈ 1–2 queries) then is **flat**; deeper than 100
buys nothing. Note lexical R@50 = 0.941 already at pool=50 — i.e. **lexical alone at depth 50 reaches
what hybrid reaches only at depth 100**, and hybrid's top-10 never catches the union.

### (b) BROAD-ENTITY (topical q19–q30, coverage measurement)

| pool | union R@pool | fused R@10 | fused R@50 | gap@10 | gap@50 |
|---|---|---|---|---|---|
| 50 | 0.875 | 0.417 | 0.708 | +0.458 | +0.167 |
| 100 | 0.917 | 0.417 | 0.750 | +0.500 | +0.167 |
| 200 | 0.917 | 0.417 | 0.667 | +0.500 | +0.250 |
| 500 | 1.000 | 0.417 | 0.708 | +0.583 | +0.292 |

Reported as a **measurement**, not recall (`|relevant| ≪ k` criterion). Same shape, more extreme:
fused R@10 flat at 0.417, gap@10 widens to +0.583, and fused R@50 is conspicuously **non-monotonic**
(0.708 → 0.750 → 0.667 → 0.708) — deeper pool actively demotes a previously-surfaced landmark.

### Per-depth paired CIs WITH leave-one-out (H9, all-scored)

| pool | comparison | mean Δ [95% CI] | excludes 0? | LOO surviving | hinges on |
|---|---|---|---|---|---|
| 50 | dense−hybrid | −0.063 [−0.126, −0.011] | yes | 25/29 | q08, q14, q18, q27 |
| 50 | lexical−hybrid | −0.155 [−0.270, −0.057] | yes | **29/29** | — |
| 100 | dense−hybrid | −0.063 [−0.126, −0.011] | yes | 25/29 | q08, q14, q18, q27 |
| 100 | lexical−hybrid | −0.155 [−0.270, −0.057] | yes | **29/29** | — |
| 200 | dense−hybrid | −0.063 [−0.126, −0.011] | yes | 25/29 | q08, q14, q18, q27 |
| 200 | lexical−hybrid | −0.155 [−0.270, −0.057] | yes | **29/29** | — |
| 500 | dense−hybrid | −0.063 [−0.126, −0.011] | yes | 25/29 | q08, q14, q18, q27 |
| 500 | lexical−hybrid | −0.155 [−0.270, −0.057] | yes | **29/29** | — |

Both edges are **identical at every depth** — the rows do not move. Because the R@10 edge lives
entirely in the (pool-invariant) top-10 fusion, the edges are pool-invariant too: the **fragile**
dense−hybrid edge stays fragile (25/29 LOO, same four hinge queries), the **robust** lexical−hybrid
edge stays robust (29/29). Neither compresses at depth.

## Interpretation

### H8 — confirmed (strongly): the bottleneck is fusion, not candidate depth

Deeper pool raises candidate-union recall exactly as expected (0.948 → 1.000 all; 1.000 throughout
for known-item), but **fused top-10 recall is completely flat (0.592 all / 0.716 known-item /
0.417 broad) and fused top-50 barely moves and non-monotonically.** The fusion gap does **not** close
— gap@10 *widens* with depth, and at pool=500 the candidate set is complete (union 1.000) yet fused
top-50 leaves the same leak as pool=50. The mechanism is RRF demotion: a gold doc sitting mid-rank in
both arms earns only small reciprocal-rank terms and is ranked below the cut no matter how many
deeper candidates are admitted (the seven candidate-present-but-fused-demoted queries at pool=500
make this concrete, several of them *both-arm* docs). **Pool depth is not the lever.**

### H9 — falsified, in a way that reinforces H8

The prediction was that edges compress at depth (fragile dense−hybrid vanishing, robust
lexical−hybrid persisting). Instead **both edges are pool-invariant** — identical CIs and LOO at
every depth. The reason is the H8 mechanism: the R@10 edge is a property of the top-10 fused ranking,
and the top-10 fused ranking does not change with pool, so the edges can't change either. The
fragile dense−hybrid edge does not vanish at depth — it stays exactly as fragile (carried by q08,
q14, q18, q27; collapses under any one's removal); the robust lexical−hybrid edge stays robust
(29/29). H9's "compress at depth" is wrong, and the corrected picture — *depth touches nothing in the
top-10* — is the same finding as H8 from the arm-comparison angle.

### H10 — operating point + lever call

**The lever is stage-2 reranking, not pool depth.** The candidate set is well-populated (union 1.000
at pool=500 all-scored, 1.000 at every depth for known-item); the recall that users see in the top-10
is gated entirely by RRF's ordering, which pool cannot fix (gap@10 +0.284 known-item / +0.356 all,
immovable). A cross-encoder reranking the existing candidate set is the intervention that can convert
that high union recall into top-k recall.

**Operating point (if staying on RRF for now): pool = 100.** It captures the only recoverable fused
gain — known-item fused R@50 0.853 → 0.941 (≈ 1–2 queries), all-scored fused R@50 peaks at 0.862 —
at a mean candidate set of ~157; deeper pools (310, 734 candidates) buy *no* fused-recall gain and
sometimes lose it (the non-monotonic R@50). So pool = 100 is the cheap, saturated setting; do not pay
for 200/500.

**Stated stage-2 boundary.** At the current pipeline (hybrid recall → RRF → top-k, no cross-encoder)
pool cost is the modest fusion cost reported above (the candidate-set-size proxy). The binding
latency cost arrives only when **stage-2 reranking exists** and the pool becomes the reranker's input
size — at which point pool=100 (~157 candidates) vs pool=500 (~734) is the latency knob to revisit.
PR #7 fixes the **recall headroom** (it is gated by fusion, and pool ≥ 100 wastes candidates); the
full latency tradeoff is deferred to when reranking lands.

### Tie-back to H5/H6 and the lever for the next PR

H5 (widening hurts at fixed pool) and H6 (the single-arm/dense-blind class scales) both pointed at a
stage-1 candidate-generation story. This sweep closes that thread: the candidate generator is **not**
the binding constraint — at pool ≥ 100 the union holds ≥ 0.966 (1.000 at pool=500), and for the
named/known-item queries it is a perfect 1.000 at every depth. What binds is the **fusion ranking**:
RRF demotes single-arm-strong and mid-rank-in-both gold below the cut, and no candidate depth repairs
it. The next retrieval lever is therefore a **stage-2 cross-encoder reranker** over a pool=100
candidate set — not a deeper pool, and not (for these query-bounded coverage gaps) more corpus.

### Verdict on the pre-registered hypotheses

- **H8 — confirmed (strongly).** Union recall rises with depth; fused top-10 is pool-invariant and
  fused top-50 barely moves (non-monotonically); the gap does not close (gap@10 widens). Bottleneck =
  RRF demotion, not candidate presence → next lever is fusion / stage-2 reranking.
- **H9 — falsified, reinforcing H8.** Edges do not compress at depth; they are pool-invariant
  because top-10 fusion is pool-invariant. The fragile dense−hybrid edge stays fragile (25/29 LOO);
  the robust lexical−hybrid edge stays robust (29/29) — at every depth.
- **H10 — lever call made.** Pool depth is not the lever; set pool = 100 (cheap, saturates the small
  recoverable fused R@50, ~157 candidates) and prioritise a stage-2 reranker, which the high union
  recall is primed to exploit. Latency tradeoff revisited when reranking lands.

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
