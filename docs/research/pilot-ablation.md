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

### Follow-on prediction (registered after seeing Recall@10, before the new readouts)

H4 is **not** part of the original pre-registration above. It was registered *after* the
Recall@10 / MRR results were in but *before* computing the depth, candidate-set/complementarity, and
bootstrap readouts that this section adds. (The Recall@10 ordering — hybrid lowest — was being
over-read as "H3 refuted"; at n=29 those deltas may be within sampling noise. These readouts
discipline the aggregate claims and answer the question Recall@10 can't: does hybrid earn its place
as the stage-1 candidate generator ahead of the reranker?)

- **H4.** Union recall at pool depth ≥ each single arm's Recall@pool (true by construction); the
  open empirical question is the **magnitude** of complementarity. If lexical-only and dense-only
  both contribute non-trivial relevant docs, hybrid candidate generation is justified despite its
  fused-top-k dilution. If union ≈ dense, lexical adds little and the hybrid stage-1 is not earning
  its complexity.

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

**Two clean patterns (point estimates):** hybrid wins **every MRR split** (it ranks the first
relevant doc highest); dense wins **every Recall@10 split** (all-scored and broad), with lexical's
lone Recall@10 win on the named set driven entirely by q12. Hybrid has the **lowest** all-scored
Recall@10 (0.690 vs dense 0.724). But these are point estimates on n=29 — the bootstrap CIs below
show the Recall@10 ordering is within sampling noise; the depth and candidate-set readouts are what
carry the architecture decision.

### Recall at depth — Recall@10 vs Recall@50

The same arms cut at the pool depth (k=50) instead of 10. Recall lifts sharply for all arms, and at
depth the ordering **inverts**: hybrid has the **highest** Recall@50 (all 0.966), because the fused
top-50 draws relevant docs from both arms' pools.

| arm | named @10 | named @50 | broad @10 | broad @50 | all @10 | all @50 |
|---|---|---|---|---|---|---|
| dense | 0.794 | 0.941 | 0.625 | 0.833 | 0.724 | 0.897 |
| lexical | 0.804 | 1.000 | 0.583 | 0.833 | 0.713 | 0.931 |
| hybrid | 0.794 | 1.000 | 0.542 | 0.917 | 0.690 | **0.966** |

### Candidate-set recall & complementarity at pool depth (k=50)

The operative metric for a reranked pipeline: recall of the candidate **set** the reranker receives.
For single arms this is the arm's top-50; for hybrid it is the **union** of both arms' top-50
(≥ max(arms) by construction). Recall columns are macro-averaged per query; `D-only / L-only / both
/ neither` are **document counts** (micro) over the relevant docs in the slice. On this corpus the
fused top-50 realizes the full union recall exactly (hybrid R@50 = union R@50 in every slice).

| slice | union R@50 | dense R@50 | lexical R@50 | Δ(union−dense) | D-only | L-only | both | neither | relevant |
|---|---|---|---|---|---|---|---|---|---|
| named | 1.000 | 0.941 | 1.000 | +0.059 | 0 | 1 | 28 | 0 | 29 |
| broad | 0.917 | 0.833 | 0.833 | +0.083 | 2 | 2 | 14 | 2 | 20 |
| all | 0.966 | 0.897 | 0.931 | +0.069 | 2 | 3 | 42 | 2 | 49 |

Single-arm-only relevant docs at pool depth (the complementarity), with the gold paper's rank in the
**other** arm's full 500-doc ranking:

- **lexical-only (3):** q12 `2022ApJ...936L..14P` (dense **#338** — truly dense-blind); q24
  `2018A&A...612A..53P` (dense #89); q28 `2024ApJ...963L...5X` (dense #54).
- **dense-only (2):** q24 `2020A&A...635A.206C` (lexical #87); q26 `2023ApJ...951...96G` (lexical #96).
- **neither (2):** q21's two H2O docs (`2024ApJ...963L...5X` dense #192 / lexical #61;
  `2018AJ....155...29W` dense #82 / lexical #74) — outside both pools.

Note: only q12 is truly single-arm-exclusive (dense #338); the other 4 single-arm-only docs sit at
ranks 54–96 in the other arm — just outside pool=50, i.e. reachable by a larger pool, not blind.

### Bootstrap 95% CIs

Resampled over queries with replacement, B=10000, fixed seed=20260531. The decision rule for "within
noise" is the **paired-difference** bootstrap (resample the per-query differences), not overlap of
marginal CIs.

**Marginal CIs** (all-scored slice):

| arm | R@10 mean [95% CI] | MRR mean [95% CI] |
|---|---|---|
| dense | 0.724 [0.586, 0.845] | 0.615 [0.454, 0.772] |
| lexical | 0.713 [0.575, 0.833] | 0.580 [0.432, 0.728] |
| hybrid | 0.690 [0.534, 0.828] | 0.623 [0.466, 0.772] |

**Paired-difference CIs** (does the 95% CI include 0?):

| comparison | slice | metric | mean Δ [95% CI] | includes 0? |
|---|---|---|---|---|
| dense−hybrid | all | R@10 | +0.034 [−0.069, +0.121] | yes |
| dense−hybrid | broad | R@10 | +0.083 [−0.167, +0.292] | yes |
| lexical−hybrid | all | R@10 | +0.023 [−0.069, +0.126] | yes |
| dense−hybrid | all | MRR | −0.008 [−0.091, +0.069] | yes |
| dense−lexical | broad | R@10 | +0.042 [−0.167, +0.250] | yes |
| dense−lexical | broad | MRR | +0.057 [−0.117, +0.254] | yes |

Every pairwise aggregate difference's CI includes 0 — no aggregate Recall@10 or MRR ordering is
statistically supported at n=29.

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

### Per-query depth & complementarity (k=50)

Each arm's `R@10/@50`, the hybrid candidate-set (`union R@50`), and the document-level arm tags
(`found-by`) — so the split single-arm-strong cases (q12, q24, q28) are visible at the document
level, not just the query level.

| id | target | gold | dense R@10/@50 | lexical R@10/@50 | hybrid R@10/@50 | union R@50 | found-by (docs) |
|---|---|---|---|---|---|---|---|
| q01 | WASP-39b | 3 | 1.00/1.00 | 1.00/1.00 | 1.00/1.00 | 1.00 | 3 both |
| q02 | WASP-39b | 1 | 1.00/1.00 | 1.00/1.00 | 1.00/1.00 | 1.00 | 1 both |
| q03 | K2-18b | 2 | 0.50/1.00 | 0.50/1.00 | 0.50/1.00 | 1.00 | 2 both |
| q04 | K2-18b | 2 | 1.00/1.00 | 1.00/1.00 | 1.00/1.00 | 1.00 | 2 both |
| q05 | TRAPPIST-1e | 2 | 0.50/1.00 | 0.50/1.00 | 0.50/1.00 | 1.00 | 2 both |
| q06 | TRAPPIST-1b | 1 | 1.00/1.00 | 1.00/1.00 | 1.00/1.00 | 1.00 | 1 both |
| q07 | HD 189733b | 2 | 1.00/1.00 | 0.50/1.00 | 1.00/1.00 | 1.00 | 2 both |
| q08 | HD 209458b | 3 | 1.00/1.00 | 0.67/1.00 | 1.00/1.00 | 1.00 | 3 both |
| q09 | 55 Cancri e | 2 | 1.00/1.00 | 1.00/1.00 | 1.00/1.00 | 1.00 | 2 both |
| q10 | GJ 1214b | 2 | 1.00/1.00 | 1.00/1.00 | 1.00/1.00 | 1.00 | 2 both |
| q11 | WASP-121b | 1 | 0.00/1.00 | 0.00/1.00 | 0.00/1.00 | 1.00 | 1 both |
| q12 | WASP-96b | 1 | 0.00/0.00 | 1.00/1.00 | 0.00/1.00 | 1.00 | 1 L-only |
| q13 | LTT 9779b | 1 | 1.00/1.00 | 1.00/1.00 | 1.00/1.00 | 1.00 | 1 both |
| q14 | GJ 486b | 2 | 1.00/1.00 | 1.00/1.00 | 1.00/1.00 | 1.00 | 2 both |
| q16 | TOI-270d | 1 | 1.00/1.00 | 1.00/1.00 | 1.00/1.00 | 1.00 | 1 both |
| q17 | WASP-17b | 1 | 1.00/1.00 | 1.00/1.00 | 1.00/1.00 | 1.00 | 1 both |
| q18 | WASP-43b | 2 | 0.50/1.00 | 0.50/1.00 | 0.50/1.00 | 1.00 | 2 both |
| q19 | CO2 | 1 | 1.00/1.00 | 1.00/1.00 | 1.00/1.00 | 1.00 | 1 both |
| q20 | SO2 | 1 | 1.00/1.00 | 1.00/1.00 | 1.00/1.00 | 1.00 | 1 both |
| q21 | H2O | 2 | 0.00/0.00 | 0.00/0.00 | 0.00/0.00 | 0.00 | 2 neither |
| q22 | CH4 | 1 | 0.00/1.00 | 1.00/1.00 | 1.00/1.00 | 1.00 | 1 both |
| q23 | CO | 1 | 1.00/1.00 | 1.00/1.00 | 1.00/1.00 | 1.00 | 1 both |
| q24 | Na | 2 | 0.50/0.50 | 0.50/0.50 | 0.00/1.00 | 1.00 | 1 D-only · 1 L-only |
| q25 | escape | 2 | 1.00/1.00 | 0.50/1.00 | 0.50/1.00 | 1.00 | 2 both |
| q26 | clouds | 2 | 0.50/1.00 | 0.50/0.50 | 0.50/1.00 | 1.00 | 1 both · 1 D-only |
| q27 | retrieval | 2 | 1.00/1.00 | 1.00/1.00 | 1.00/1.00 | 1.00 | 2 both |
| q28 | C/O ratio | 2 | 0.50/0.50 | 0.50/1.00 | 0.00/1.00 | 1.00 | 1 both · 1 L-only |
| q29 | emission | 2 | 0.50/1.00 | 0.00/1.00 | 0.00/1.00 | 1.00 | 2 both |
| q30 | phase curve | 2 | 0.50/1.00 | 0.00/1.00 | 0.50/1.00 | 1.00 | 2 both |

The `found-by` column makes the mechanism document-level: q12 is `1 L-only` (the dense-blind ERO),
q24 is `1 D-only · 1 L-only` (split relevance — each gold strong in a different arm), q21 is
`2 neither` (both H2O docs outside both pools). Every query's `union R@50` ≥ each arm's R@50.

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

### What this ablation establishes (robust to sample size)

The load-bearing findings are per-query existence proofs, which do not depend on n:

- **H1 confirmed.** Lexical recovers single-arm-strong targets that hybrid buries. q12's ERO sits
  at lexical #4 / dense #338; RRF ranks it below 10 because items appearing mid-rank in BOTH lists
  outscore a single 1/(60+4) contribution (a dense-#8 + lexical-#15 item gets 1/68 + 1/75 ≈ 0.028 >
  0.016). q24 is a second instance. This is a clean demonstration of RRF's penalty on
  single-arm-strong items — a fusion failure, not a pool failure (q12's ERO at dense #338 is
  unreachable by any sane pool; the q12 caveat, now shown).
- **Complementarity (H4).** Union recall at pool depth = 0.966 all / 1.000 named / 0.917 broad vs
  dense 0.897 / 0.941 / 0.833 and lexical 0.931 / 1.000 / 0.833. Of relevant docs found at pool
  depth, 2 were dense-only, 3 lexical-only, 42 both (of 49 relevant; 2 in neither). The 3 lexical-only
  docs are ones lexical ranks inside pool=50 but dense ranks outside it — though only q12's ERO is
  *truly* dense-blind (dense #338); the other two sit at dense #54 and #89, just past the pool.
  Symmetrically, the 2 dense-only docs sit at lexical #87 and #96 — just past the pool, not lexically
  blind (and one is q24's "Na", a molecule query, not a purely conceptual one). So complementarity is
  modest but bidirectional (5 of 49 relevant docs, ~10%, reach only one arm at pool=50), lifting
  pool-depth recall to 0.966 — above the better single arm (lexical 0.931) — enough to justify hybrid
  candidate generation. The honest caveat: 4 of the 5 single-arm-only docs are merely "just outside
  the other arm's pool of 50" rather than truly single-arm-exclusive; only q12 is. That distinction
  matters for PR #6 — most of this complementarity is pool-reachable, but the q12-class fusion failure
  is not.

### What it does NOT establish (within noise at n=29)

- **Aggregate recall@10 ordering.** Hybrid's lower point estimate (0.690 all vs dense 0.724) is a
  0.034 gap on 29 queries — on the order of one query's relevant-doc recall, concentrated in the
  broad slice (n=12; the named-slice gap is exactly 0.000). The paired-difference bootstrap
  (dense−hybrid) gives 95% CI [−0.069, +0.121], which includes 0. The recall@10 ordering is therefore
  best read as suggestive, not as a directional result the data can support.
- **H2 is directionally confirmed but within noise.** Dense > lexical on broad for both recall
  (0.625 vs 0.583) and MRR (0.374 vs 0.317), consistent with "no entity string to anchor on," but the
  0.042 gap on n=12 is inside the CI (dense−lexical broad R@10 95% CI [−0.167, +0.250], includes 0).
- **The MRR pattern is small but directionally consistent.** Hybrid leads MRR on all three slices
  (+0.007 to +0.009), which is what RRF should produce — it optimizes rank position. The consistency
  across slices is the signal; the magnitude is within noise (dense−hybrid MRR all 95% CI
  [−0.091, +0.069], includes 0).

### Why recall@10 is the wrong target metric for this architecture

Stage-1 hybrid recall feeds a stage-2 reranker over the candidate pool. The reranker repairs top-k
ordering, so stage 1's job is to maximize the recall of the candidate SET it hands up, not its own
top-10 ranking. Hybrid's fused-top-10 dilution is therefore cosmetically bad but architecturally
irrelevant IF union recall is high — and union recall = 0.966 vs dense 0.897 confirms this (the fused
top-50 realizes the full union here, so the reranker receives a candidate set with strictly higher
recall than either arm at the same depth). The metric that should drive the stage-1 decision is
candidate-set recall + complementarity, not recall@10 of the fusion in isolation.

### Verdict on the pre-registered hypotheses

- **H1 — confirmed** (existence proof; q12, q24).
- **H2 — directionally confirmed, not statistically supported at n=29** (within CI).
- **H3 — split.** Clause (b) "beaten by a single arm on specific tails" is **confirmed strongly**
  (q12, q29, and q24/q28 beaten by both arms via split single-arm-strong relevance). Clause (a)
  "best on average" is **underpowered, neither confirmed nor refuted**: hybrid's recall@10 point
  estimate is lowest but the paired difference CI includes 0, and it is modestly favored on MRR. The
  earlier "refuted" framing is withdrawn — n=29 cannot license a confident negative.
- **H4 — confirmed.** Union recall exceeds each arm (by construction); complementarity magnitude is
  modest but bidirectional — 3 lexical-only (incl. the dense-blind q12) and 2 dense-only of 49
  relevant docs, lifting pool-depth recall to 0.966 (above lexical's 0.931 and dense's 0.897).

### Implication for the architecture decision and for PR #6

Keep hybrid as the stage-1 candidate generator: q12 alone proves lexical recovers dense-blind targets
the reranker can then promote, and union recall exceeds dense alone (+0.069 at pool depth, +0.035 over
the better single arm). The fused-top-k dilution is the reranker's problem to solve, not a reason to
drop an arm. The transferable result here is the FUSION-DILUTION MECHANISM and the COMPLEMENTARITY,
not the specific recall numbers — all of which are on 500 abstracts. PR #6 (corpus widening) tests
whether the mechanism survives and re-measures the magnitudes; the mechanism should transfer, the
numbers will move.

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
