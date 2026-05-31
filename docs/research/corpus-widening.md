# Corpus Widening — Same Three Arms on a Widened/Recent ADS Slice (method-fixed)

Follow-up to the [pilot retrieval ablation](pilot-ablation.md). That experiment held the **corpus
frozen** and varied the retrieval **arm** (dense / lexical / hybrid). This one is its mirror image:
it holds the **method** frozen and varies the **corpus** — re-running the *same* three arms, with
the *same* readout suite (R@10, R@50, candidate-set/union recall + complementarity, paired-bootstrap
CIs) and the *same* labeling protocol, on a **widened** 2,500-abstract corpus, so any delta is
attributable to the corpus, not the method.

This is the "widen the ADS slice" commission queued after the pilot ablation merged (PR #5).

## Setup — what is fixed and what varies

### The widened corpus (the manipulated variable)

Re-ingested from the NASA ADS search API on 2026-05-31 with the **same query and year filter** as
the pilot, only a deeper citation-ranked cut. The original 500-abstract pilot corpus is a
**byte-identical strict subset** (the widening tool keeps it as a prefix; see Reproduce).

| property | pilot | widened |
|---|---|---|
| ADS query | `abs:"exoplanet atmosphere"` | `abs:"exoplanet atmosphere"` (identical) |
| year filter | `year:[2018 TO *]` | `year:[2018 TO *]` (identical) |
| sort / cut | citation_count desc, top **500** | citation_count desc, top **2,500** |
| corpus size | 500 | **2,500** (orig 500 ⊂ widened, overlap 500/500) |
| pipeline | `pilot-retrieval-0.1.0` (one chunk = one abstract) | `pilot-retrieval-0.1.0` (identical) |

Widening this corpus required a **re-ingest + re-embed + re-index** — the deliberate opposite of the
frozen ablation, which re-used a single frozen index. The pilot remains reproducible: its 500
abstracts are preserved verbatim and the pilot index rebuilds from them deterministically (verified:
re-running the harness on the 500-corpus reproduces the frozen pilot aggregates exactly).

**The 2,000 new abstracts skew strongly recent** (deeper in the citation ranking ⇒ younger,
less-cited papers):

| year | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---|---|---|---|---|---|---|---|---|
| original 500 | 56 | 70 | 118 | 76 | 69 | 62 | 40 | 7 | 2 |
| new 2,000 | 98 | 104 | 203 | 211 | 258 | 263 | **388** | **380** | **95** |

### Stated confound (read this before the results)

**Size and recency move together here, by construction.** A deeper citation-ranked slice of the
same query admits more recent, lower-citation papers — so this experiment does **not** isolate
*corpus size* from *corpus recency*. Decomposing those two axes is a separate later experiment if
warranted. What **is** controlled here is the retrieval **method**.

### Reverse single-variable rule — the controlled variable is the METHOD

Held fixed and identical to the pilot (only corpus membership changes):

| held fixed | value |
|---|---|
| dense model | BGE-small-en-v1.5, 384-dim, normalized |
| dense store | pgvector HNSW (`m=32`, `ef_construction=256`), cosine |
| lexical store | SQLite FTS5, BM25, OR-of-tokens match |
| chunking | naive — one chunk = title + abstract |
| arms | dense / lexical / hybrid (RRF) — same `retrieve()` code |
| cutoff | `k = 10` |
| candidate pool | `pool = 50` |
| RRF constant | `RRF_K = 60` |
| harness | `ablation_retrieval.py` (unchanged); readouts derived from full-depth arm reads |

**Pre-registered pool confound.** `pool` stays at 50 even though it is now a much smaller fraction
of the index (50 / 2,500 = 2% vs 50 / 500 = 10%). We do **not** sweep the pool here — that remains a
separate experiment. So the "widened" arm reads a *fixed-size* candidate set out of a 5× larger
index; any recall change conflates "more distractors in the corpus" with "fixed pool is now a
shallower fraction." Both are corpus-scale effects (not method changes), and both are intended to be
in scope for H5.

## Coverage-membership probe (pre-registration input)

The commission's premise was that widening would pull the pilot's two coverage-miss targets into the
corpus — q15 (LHS 475b, unscored in the pilot for lack of an in-corpus target) and q11's canonical
WASP-121b thermal-inversion paper. **Before pre-registering, we checked whether they can enter at
all.** They cannot — and *not* because of corpus depth. Two distinct query-level exclusion
mechanisms keep them out at **any** depth of this query:

| coverage target | bibcode | year | in `abs:"exoplanet atmosphere"`? | in widened corpus? | why excluded |
|---|---|---|---|---|---|
| q15 LHS 475b (JWST discovery) | `2023NatAs...7.1317L` | 2023 | **no** | absent | **phrase** — abstract lacks the exact phrase "exoplanet atmosphere" |
| q15 LHS 475b (Venus analog) | `2024AJ....167..197M` | 2024 | **no** | absent | **phrase** |
| q11 WASP-121b stratosphere (Evans 2017) | `2017Natur.548...58E` | 2017 | no | absent | **year** — predates `year_start=2018` |
| q11 dayside thermal inversion (Sedaghati 2017) | `2017ApJ...850L..32S` | 2017 | no | absent | **year** |
| q11 TiO/VO (Evans 2016) | `2016ApJ...822L...4E` | 2016 | yes | absent | **year** |
| q11 H⁻/dissociation dayside (Arcangeli 2018) | `2018ApJ...855L..30A` | 2018 | **no** | absent | **phrase** |
| q11 CO-emission inverted atm. (Vince 2023) | `2023MNRAS.522.2145V` | 2023 | yes | **already in pilot 500** | — (present, pilot-unlabeled) |

*Provenance: targets identified 2026-05-31 by ADS title/abstract/identifier queries against the live
search API; membership of each in the pilot phrase slice and in the widened corpus verified
programmatically. The pilot's original 29 labels are unchanged.*

**Mechanism.** The pilot's phrase query is itself narrow: the entire `abs:"exoplanet atmosphere"`
universe is **6,492 papers** (4,845 since 2018). Many obviously-relevant papers — the LHS 475b
discovery papers, half the WASP-121b inversion literature — live **outside** that phrase universe
(phrase-excluded) or **before 2018** (year-excluded). **Widening the citation cut (500 → 2,500)
moves neither bound.** q15 is a phrase-recall problem; q11's canonical inversion paper is a
year-cutoff problem. Neither is a corpus-size problem, so neither is fixable by this manipulation.
(Notably, the pilot 500 already contained an *inverted-atmosphere* WASP-121b emission paper —
`2023MNRAS.522.2145V` — that the pilot did not label for q11, so q11's concept was already partly
in-corpus.) **Consequently no originally-unscored query becomes scorable, and H7 is resolved at this
membership stage — see below.** The query-recall bottleneck is quantified further in the
[appendix](#appendix-the-query-recall-bottleneck).

## Predictions (pre-registered)

Registered **before** the retrieval readouts below. H5–H7 are the commission's hypotheses, authored
before this corpus was built; the membership probe above is the only pre-result fact consulted.

- **H5 (the controlled test).** On the **same 29 queries at fixed pool=50**, widening does **not**
  improve — and may slightly reduce — R@10 / R@50, because the 2,000 added papers are distractors
  competing for fixed pool slots (and pool=50 is now a shallower fraction of the index). Any coverage
  gain should accrue to *newly-scorable* queries, not the original 29.
- **H6 (the strategic one).** The single-arm-exclusive (dense-blind) class — q12's ERO in the pilot —
  does **not** stay a singleton. If newly-present **entity** targets land dense-blind the way the ERO
  did, the dense-blind class scales with the corpus, and hybrid stage-1 is justified as a *growing*
  safety net rather than q12-specific over-engineering. **Falsifiable:** report the dense-rank of
  newly-present entity targets and whether lexical/hybrid recover what dense buries.
- **H7 (coverage).** Newly-present coverage targets (q11 inversion paper, q15 LHS 475b) are
  retrievable once in-corpus; this is a coverage/recency win, not an arm win — but entity targets may
  be dense-blind, feeding H6.
- **"neither" note.** The pilot's two "neither" docs (q21's H₂O pair) are present-but-unretrieved by
  both arms in the top-50 — a class distinct from the absent coverage misses. Widening does not
  directly fix "neither" (already in corpus); track whether it changes, but do not expect it to.

**H7 is resolved at the membership stage and is falsified by construction:** per the probe above,
*neither* coverage target enters the widened corpus under the fixed query — q15 is phrase-excluded,
q11's canonical inversion paper is year-excluded — so neither becomes retrievable at any depth. This
is a result about the *query*, not the corpus size, and is exactly why the commission was scoped to
hold the query fixed (the single-variable rule). The retrieval readouts below therefore test H5 and
H6; H7's coverage premise is refuted before retrieval is even run.

## Results

> **Pending the retrieval readouts** (recorded in the next commit, after this pre-registration is
> committed — mirroring the pilot's H1–H4 audit trail). The scored experiment is report (a): the
> original 29 queries with their **frozen** labels, re-scored on the widened 2,500-abstract index,
> compared against the frozen pilot baseline. H6 evidence (dense-ranks of newly-present on-target
> docs) is reported alongside.

## Reproduce

```bash
# 1. Widen the corpus (same query/year as the pilot, deeper cut; keep the 500 as a strict subset)
uv run python packages/data-pipeline/src/ingest_ads.py \
    --limit 2500 \
    --output-dir data/raw/ads_exoplanet_atmospheres_widened \
    --merge-with data/raw/ads_exoplanet_atmospheres/abstracts.jsonl

# 2. Re-embed + re-index (pgvector + FTS5; --reset rebuilds over the pilot index)
uv run python packages/rag/src/index_corpus.py \
    --input data/raw/ads_exoplanet_atmospheres_widened/abstracts.jsonl --reset

# 3. Re-run the three arms — same harness, same labels (report a)
uv run python packages/evaluation/src/ablation_retrieval.py \
    --queries packages/evaluation/queries/pilot_exoplanet_atmospheres.yaml \
    --json-out docs/research/corpus-widening-results.json
```

Raw per-query results (every arm, every query, plus diagnostic full-corpus ranks) are written to
`docs/research/corpus-widening-results.json`.

## Appendix — the query-recall bottleneck

A measurement (not a second experiment) requested alongside the controlled result, to quantify *why*
size-widening cannot deliver the coverage the commission expected. All counts are ADS `numFound`
probes against the live search API on 2026-05-31.

| ADS probe | numFound | reading |
|---|---|---|
| `abs:"exoplanet atmosphere"` (all years) | 6,492 | the **entire** phrase universe the pilot query can ever return |
| `abs:"exoplanet atmosphere"` AND `year:[2018 TO *]` | 4,845 | the slice widening samples from; we took the top **2,500** = **52%** of it |
| `abs:"exoplanet atmosphere"` AND `abs:"LHS 475"` | **0** | the q15 discovery papers are phrase-external — unreachable at **any** cut |
| `abs:"exoplanet atmosphere"` AND `full:"WASP-121"` AND `abs:"stratosphere"` | 1 | the lone in-phrase WASP-121b stratosphere paper is itself year-excluded (2016) |

**Three things this shows:**

1. **The query, not the depth, is the binding constraint.** At 2,500 we already sample over half of
   the phrase universe; pushing the cut deeper hits diminishing returns *within the same 6,492-paper
   ceiling* and never reaches a phrase-external paper. LHS 475b sits at `numFound = 0` against this
   query — no citation cut admits it.
2. **The two exclusion mechanisms are distinct and both query-level.** q15 is *phrase*-excluded
   (its abstracts omit the literal phrase, despite being squarely about an exoplanet atmosphere);
   q11's canonical inversion papers are *year*-excluded (2016–2017, before `year_start=2018`). One in
   the year range (Arcangeli 2018) is *also* phrase-excluded — so even relaxing the year alone would
   not capture it.
3. **The fix is a different lever — and a different experiment.** Capturing these targets requires
   widening the **query** (object/title resolution, or relaxing the exact phrase), which changes the
   retrieval-query variable. That deliberately violates the single-variable rule this commission was
   scoped around, so it is out of scope here and flagged as a distinct future lever: **query/recall
   widening ≠ corpus-size widening.** They are orthogonal axes, and only the latter was manipulated.
