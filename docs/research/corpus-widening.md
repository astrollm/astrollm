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

The scored experiment is **report (a)**: the original 29 queries with their **frozen** labels,
re-scored on the widened 2,500-abstract index. Same harness, same labels, same method — only the
corpus changed.

### Report (b) — stratification and the `|relevant| ≪ k` criterion

Report (b) ("full widened gold set") is stratified rather than collapsed, under a stated methodology
criterion:

> **Recall@k is reported only where `|relevant| ≪ k`.** Where the in-corpus relevant set is small
> (known-item / single-target queries), Recall@k measures ranking quality and is scored. Where it
> balloons — a named entity on a 5× corpus matches dozens of in-corpus papers, so a full-recall
> relabel makes Recall@10 cap near `10/|relevant|` and stop measuring ranking — R@k **degenerates**
> and that stratum is reported as a **coverage measurement** (arm ranks of newly-present docs), not a
> score. This is the same limitation the pilot flagged for the broad queries, now stated as a rule.

**Path taken for the scored (b) slice: documented and deferred to PR #7** (the guardrail path), for
three reasons:

1. **No query became newly scorable.** The only pilot query lacking an in-corpus target was q15, and
   the membership probe shows it stays absent (phrase-excluded). So the strict set of *newly-scorable
   known-item targets* is empty — there is no objective small relabel that report (a) does not
   already cover.
2. **A looser known-item relabel is subjective and exceeds the ~10-target guardrail.** Adding
   newly-present landmarks to already-scored queries means choosing "the" landmark among dozens of
   on-target newcomers per entity (the [H6 scan](#single-arm-exclusive-class-growth-h6) shows the
   scale). That is a judgment-heavy, result-adjacent relabel of well over ten targets — it would
   delay merge and is exactly what the guardrail says to defer.
3. **PR #7 re-runs selection anyway.** The pool sweep recomputes the fused candidate set at every
   depth, so a single consistent known-item relabel scored across pool depths belongs there.

**This PR is the pool=50 slice only** — it fixes the criterion above and the report-(a) baseline. The
**full across-pool-depth stratified (b)** — known-item relabel (with provenance) scored at each pool
depth — **rides PR #7.** Until then, newly-present on-target docs are reported here as a measurement
(their arm ranks, in the H6 section), not as scored gold.

### Headline — pilot → widened, all-scored (H5)

Best per column **bold**. Every arm loses recall and MRR on the *same* queries; the loss is large,
not slight.

| arm | R@10 (pilot → widened) | R@50 (pilot → widened) | MRR (pilot → widened) |
|---|---|---|---|
| dense | 0.724 → 0.529 (**−0.195**) | 0.897 → 0.776 (−0.121) | 0.615 → 0.377 (−0.238) |
| lexical | 0.713 → 0.437 (**−0.276**) | 0.931 → 0.810 (−0.121) | 0.580 → 0.288 (−0.292) |
| hybrid | 0.690 → **0.592** (−0.098) | 0.966 → 0.793 (−0.173) | 0.623 → **0.371** (−0.252) |

The arm ordering **flips**: in the pilot hybrid had the *lowest* R@10; on the widened corpus it has
the *highest* on every split, and it degrades least (−0.098 vs dense −0.195, lexical −0.276). Lexical
collapses most — BM25's OR-of-tokens match drowns in 5× more token-matching distractors.

### Aggregate — Recall@10 / MRR by arm and split (widened)

| arm | named R@10 | named MRR | broad R@10 | broad MRR | all R@10 | all MRR |
|---|---|---|---|---|---|---|
| dense | 0.637 | 0.490 | 0.375 | 0.217 | 0.529 | 0.377 |
| lexical | 0.539 | 0.413 | 0.292 | 0.111 | 0.437 | 0.288 |
| hybrid | **0.716** | 0.456 | **0.417** | **0.251** | **0.592** | 0.371 |

### Recall at depth — Recall@10 vs Recall@50 (widened)

| arm | named @10 | named @50 | broad @10 | broad @50 | all @10 | all @50 |
|---|---|---|---|---|---|---|
| dense | 0.637 | 0.853 | 0.375 | 0.667 | 0.529 | 0.776 |
| lexical | 0.539 | **0.941** | 0.292 | 0.625 | 0.437 | **0.810** |
| hybrid | 0.716 | 0.853 | 0.417 | 0.708 | 0.592 | 0.793 |

The pilot's depth **inversion** (hybrid highest R@50) is **gone**: hybrid's fused top-50 (0.793) no
longer realizes the full union (0.948) — at pool=50 on a 2,500-index the fusion drops candidates that
the *union* keeps. This is the pre-registered pool confound made visible: a fixed pool is now a 2%
slice, so RRF's fused top-50 leaks recall that lives in the wider union.

### Candidate-set recall & complementarity at pool depth (k=50) (widened)

| slice | union R@50 | dense R@50 | lexical R@50 | Δ(union−dense) | D-only | L-only | both | neither | relevant |
|---|---|---|---|---|---|---|---|---|---|
| named | 1.000 | 0.853 | 0.941 | +0.147 | 2 | 3 | 24 | 0 | 29 |
| broad | 0.875 | 0.667 | 0.625 | +0.208 | 6 | 4 | 7 | 3 | 20 |
| all | 0.948 | 0.776 | 0.810 | +0.172 | 8 | 7 | 31 | 3 | 49 |

The **union premium grows**: Δ(union−dense) was +0.069 (all) in the pilot, now **+0.172**. The union
candidate set (what a reranker receives from hybrid stage-1) holds at 0.948 while each single arm's
pool-50 recall collapses (dense 0.897→0.776, lexical 0.931→0.810). Complementarity is the part of
the architecture that *improves* with scale.

### Single-arm-exclusive class growth (H6)

The single-arm-only (one arm's pool-50 misses it, the other catches it) doc class **grew 5 → 15** of
the relevant docs — and bidirectionally (dense-only 2→8, lexical-only 3→7). The cause is **incumbent
displacement**: 2024–2026 newcomers push the (mostly 2018–2023) gold docs deeper, dropping them out
of *one* arm's pool-50.

| query | gold doc | pilot tag → widened tag | widened dense# / lex# |
|---|---|---|---|
| q05 TRAPPIST-1e | `2020ApJ...901..126W` | both → **dense-only** | #29 / #69 |
| q07 HD 189733b | `2024ApJ...973L..41I` | both → **dense-only** | #8 / #76 |
| q11 WASP-121b | `2023ApJ...943L..17M` | both → **lexical-only** (dense-blind) | #70 / #33 |
| q18 WASP-43b | `2018ApJ...869..107M` | both → **lexical-only** | #84 / #40 |
| q22 CH4 | `2023ApJ...956L..13M` | both → **lexical-only** | #60 / #17 |
| q25 escape | `2020ApJ...890...79J` | both → **dense-only** | #36 / #83 |
| q26 clouds | `2023ApJ...951...96G` | dense-only → **neither** | #55 / #279 |
| q28 C/O | `2021ApJ...914...12L` | both → **dense-only** | #12 / #176 |
| q29 emission | `2020ApJ...890..176V` | both → **lexical-only** | #99 / #38 |
| q29 emission | `2023ApJ...943L..17M` | both → **dense-only** | #20 / #95 |
| q30 phase curve | `2020AJ....160..137F` | both → **dense-only** | #21 / #60 |
| q30 phase curve | `2020AJ....160..155W` | both → **dense-only** | #37 / #86 |

Single-arm membership is **query-dependent**, not a property of the paper: `2023ApJ...943L..17M`
(the WASP-121b JWST phase curve) is *lexical-only* for q11 (dense #70 / lex #33) yet *dense-only* for
q29 (dense #20 / lex #95) — the same doc, blinded by a different arm depending on the query phrasing.

**q12's ERO is the limiting case.** `2022ApJ...936L..14P` goes from dense #338 / lexical #4 (pilot)
to **dense #1594 / lexical #31** (widened) — more dense-blind, and its lexical rank slipped past the
top-10. So the pilot's H1 result (lexical recovers the ERO *into the top-10*) **weakens** at scale:
lexical now recovers it only into the *pool* (#31 < 50), not the top-10 — q12 is a top-10 miss for
**all three arms** on the widened corpus. But the architectural point **strengthens**: the ERO is
still in the hybrid candidate set for a reranker, and q11 joins it as a second dense-blind incumbent
(dense #70 / lexical #33).

**Newcomers are not the dense-blind ones.** A scan of every named entity's newly-present on-target
papers shows they overwhelmingly rank dense top-5 (q01 d#1, q05 d#1, q06 d#1, q07 d#1, q09 d#1, q13
d#1, q14 d#1, q18 d#1) — the *new* relevant docs are mostly dual-found near the top; it is the
*incumbents* that get displaced into the single-arm tail. The exception confirms the mechanism: a
generic-abstract tool paper (`2026AJ....171...98B`, "Virga" cloud model) lands dense #1550 / lexical
#67 — the same generic-abstract → dense-blind mechanism as q12's ERO. And the tail is bidirectional:
two genuine HD 209458b H₂O detections (`2024A&A...690A..63B` d#4/l#301; `2025AJ....170..223F`
d#2/l#307) are **lexical**-blind — the mirror of the ERO.

### Bootstrap 95% CIs (widened)

**Marginal 95% CIs** (B=10000, seed=20260531), all-scored slice:

| arm | R@10 mean [95% CI] | MRR mean [95% CI] |
|---|---|---|
| dense | 0.529 [0.374, 0.684] | 0.377 [0.231, 0.534] |
| lexical | 0.437 [0.287, 0.598] | 0.288 [0.167, 0.427] |
| hybrid | 0.592 [0.431, 0.747] | 0.371 [0.235, 0.518] |

**Paired-difference 95% CIs** (does the CI include 0?):

| comparison | slice | metric | mean Δ [95% CI] | includes 0? |
|---|---|---|---|---|
| dense−hybrid | all | R@10 | −0.063 [−0.126, −0.011] | **no** |
| dense−hybrid | broad | R@10 | −0.042 [−0.125, +0.000] | yes |
| lexical−hybrid | all | R@10 | −0.155 [−0.276, −0.052] | **no** |
| dense−hybrid | all | MRR | +0.006 [−0.094, +0.112] | yes |
| dense−lexical | broad | R@10 | +0.083 [+0.000, +0.208] | yes |
| dense−lexical | broad | MRR | +0.106 [−0.075, +0.324] | yes |

The change from the pilot: **hybrid > the single arms at R@10 now has CI support** — dense−hybrid
(all R@10) = −0.063 [−0.126, −0.011] and lexical−hybrid = −0.155 [−0.276, −0.052] both exclude 0,
where in the pilot every pairwise R@10 CI included 0 (within noise). But the two differ sharply in
robustness, so they should not be read as equally solid:

- **hybrid > lexical is robust.** The −0.155 gap is carried by 7 queries (q16, q27 at −1.00; q08
  −0.67; q09, q10, q25 −0.50; q01 −0.33) and survives **all 29** leave-one-out drops (every n=28
  subset still excludes 0).
- **hybrid > dense is suggestive, not robust — concentrated and fragile.** The −0.063 gap is carried
  by just **4 queries** (q14, q18, q27 at −0.50; q08 at −0.333); the other 25 are tied. Leave-one-out
  fails: dropping *any single one* of those 4 makes the CI include 0 (e.g. drop q27 → −0.048
  [−0.107, +0.000]). At n=29 the dense−hybrid R@10 advantage is best read as **suggestive**, not a
  result the data robustly support. (The robust statements remain: hybrid > lexical at R@10, and the
  growing union/candidate-set premium below, which does not rest on a handful of queries.)

### Per-query (all scored, raw) (widened)

| id | target | kind | gold | dense R@10 (rank) | lexical R@10 (rank) | hybrid R@10 (rank) |
|---|---|---|---|---|---|---|
| q01 | WASP-39b | named | 3 | 0.67 (#3) | 0.33 (#5) | 0.67 (#6) |
| q02 | WASP-39b | named | 1 | 1.00 (#1) | 1.00 (#1) | 1.00 (#1) |
| q03 | K2-18b | named | 2 | 0.50 (#1) | 0.50 (#1) | 0.50 (#1) |
| q04 | K2-18b | named | 2 | 1.00 (#6) | 1.00 (#6) | 1.00 (#4) |
| q05 | TRAPPIST-1e | named | 2 | 0.00 (—) | 0.00 (—) | 0.00 (—) |
| q06 | TRAPPIST-1b | named | 1 | 1.00 (#2) | 1.00 (#3) | 1.00 (#1) |
| q07 | HD 189733b | named | 2 | 0.50 (#8) | 0.50 (#6) | 0.50 (#7) |
| q08 | HD 209458b | named | 3 | 0.67 (#1) | 0.33 (#1) | 1.00 (#3) |
| q09 | 55 Cancri e | named | 2 | 1.00 (#5) | 0.50 (#5) | 1.00 (#5) |
| q10 | GJ 1214b | named | 2 | 1.00 (#1) | 0.50 (#4) | 1.00 (#1) |
| q11 | WASP-121b | named | 1 | 0.00 (—) | 0.00 (—) | 0.00 (—) |
| q12 | WASP-96b | named | 1 | 0.00 (—) | 0.00 (—) | 0.00 (—) |
| q13 | LTT 9779b | named | 1 | 1.00 (#2) | 1.00 (#2) | 1.00 (#2) |
| q14 | GJ 486b | named | 2 | 0.50 (#2) | 1.00 (#1) | 1.00 (#2) |
| q16 | TOI-270d | named | 1 | 1.00 (#1) | 0.00 (—) | 1.00 (#2) |
| q17 | WASP-17b | named | 1 | 1.00 (#1) | 1.00 (#1) | 1.00 (#1) |
| q18 | WASP-43b | named | 2 | 0.00 (—) | 0.50 (#5) | 0.50 (#6) |
| q19 | CO2 | broad | 1 | 1.00 (#4) | 1.00 (#4) | 1.00 (#1) |
| q20 | SO2 | broad | 1 | 1.00 (#1) | 1.00 (#3) | 1.00 (#1) |
| q21 | H2O | broad | 2 | 0.00 (—) | 0.00 (—) | 0.00 (—) |
| q22 | CH4 | broad | 1 | 0.00 (—) | 0.00 (—) | 0.00 (—) |
| q23 | CO | broad | 1 | 1.00 (#7) | 1.00 (#4) | 1.00 (#5) |
| q24 | Na | broad | 2 | 0.00 (—) | 0.00 (—) | 0.00 (—) |
| q25 | escape | broad | 2 | 0.50 (#1) | 0.00 (—) | 0.50 (#7) |
| q26 | clouds | broad | 2 | 0.50 (#10) | 0.50 (#2) | 0.50 (#3) |
| q27 | retrieval | broad | 2 | 0.50 (#9) | 0.00 (—) | 1.00 (#3) |
| q28 | C/O ratio | broad | 2 | 0.00 (—) | 0.00 (—) | 0.00 (—) |
| q29 | emission | broad | 2 | 0.00 (—) | 0.00 (—) | 0.00 (—) |
| q30 | phase curve | broad | 2 | 0.00 (—) | 0.00 (—) | 0.00 (—) |

### Per-query depth & complementarity (k=50) (widened)

| id | target | gold | dense R@10/@50 | lexical R@10/@50 | hybrid R@10/@50 | union R@50 | found-by (docs) |
|---|---|---|---|---|---|---|---|
| q01 | WASP-39b | 3 | 0.67/1.00 | 0.33/1.00 | 0.67/1.00 | 1.00 | 3 both |
| q02 | WASP-39b | 1 | 1.00/1.00 | 1.00/1.00 | 1.00/1.00 | 1.00 | 1 both |
| q03 | K2-18b | 2 | 0.50/1.00 | 0.50/1.00 | 0.50/1.00 | 1.00 | 2 both |
| q04 | K2-18b | 2 | 1.00/1.00 | 1.00/1.00 | 1.00/1.00 | 1.00 | 2 both |
| q05 | TRAPPIST-1e | 2 | 0.00/1.00 | 0.00/0.50 | 0.00/1.00 | 1.00 | 1 both · 1 D-only |
| q06 | TRAPPIST-1b | 1 | 1.00/1.00 | 1.00/1.00 | 1.00/1.00 | 1.00 | 1 both |
| q07 | HD 189733b | 2 | 0.50/1.00 | 0.50/0.50 | 0.50/1.00 | 1.00 | 1 both · 1 D-only |
| q08 | HD 209458b | 3 | 0.67/1.00 | 0.33/1.00 | 1.00/1.00 | 1.00 | 3 both |
| q09 | 55 Cancri e | 2 | 1.00/1.00 | 0.50/1.00 | 1.00/1.00 | 1.00 | 2 both |
| q10 | GJ 1214b | 2 | 1.00/1.00 | 0.50/1.00 | 1.00/1.00 | 1.00 | 2 both |
| q11 | WASP-121b | 1 | 0.00/0.00 | 0.00/1.00 | 0.00/0.00 | 1.00 | 1 L-only |
| q12 | WASP-96b | 1 | 0.00/0.00 | 0.00/1.00 | 0.00/0.00 | 1.00 | 1 L-only |
| q13 | LTT 9779b | 1 | 1.00/1.00 | 1.00/1.00 | 1.00/1.00 | 1.00 | 1 both |
| q14 | GJ 486b | 2 | 0.50/1.00 | 1.00/1.00 | 1.00/1.00 | 1.00 | 2 both |
| q16 | TOI-270d | 1 | 1.00/1.00 | 0.00/1.00 | 1.00/1.00 | 1.00 | 1 both |
| q17 | WASP-17b | 1 | 1.00/1.00 | 1.00/1.00 | 1.00/1.00 | 1.00 | 1 both |
| q18 | WASP-43b | 2 | 0.00/0.50 | 0.50/1.00 | 0.50/0.50 | 1.00 | 1 both · 1 L-only |
| q19 | CO2 | 1 | 1.00/1.00 | 1.00/1.00 | 1.00/1.00 | 1.00 | 1 both |
| q20 | SO2 | 1 | 1.00/1.00 | 1.00/1.00 | 1.00/1.00 | 1.00 | 1 both |
| q21 | H2O | 2 | 0.00/0.00 | 0.00/0.00 | 0.00/0.00 | 0.00 | 2 neither |
| q22 | CH4 | 1 | 0.00/0.00 | 0.00/1.00 | 0.00/1.00 | 1.00 | 1 L-only |
| q23 | CO | 1 | 1.00/1.00 | 1.00/1.00 | 1.00/1.00 | 1.00 | 1 both |
| q24 | Na | 2 | 0.00/0.50 | 0.00/0.50 | 0.00/1.00 | 1.00 | 1 D-only · 1 L-only |
| q25 | escape | 2 | 0.50/1.00 | 0.00/0.50 | 0.50/0.50 | 1.00 | 1 both · 1 D-only |
| q26 | clouds | 2 | 0.50/0.50 | 0.50/0.50 | 0.50/0.50 | 0.50 | 1 both · 1 neither |
| q27 | retrieval | 2 | 0.50/1.00 | 0.00/1.00 | 1.00/1.00 | 1.00 | 2 both |
| q28 | C/O ratio | 2 | 0.00/0.50 | 0.00/0.50 | 0.00/0.50 | 1.00 | 1 D-only · 1 L-only |
| q29 | emission | 2 | 0.00/0.50 | 0.00/0.50 | 0.00/0.50 | 1.00 | 1 D-only · 1 L-only |
| q30 | phase curve | 2 | 0.00/1.00 | 0.00/0.00 | 0.00/0.50 | 1.00 | 2 D-only |

### Decisive queries — arms disagree on top-10 hit/miss (widened)

| id | target | dense (top10 / full) | lexical (top10 / full) | hybrid (top10) |
|---|---|---|---|---|
| q16 | TOI-270d | #1 / #1 of 2500 | miss / #12 of 2500 | #2 |
| q18 | WASP-43b | miss / #14 of 2500 | #5 / #5 of 2500 | #6 |
| q25 | escape | #1 / #1 of 2500 | miss / #33 of 2500 | #7 |
| q27 | retrieval | #9 / #9 of 2500 | miss / #11 of 2500 | #3 |

## Interpretation

### H5 — confirmed (strongly)

Widening hurts. On the *same* 29 queries with the *same* labels, every arm loses R@10 (dense −0.195,
lexical −0.276, hybrid −0.098), R@50 (−0.12 to −0.17), and MRR (−0.24 to −0.29). The mechanism is
exactly as pre-registered: the 2,000 added papers are distractors that push the (frozen) relevant
docs deeper, and a fixed pool=50 — now 2% of the index instead of 10% — leaks recall that a
proportionally-scaled pool would keep. Coverage gains, had any existed, would have accrued to
newly-scorable queries; there were none (H7). This is the load-bearing controlled result: **scaling
the corpus while holding pool fixed degrades the original queries**, and the degradation is large
enough to be visible even at n=29.

### H6 — confirmed, but via displacement, not dense-blind newcomers

The single-arm-exclusive class is **not** a singleton and **scales with the corpus** — 5 → 15
relevant docs at pool depth, bidirectionally (dense-only 2→8, lexical-only 3→7). The union-recall
premium over the best single arm grew (+0.069 → +0.172), so hybrid stage-1's value as a candidate
generator *increases* with scale: the union holds at 0.948 while each arm collapses toward ~0.79.
**This validates hybrid stage-1 as a growing safety net** — the strategic claim H6 was built to test.

But the *mechanism* is the opposite of H6's literal wording. H6 guessed the class would grow because
*newly-scorable entity targets* land dense-blind like the ERO. There were no newly-scorable entity
queries, and the scan shows newcomers are mostly dual-found near the top (dense #1–#5). The class
grows because **incumbents are displaced**: recent papers crowd the older gold docs out of one arm's
pool-50 (q11's gold → dense #70/lex #33, dense-blind; q12's ERO → dense #1594). The dense-blind
*mechanism* recurs (and is mirrored by lexical-blind high-resolution detections), but the population
that feeds it is the existing relevant set under pressure, not a stream of ERO-like newcomers.

### H7 — falsified by construction (resolved before retrieval)

Neither coverage target enters the widened corpus under the fixed query: q15's LHS 475b is
phrase-excluded (numFound 0 against `abs:"exoplanet atmosphere"`), q11's canonical inversion papers
are year-excluded (2016–2017). Widening the citation cut cannot move a query-level bound. The
coverage miss is a **query-recall** problem, not a corpus-size one — the central, slightly
uncomfortable finding of this PR, and the reason the appendix exists. Closing these gaps needs a
*different* lever (query/recall widening), held out of scope to keep the single-variable control.

### "neither" — moved slightly, as flagged

The present-but-unretrieved class went 2 → 3. The pilot's two pre-registered "neither" docs (q21's
H₂O pair) **stayed neither and moved deeper** — neither entered the top-50 of either arm, and both
fell further out: `2024ApJ...963L...5X` dense #192→#622 / lex #61→#230, and `2018AJ....155...29W`
dense #82→#256 / lex #74→#263. A third joined them: q26's `2023ApJ...951...96G` fell from dense-only
to neither (dense #55 / lex #279). Widening does not fix "neither" — it adds to it by displacement,
exactly the non-effect pre-registered (the original pair did not improve; it got worse).

### Implication for the architecture decision and for PR #6

The pilot's recommendation **holds and hardens**: keep hybrid as the stage-1 candidate generator.
On the widened corpus the case is stronger than on the pilot — hybrid is the best R@10 arm on every
split (robustly over lexical; suggestively, not robustly, over dense — the dense−hybrid R@10 CI is
carried by 4 queries and fails leave-one-out, see the bootstrap caveat), it degrades least under
distractor pressure, and its union candidate set is the only readout that resists corpus growth
(0.966 → 0.948 while single arms
fall ~0.12–0.17). The transferable findings are the **fusion-dilution mechanism** (now visible as the
lost depth-inversion: fused top-50 leaks recall the union keeps, because pool=50 is too shallow a
fraction) and the **growing, bidirectional complementarity**.

Two concrete follow-ups this experiment motivates, both already implied by the pre-registered
confounds:

1. **Sweep the pool** (the deliberately-held confound). pool=50 leaking recall at 2% of the index is
   the most actionable lever here: union R@50 = 0.948 says the recall *exists* in the candidate space;
   the fused top-50 just doesn't capture it. A pool proportional to corpus size is the obvious test.
2. **Widen the query, not just the corpus** (the H7 finding). The coverage gaps (LHS 475b, the
   pre-2018 inversion literature) are query-bounded; object/title resolution or phrase relaxation —
   not a deeper citation cut — is what reaches them.

### Verdict on the pre-registered hypotheses

- **H5 — confirmed (strongly).** All arms lose R@10/R@50/MRR on the fixed 29 queries; the fixed pool
  on a 5× index is the mechanism.
- **H6 — confirmed (strategic claim), mechanism corrected.** The single-arm/dense-blind class scales
  (5→15) and hybrid's candidate-set premium grows (+0.069→+0.172); but it grows by incumbent
  displacement, not by dense-blind newcomers as the literal wording guessed.
- **H7 — falsified by construction.** Coverage targets are query-excluded (q15 phrase, q11 year), not
  size-excluded; no widening of the citation cut admits them. Resolved at the membership stage.
- **"neither" — changed slightly (2→3), by displacement**, as anticipated.

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
| `identifier:(Evans17 OR Sedaghati17 OR Evans16)` ¹ | 3 | the canonical q11 WASP-121b inversion papers exist in ADS … |
| … same AND `year:[2018 TO *]` | **0** | … but **all** fall below the pilot's year cutoff — year-excluded at any citation cut |
| `abs:"exoplanet atmosphere"` AND `full:"WASP-121"` AND `abs:"stratosphere"` | 1 | the lone in-phrase WASP-121b stratosphere paper is itself year-excluded (2016) |

¹ `identifier:("2017Natur.548...58E" OR "2017ApJ...850L..32S" OR "2016ApJ...822L...4E")` — Evans et al.
2017 (stratosphere), Sedaghati et al. 2017 (dayside thermal inversion), Evans et al. 2016 (TiO/VO).
This is the q11 mirror of q15's `numFound = 0` probe: the targets are recorded by identifier, then
shown to vanish under the year filter — the exclusion is a measured fact, not an assertion.

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
