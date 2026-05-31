# Pilot Retrieval — Relevance Label Review

Follow-up to PR #3 (`docs/lab/week-01.md`). The pilot's `expected_bibcodes` in
`packages/evaluation/queries/pilot_exoplanet_atmospheres.yaml` were a first-pass, title-level
labeling. This doc records the domain review that turns them into vetted labels, one query at a
time, with the reasoning kept so the eval number is defensible.

**Method (per query):** list every corpus paper matching the target entity (independent of the
ranker), read the abstracts, and decide keep / remove / add against the query's *intent*. Note
genuine retrieval misses (relevant paper exists but isn't retrieved) separately from labeling
errors — only the latter change the labels.

**Review complete (28 of 30 queries scored; q15 has no in-corpus target).** Final numbers,
reported split because the two query kinds mean different things:

| set | n | Recall@10 | MRR |
|---|---|---|---|
| named-target (q01–18) | 17 | **0.794** | **0.794** |
| broad known-item (q19–30) | 12 | **0.542** | **0.381** |
| all scored | 29 | 0.690 | 0.623 |

The 16-query named set peaked at 0.844/0.844 mid-review; it settles to **0.794** once **q11** is
included as a scored *miss* (see below). Discipline held throughout: relevance judged from
abstracts not rank (q08's hybrid #1 was rejected), and genuine misses were kept rather than
relabeled upward. **Headline finding:** the hybrid pilot is markedly weaker on conceptual/topical
queries (0.542) than on named entities (0.794) — and the broad number is *optimistic* (known-item
exemplars, not exhaustive recall).

## Status — complete

- [x] **q12** — WASP-96b (off-target label removed)
- [x] **q07 / q08** — HD 189733b / HD 209458b (added missed on-target papers)
- [x] **q03, q05, q06, q18** — the four 0.50 queries (only q06 relabeled)
- [x] **q11 / q15** — WASP-121b (labeled, scored miss) / LHS 475b (absent, unscored)
- [x] **q19–q30** — broad queries labeled as known-item exemplars
- [ ] q11 (WASP-121b), q15 (LHS 475b) — currently empty; confirm no in-corpus target
- [ ] q19–q30 — broad molecule/process queries; build relevant sets

## q12 — "cloud-free water detection in WASP-96b with NIRISS"

**Corpus has exactly 2 papers mentioning WASP-96:**

| bibcode | title | verdict |
|---|---|---|
| `2022ApJ...936L..14P` | *The JWST Early Release Observations* | **keep** — the WASP-96b NIRISS cloud-free water spectrum was released here |
| `2021AJ....161....4Y` | *On the Compatibility of Ground-based and Space-based Data: WASP-96 b, an Example* | **remove** — data-combination methodology, pre-JWST/NIRISS; not a water detection |

**Label change:** `expected_bibcodes` → `["2022ApJ...936L..14P"]`. The headline number is
unaffected (q12 scored 0/2 before, 0/1 now — both 0.00), but it now measures against the correct
relevant set.

**Why it's a genuine miss (not a labeling fix):** the ERO paper's *abstract* is generic
public-outreach text — it never describes the WASP-96b science — so it embeds far from the query
(**dense #338**) while matching lexically (**#4**). The other (now-removed) paper was the mirror
image (**dense #1, lexical #73**). RRF with a pool of 50 rewards papers ranked decently in *both*
arms, so a paper that is strong in only one arm and outside the other arm's pool falls below the
top-10 cutoff. Both WASP-96b papers were single-arm-strong, so q12 retrieves thematically similar
NIRISS/water papers about *other* planets instead.

**Implications (for the planned ablation / later work):**
- Lexical-only would have surfaced the ERO at #4 — a concrete case where hybrid RRF *underperforms*
  a single arm. Worth quantifying in the dense-vs-lexical-vs-hybrid ablation.
- Abstract-only indexing cannot represent a multi-target release paper's per-target science. This
  is exactly what section-aware / full-text chunking (weeks 5–10) is meant to fix; noting it here
  as motivating evidence, not acting on it now.

## q07 — "sodium absorption and haze in HD 189733b"  (RR 0.17 → 1.00)

The query has two facets (sodium, haze); the first pass labeled only the haze paper.

| bibcode | title | verdict |
|---|---|---|
| `2018A&A...612A..53P` | *Combining low- to high-resolution transit spectroscopy of HD 189733b* | **add** — abstract is about HD 189733b "alkali doublets / cores of the alkali lines" (= Na); the missed sodium facet |
| `2024ApJ...973L..41I` | *Quartz Clouds in the Dayside Atmosphere of … HD 189733 b* | **keep** — the haze facet |
| `2021MNRAS.502.5643L` | *Impact of photochemical hazes … thermal structure* | skip — general hot-Jupiter haze modeling, not HD 189733b-specific |

**Label:** q07 → `["2018A&A...612A..53P", "2024ApJ...973L..41I"]`. Both retrieved (#1, #6), so
recall stays 1.0 and RR goes 0.17 → 1.00. The added paper is the on-target sodium result, judged
from its abstract (it also happens to rank #1).

## q08 — "water vapor and hydrogen escape from HD 209458b"  (RR 0.25 → 0.50)

Two facets (water vapor, hydrogen escape); the first pass labeled only two helium-escape papers.

| bibcode | title | verdict |
|---|---|---|
| `2024ApJ...963L...5X` | *JWST Transmission Spectroscopy of HD 209458b: Supersolar Metallicity…* | **add** — "we detect strong features of H₂O"; the missed water-vapor facet |
| `2020AJ....159..111C` | *Near-UV Transmission Spectroscopy of HD 209458b: … Beyond the Roche Lobe* | **add** — "escaping hydrogen and metals in the upper atmosphere"; the escape facet, HD 209458b-specific |
| `2020A&A...636A..13L` | *Modelling the He I triplet … in the atmosphere of HD 209458b* | **keep** — He triplet probes HD 209458b's escaping upper atmosphere |
| `2018ApJ...855L..11O` | *A New Window into Escaping Exoplanet Atmospheres: 10830 Å He* | **remove** — general He-line *method* paper, not an HD 209458b result |

**Label:** q08 → `["2024ApJ...963L...5X", "2020AJ....159..111C", "2020A&A...636A..13L"]`. All three
retrieved (#2, #3, #4) → recall 1.0, RR 0.25 → 0.50. **Not** added: q08's hybrid #1
`2020ApJ...890...79J` ("Hydrodynamic Escape of Water Vapor Atmospheres near Very Active stars") —
it is general, not HD 209458b, so it stays out despite ranking first. Removing the general method
paper (which sat at #7) while recall holds shows the labels track relevance, not rank.

## The four 0.50 partial-recall queries (q03, q05, q06, q18)

Each found one of two labeled papers. The review question is whether the *missed* paper is a
genuine recall miss (keep) or a mislabel (remove). Verdict: three are genuine misses, one is a
mislabel.

| query | missed paper (rank) | verdict |
|---|---|---|
| **q03** K2-18b CH₄/CO₂ | `2023ApJ...956L..13M` *Carbon-bearing Molecules in a Possible Hycean Atmosphere* (dense#18 / lex#11) | **keep** — this is the actual K2-18b detection paper; a real miss. Notably the **rebuttal** (`2024ApJ...963L...7W`) ranks #1 while the detection ranks ~#11–18. |
| **q05** TRAPPIST-1e atmosphere | `2020ApJ...901..126W` *Distinguishing Wet and Dry Atmospheres of TRAPPIST-1 e/f* (dense#11 / lex#18) | **keep** — genuinely on-topic; a real miss just outside the pool. |
| **q06** TRAPPIST-1b emission | `2023ApJ...955L..22L` *Atmospheric Reconnaissance … Strong Stellar Contamination* (dense#10 / lex#42) | **remove** — a *transmission*/contamination study, not dayside emission. Bullseye `2023ApJ...952L...4I` (secondary eclipse) is #1, so q06 → recall 1.0. |
| **q18** WASP-43b phase curve | `2018ApJ...869..107M` *3D Circulation Driving Chemical Disequilibrium in WASP-43b* (dense#25 / lex#16) | **keep** — circulation directly underlies phase curves (same observable domain); a real miss. The corpus has no nightside-clouds phase-curve paper (Bell 2024 isn't in this top-500-by-citation slice). |

**Net:** only q06 relabeled → Recall@10 0.812 → **0.844**; MRR stays 0.844 (q06's bullseye was
already #1). q03/q05/q18 stay honest 0.50 misses — kept deliberately rather than relabeled upward.

The q06-vs-q18 contrast is the principle: q06's missed paper is the *wrong observable*
(transmission ≠ dayside emission) so it's removed; q18's missed paper is the *right observable*
(circulation → phase curve) so it stays a genuine miss.

## q11 / q15 — the two previously-empty named queries

- **q11** (WASP-121b thermal inversion / metal emission): labeled `2023ApJ...943L..17M` (the JWST
  NIRSpec WASP-121b dayside-emission phase curve — the corpus's emission paper; the canonical
  optical Fe-emission/inversion papers predate the 2018 cutoff). It is **not retrieved** in the top
  10 for the query, so q11 scores **0.00** — a genuine miss now on the books (previously excluded).
- **q15** (LHS 475b): confirmed **no LHS 475b paper** in this top-500-by-citation slice (Lustig-Yaeger
  et al. 2023 didn't make the cut). Left **unscored** — you can't measure recall with no relevant doc.

## q19–q30 — broad / topical queries (known-item labeling)

These ask about a molecule or process, not a named planet, so the true relevant set is large and
graded. Labeling it exhaustively isn't feasible (or fair to Recall@10, which caps at `10/|relevant|`).
Instead each is labeled with **1–2 known-item landmark papers** — the unambiguous canonical result
for that topic, verified present in the corpus and on-topic by abstract (e.g. q19 CO₂ → the WASP-39b
"Identification of carbon dioxide" paper; q27 retrieval → the Aurora and PYRAT BAY frameworks). The
score therefore answers *"does the landmark surface in the top 10?"*, **not** true topical recall.

**Result: 0.542 Recall@10 / 0.381 MRR over the 12 — clearly below the named-target set (0.794).**
Conceptual queries are harder for the hybrid pilot than entity lookups; the landmark often isn't in
the top 10 (e.g. q11-style misses recur). Two caveats: this is *optimistic* (exemplars, not full
recall), and a proper broad-query eval needs **pooled precision@k** with human judgments — flagged
as future work, not built here.

## Outcome & follow-ups

- Labels in `pilot_exoplanet_atmospheres.yaml` are now reviewed (28/30 scored; q15 absent).
- Numbers: named-target **0.794 / 0.794**, broad known-item **0.542 / 0.381**, all-scored 0.690 / 0.623.
- Strongest signals for next work: (1) the **named-vs-conceptual gap** motivates reranking / query
  understanding; (2) recurring **single-arm-strong misses** (q12, q11) motivate the dense-vs-lexical-vs-hybrid
  ablation and a larger fusion pool; (3) **abstract-only indexing** misses multi-target release papers
  (q12) — evidence for section-aware/full-text chunking (weeks 5–10).
