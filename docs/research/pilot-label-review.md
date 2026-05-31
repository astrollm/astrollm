# Pilot Retrieval — Relevance Label Review

Follow-up to PR #3 (`docs/lab/week-01.md`). The pilot's `expected_bibcodes` in
`packages/evaluation/queries/pilot_exoplanet_atmospheres.yaml` were a first-pass, title-level
labeling. This doc records the domain review that turns them into vetted labels, one query at a
time, with the reasoning kept so the eval number is defensible.

**Method (per query):** list every corpus paper matching the target entity (independent of the
ranker), read the abstracts, and decide keep / remove / add against the query's *intent*. Note
genuine retrieval misses (relevant paper exists but isn't retrieved) separately from labeling
errors — only the latter change the labels.

Headline: **Recall@10 = 0.812** (unchanged — per-query recall@10 stays 1.0 for the reviewed
queries), **MRR 0.776 → 0.844**. The MRR rise comes from q07/q08, where the first pass had
labeled only secondary papers and missed the genuinely on-target ones; correcting that moves the
first relevant hit up. Discipline check: relevance is judged from abstracts, and q08's hybrid #1
(`2020ApJ...890...79J`, general water-escape, not HD 209458b) was **rejected** despite its rank.

## Status

- [x] **q12** — WASP-96b
- [x] **q07** — HD 189733b (RR 0.17 → 1.00)
- [x] **q08** — HD 209458b (RR 0.25 → 0.50)
- [ ] q03, q05, q06, q18 — the four 0.50 (partial-recall) queries
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
