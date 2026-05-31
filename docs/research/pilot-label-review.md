# Pilot Retrieval — Relevance Label Review

Follow-up to PR #3 (`docs/lab/week-01.md`). The pilot's `expected_bibcodes` in
`packages/evaluation/queries/pilot_exoplanet_atmospheres.yaml` were a first-pass, title-level
labeling. This doc records the domain review that turns them into vetted labels, one query at a
time, with the reasoning kept so the eval number is defensible.

**Method (per query):** list every corpus paper matching the target entity (independent of the
ranker), read the abstracts, and decide keep / remove / add against the query's *intent*. Note
genuine retrieval misses (relevant paper exists but isn't retrieved) separately from labeling
errors — only the latter change the labels.

Headline number is unchanged so far: **Recall@10 = 0.812, MRR = 0.776** over 16 scored queries.

## Status

- [x] **q12** — WASP-96b (reviewed below)
- [ ] q07 — HD 189733b (R@10=1.0 but RR 0.17 — relevant paper ranked deep)
- [ ] q08 — HD 209458b (R@10=1.0 but RR 0.25)
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
