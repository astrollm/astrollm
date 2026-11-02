# D1 — Serving Context Cut Under the Context Budget

Generated 2026-11-02 by `packages/evaluation/src/pool_sweep.py --d1-cut` (raw: [`d1-context-cut-results.json`](d1-context-cut-results.json)). Decision D1 of [Plan Revision 2](../V1_PLAN_R2.md): score the fused hybrid pool-100 ranking at cuts {10, 20, 30, 50}, rendered exactly as the model sees it, under a 16,384-token context budget; overflow drops whole abstracts from the bottom; recall is scored on the delivered context.

## Setup

- Ranking: fused hybrid RRF, pool=100 per arm, full-depth exact arm rankings truncated to pool (the gated pool-sweep ranking); RRF_K = 60; corpus 2,500 abstracts (`sha256:08736961f2bf…`).
- Rendering: sft/context.render_context(assembled, snapshot, limit=n) — includes the 'query:' and 'retrieved … pool=100' header lines.
- Tokens: `Qwen/Qwen3.5-4B` tokenizer (revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`), `add_special_tokens=False`, context rendering only; system prompt + question reserved separately (1,024), answer allowance 2,048.
- Gold: the frozen 29-query set (17 known-item, 12 broad-entity; unscored: q15).

## Validation gates (all passed before any D1 number was computed)

- pool=50 reproduces PR #6: dense R@10 0.529, lexical R@10 0.437, hybrid R@10 0.592, hybrid R@50 0.793, union 0.948.
- pool=100 hybrid R@10 / R@50 equal `docs/research/pool-sweep-results.json` exactly: all 0.592 / 0.862, known-item 0.716 / 0.941, broad-entity 0.417 / 0.750.
- Live pgvector and FTS bibcode sets equal the corpus snapshot.

## Result

| k | delivered recall (all) | known-item | broad-entity | median tokens | p95 tokens | % queries overflowing | mean abstracts dropped |
|---|---|---|---|---|---|---|---|
| 10 | 0.592 | 0.716 | 0.417 | 4,593 | 5,035 | 0.0% | 0.00 |
| 20 | 0.655 | 0.765 | 0.500 | 8,883 | 9,673 | 0.0% | 0.00 |
| 30 | 0.707 | 0.824 | 0.542 | 13,281 | 15,127 | 0.0% | 0.00 |
| 50 | 0.741 | 0.853 | 0.583 | 22,238 | 25,005 | 100.0% | 13.34 |

Token columns are the **rendered top-k (pre-drop)** context tokens over all scored queries (p95: linear interpolation). Known-item / broad-entity columns are delivered recall on those partitions (broad-entity is a coverage measurement, not true recall). Max rendered tokens: k=10 5,146, k=20 10,578, k=30 16,099, k=50 25,753 (budget 16,384).

**Candidate per pre-registered rule: k = 30** — branch `within_tolerance`: smallest k with delivered recall within 0.05 of delivered R@50 (0.741); gap +0.034.

### Overflow log (whole abstracts dropped from the bottom)

| query | k | rendered tokens | dropped | delivered tokens |
|---|---|---|---|---|
| q01 | 50 | 21,319 | 12 | 16,254 |
| q02 | 50 | 22,016 | 15 | 16,276 |
| q03 | 50 | 23,937 | 15 | 16,145 |
| q04 | 50 | 22,706 | 14 | 16,126 |
| q05 | 50 | 21,195 | 11 | 16,027 |
| q06 | 50 | 22,666 | 14 | 16,106 |
| q07 | 50 | 22,238 | 13 | 16,313 |
| q08 | 50 | 21,150 | 9 | 16,274 |
| q09 | 50 | 19,898 | 8 | 16,010 |
| q10 | 50 | 23,790 | 12 | 16,005 |
| q11 | 50 | 22,666 | 14 | 16,144 |
| q12 | 50 | 24,731 | 20 | 16,099 |
| q13 | 50 | 22,651 | 14 | 16,042 |
| q14 | 50 | 20,701 | 12 | 16,124 |
| q16 | 50 | 25,753 | 18 | 16,293 |
| q17 | 50 | 22,379 | 13 | 16,309 |
| q18 | 50 | 22,157 | 16 | 16,157 |
| q19 | 50 | 22,256 | 13 | 16,287 |
| q20 | 50 | 25,047 | 15 | 15,969 |
| q21 | 50 | 21,691 | 13 | 15,987 |
| q22 | 50 | 24,942 | 17 | 16,094 |
| q23 | 50 | 21,826 | 14 | 15,942 |
| q24 | 50 | 22,570 | 14 | 15,938 |
| q25 | 50 | 20,381 | 9 | 16,341 |
| q26 | 50 | 20,565 | 11 | 16,178 |
| q27 | 50 | 21,212 | 12 | 16,263 |
| q28 | 50 | 22,894 | 15 | 16,314 |
| q29 | 50 | 21,728 | 12 | 16,073 |
| q30 | 50 | 20,885 | 12 | 16,175 |

## Serving-path diagnostic (not the pre-registered measurement)

The same rendering, budget and overflow policy applied to what `pilot_retrieval.retrieve(arm="hybrid", k=100, pool=100)` returns live (the call `sft/context.assemble_context` makes). pgvector 0.8.7; the dense arm runs exact kNN (index scan disabled per query, so `hnsw.ef_search` = 40 does not bound it): top-100 identical to the harness ranking on 29/29 queries.

| k | serving delivered recall (all) | harness delivered recall (all) | serving % overflowing |
|---|---|---|---|
| 10 | 0.592 | 0.592 | 0.0% |
| 20 | 0.655 | 0.655 | 0.0% |
| 30 | 0.707 | 0.707 | 0.0% |
| 50 | 0.741 | 0.741 | 100.0% |
