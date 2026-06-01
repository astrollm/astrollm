# AstroLLM Research Log

A living document tracking experiments, findings, and decisions.

!!! tip "Claude Code users"
    Use the `/research-log` command to add structured entries automatically.

---

## Experiment Template

```
### EXP-XXX: [Title]
**Date**: YYYY-MM-DD
**Type**: training | retrieval | data | evaluation | infrastructure
**W&B Run**: [link]
**Config**: configs/[filename].yaml

**Hypothesis**: What we expect to happen and why.

**Setup**:
- Base model:
- Method:
- Dataset:
- GPU:
- Training time:

**Results**:
| Metric | Base Model | Previous Best | This Run | Delta |
|--------|-----------|---------------|----------|-------|
| AstroMLab-1 (subset) | | | | |
| Grounding accuracy | | | | |
| Tool routing F1 | | | | |
| Abstention recall | | | | |
| Pedagogy score | | | | |

**Observations**: What we learned.

**Next Steps**: What to try based on these results.
```

---

## Experiments

### EXP-000: Baseline Measurements
**Date**: TBD
**Status**: Pending

**Goal**: Establish baseline scores for unmodified base models on our evaluation suite.

**Models to evaluate**:
- [ ] Qwen3.5-9B (primary base model)
- [ ] Qwen3.5-4B (smaller experiment)
- [ ] Gemma 4 E4B (Track B cross-family comparison)
- [ ] Qwen 2.5 7B Instruct (prior-generation comparison)

**Metrics**: AstroMLab-1, Astro-QA, custom pedagogy eval, perplexity on held-out astro text

---

> **Retrieval experiments (EXP-001 – EXP-003) use a condensed entry.** The EXP-XXX template above is
> training-shaped (W&B run, GPU, AstroMLab metrics), which doesn't fit a retrieval ablation, so these
> log the date / type / status / outcome and link the per-experiment doc for full method,
> pre-registration, tables, and per-query results.

### EXP-001: Pilot retrieval ablation — dense vs lexical vs hybrid
**Date**: 2026-05-31 · **Type**: retrieval · **Status**: complete (PR #5) · **Detail**: [pilot-ablation.md](research/pilot-ablation.md)

500-abstract ADS corpus (`abs:"exoplanet atmosphere"`, year ≥ 2018), 29-query reviewed gold. Three
arms over one frozen index — dense (pgvector/BGE), lexical (FTS5/BM25), hybrid (RRF; k=10, pool=50,
RRF_K=60). Hybrid wins MRR on every split; the Recall@10 ordering sits within the paired-bootstrap
noise at n=29. The decision driver was **candidate-set recall**, not Recall@10: union recall at pool
depth = 0.966, and q12's dense-blind ERO (dense #338 / lexical #4) is an existence proof that lexical
recovers targets dense buries. **Outcome: hybrid selected as the stage-1 candidate generator.**

### EXP-002: Corpus widening (500 → 2,500), method-fixed
**Date**: 2026-05-31 · **Type**: retrieval · **Status**: complete (PR #6, merge `d84f3ff`) · **Detail**: [corpus-widening.md](research/corpus-widening.md)

Same query + year, deeper citation cut; the original 500 a byte-identical strict subset. Vary the
corpus, hold the method.

- **H5 confirmed** — every arm loses Recall@10 at fixed pool=50 (hybrid least, −0.098; dense −0.195;
  lexical −0.276), and hybrid becomes the best Recall@10 arm.
- **H6 confirmed, via incumbent displacement** — the single-arm-exclusive class scales 5 → 15, the
  union premium grows +0.069 → +0.172, and union R@50 holds 0.948.
- **H7 falsified by construction** — the coverage premise collapsed: q15 (LHS 475b) is
  phrase-excluded and q11's canonical inversion paper is year-excluded, so a deeper citation cut
  yields **zero coverage win** (the bottleneck is a query-recall bound, not corpus size).
- Edge robustness: the dense−hybrid Recall@10 edge is **fragile** (carried by 4 queries, collapses
  under leave-one-out); the lexical−hybrid edge is **robust** (survives all 29 LOO drops).

### EXP-003: Pool-depth sweep (frozen 2,500 index)
**Date**: 2026-06-01 · **Type**: retrieval · **Status**: complete (PR #7, merge `22d2793`) · **Detail**: [pool-sweep.md](research/pool-sweep.md)

Frozen-index ablation (no re-index); sweep only `pool ∈ {50, 100, 200, 500}`.

- **H8 confirmed (strongly)** — candidate-union recall rises to **1.000** with depth, but **fused
  top-10 is exactly pool-invariant** (0.592 all / 0.716 named) and fused top-50 barely moves
  (non-monotonically); the gap does not close. The bottleneck is **RRF fusion demotion of gold that
  is already in the candidate set**, not candidate absence.
- **H9 falsified** — both arm edges are pool-invariant, which is *entailed* by the top-10 invariance
  (not an independent result): the fragile dense−hybrid edge stays fragile and the robust
  lexical−hybrid edge stays robust at every depth.
- **H10 lever call** — the lever is stage-2 reranking, not pool depth; operating point **pool = 100**.

### EXP-004: RAG-SFT pilot — curation-recipe pre-registration (Phase A)
**Date**: 2026-06-01 · **Type**: data · **Status**: pre-registered (PR #10), no data generated · **Detail**: [sft-pilot.md](research/sft-pilot.md)

Executes the 2026-06-01 pivot (Decision Log below): from the closed retrieval thread to SFT data
curation, the active critical path for the week-12 beta. Phase A holds the exoplanet-atmosphere
corpus **frozen** (the 2,500-abstract pool-sweep index, context assembled hybrid RRF @ pool=100) and
treats the **curation recipe** as the single manipulated variable — gold seed (150–250 hand-curated,
disjoint verifier-calibration / eval-seed partitions) → teacher synthesis (Claude API, grounded on
the frozen `retrieve()`) → in-repo citation-grounding verification → manifest + 95/5 split by task
family. Same pilot→widen, single-variable discipline as the retrieval thread; Phase B (separate
commission) freezes the recipe and widens the corpus to a general astro-ph slice for the shippable v1
set. Every example is retrieval-augmented `(query + retrieved abstracts) → cited answer` across three
families — literature-grounded QA (~45%), citation-grounded summarization (~35%), explicit abstention
(~20%, deliberately over-weighted).

Five hypotheses registered **before any data is generated**; comparison is always QLoRA-SFT
Qwen3.5-4B vs the same base on the held-out eval-seed set, with paired-bootstrap CIs as in the
retrieval thread:

- **SFT-H1 (grounding lift)** — CI on (SFT − base) faithfulness excludes 0, target ≥ +0.10 absolute.
- **SFT-H2 (citation accuracy)** — cited-bibcode-supports-claim clears the V1 >80% target and beats base.
- **SFT-H3 (abstention, two-sided)** — refusal rate beats base, subject to a false-refusal cap ≤ 0.10.
- **SFT-H4 (no knowledge regression)** — AstroMLab-1 subset drops ≤ 2 pp vs base (larger = kill signal).
- **SFT-H5 (verifier validity)** — verifier precision/recall vs gold; trusted as a filter only at
  precision ≥ 0.85, else every "passed" example is human-reviewed.

**Gate**: fine-tune into the beta only if H1 ∧ H2 ∧ H4 hold; otherwise the honest status is
"RAG-grounded beta on the base model, SFT iterating" — the week-12 date does not override the gates.
Pre-registration only: nothing is fine-tuned and no SFT data is curated yet. Results /
Interpretation / Verdict append at the results step (separate commit).

---

## Key Learnings

### From AstroSage Papers
1. **Catastrophic forgetting is real**: Early AstroLLaMA (7B) scored LOWER than base LLaMA after CPT
2. **Data quality > quantity**: Curated, diverse SFT data matters more than raw corpus size
3. **Model merging saves capabilities**: SLERP/TIES merging of SFT model with instruct model preserves reasoning
4. **Loss masking is essential**: Train only on assistant completions, not user prompts
5. **Synthetic data works**: Generated Q&A pairs with randomized question styles improve diversity
6. **Metadata-based Q&A helps**: Questions about paper titles, dates, arXiv IDs build factual grounding

### From "Talking with the Latents" (2026)
1. **Teacher-student distillation**: Large LLM generates synthetic Q&A → smaller LLM learns from it
2. **LoRA is sufficient**: Lightweight adapters + frozen base achieves strong domain performance
3. **Latent feature fusion**: Physical features can be injected into LLMs via adapters

---

## Decision Log

| Date | Decision | Rationale |
|------|----------|-----------|
| TBD | Start with Qwen3-8B | Apache 2.0 license, strong fine-tuning results at 4B/8B scale, dual thinking mode |
| TBD | QLoRA for initial experiments | Cloud-only, need to minimize GPU costs per experiment |
| TBD | Researcher-first audience | Highest value differentiation, tool integration is unique angle |
| 2026-04-23 | Upgrade base models from Qwen3-4B/8B to **Qwen3.5-4B/9B** | Qwen3.5 released Feb 2026 (Apache 2.0, natively multimodal, 262K ctx). Same family preserves plan/tooling/eval continuity. V1_FINAL_PLAN.md remains frozen; this Decision Log entry overrides it. |
| 2026-04-23 | Add **Gemma 4 E4B** as Track B comparison arm | Gemma 4 released Apr 2, 2026. Unsloth day-0 support (~2× faster training, ~60% less memory). Cheap cross-family data point via existing week 7-8 experiment matrix. Dense variant only — Gemma 4 MoE has a known 3D-tensor QLoRA bug. |
| 2026-04-24 | **Watch**: Qwen3.6 dense 4B/9B variants | As of 2026-04-24, only Qwen3.6-27B (dense, coding-focused) and Qwen3.6-35B-A3B (MoE) are released as open weights; 4B/9B are "expected" but not shipped. If dense 4B/9B land before week 7, revisit base-model choice. Otherwise ship v1 on Qwen3.5 and evaluate 3.6 in Phase 2. |
| 2026-05-30 | Repo-wide reconcile to Qwen3.5 base + verified HF repo IDs | Propagated the Qwen3.5-4B/9B + Gemma 4 E4B Track-B decision across all configs, docs, personas, and skill commands, superseding the original Qwen3-4B/8B (and earlier Llama 3.1) references. Note 9B, not 8B — Qwen3.5 has no 8B dense. Verified exact HF repo IDs against Hugging Face: `Qwen/Qwen3.5-9B` and `Qwen/Qwen3.5-4B` (post-trained; no `-Instruct` suffix, `-Base` = pretrained), and `google/gemma-4-E4B-it` (capital `E4B`). |
| 2026-05-31 | Docker Compose project-name collision (resolved) | During pilot stack bringup, `docker compose` derived the project name "docker" from the `docker/` directory and collided with an unrelated local Docker project that derives the same name; Compose recreated that project's database container against the pilot volume. Remediated by rebinding the other project's data volume and re-isolating the pilot stack. Structural fix: added a top-level `name: astrollm-pilot` key to `docker/docker-compose.pilot.yml` so isolation is a property of the file, not a `-p` flag (landed in PR #3, merge cfcd21b). **OPEN**: independent integrity verification of the other project's database (row counts) is still pending — the self-remediation does not close it. |
| 2026-05-31 | Pilot label review: eval baseline updated | First-pass eval (PR #3): Recall@10 0.812 / MRR 0.776 over 16 queries with title-level entity-matched labels. After full review (PR #4; 29/30 scored, q15 has no in-corpus target): overall Recall@10 0.690 / MRR 0.623; named-target (n=17) 0.794 / 0.794; broad known-item (n=12) 0.542 / 0.381. The headline moved **down** by intent — the review added an honest q11 0.00 miss and corrected mislabels in both directions rather than inflating. **Decision**: the reviewed gold set is the eval baseline going forward. **Finding** (treat as DIRECTIONAL): hybrid retrieval is markedly weaker on conceptual/topical queries than on named entities — but the two sets used different labeling protocols (broad = single known-item landmarks, so its number is optimistic) and n is small (12 broad / 17 named), so the pattern is robust while the decimals are noisy. Motivates the dense-vs-lexical-vs-hybrid ablation and the stage-2 reranker. |
| 2026-06-01 | **Retrieval thread closed** — beta ships hybrid RRF stage-1 @ pool=100 | Stage-1 candidate generation is **not** the binding constraint (union recall ≥ 0.966, and 1.000 for named queries — see EXP-001/002/003); the bottleneck is **fusion ranking**. The beta default is therefore **hybrid RRF stage-1 at pool=100** (saturates the recoverable fused recall at ~157 candidates, per EXP-003). Fusion-ranking improvement — the ladder **RRF_K sweep → weighted/score-based fusion → cross-encoder reranking** — is **deferred to the post-beta retrieval cycle** (aligns with V1_FINAL_PLAN placing Stage-2 reranking post-week-12). The **query-widening** lever (to reach the query-excluded coverage targets q15/q11) is also deferred. The active critical path **pivots to SFT data curation** (plan weeks 5-6). Retrieval was stopped because it is **ship-good, not because it failed**. **Dissemination**: PR #6 → blog Post 3, PR #7 → blog Post 4 (nandan.me/writing/notes, "building on a budget" series); drafting deferred, tracked. |
