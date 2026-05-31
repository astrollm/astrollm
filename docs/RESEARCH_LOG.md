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
