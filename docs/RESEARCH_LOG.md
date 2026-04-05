# AstroLLM Research Log

A living document tracking experiments, findings, and decisions.

> **Tip**: Use the `/research-log` skill in Claude Code to add structured entries automatically.

---

## Experiment Template

```
### EXP-XXX: [Title]
**Date**: YYYY-MM-DD
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
| Metric | Base Model | This Run | Delta |
|--------|-----------|----------|-------|
| AstroMLab-1 | | | |
| Astro-QA | | | |
| Perplexity (astro) | | | |

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
- [ ] Qwen3-8B (primary base model)
- [ ] Qwen3-4B (smaller experiment)
- [ ] Qwen 2.5 7B Instruct (comparison)
- [ ] Mistral 7B v0.3 Instruct (comparison)

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
