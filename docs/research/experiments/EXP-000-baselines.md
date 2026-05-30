# EXP-000: Baseline Measurements

**Date**: TBD
**Status**: Pending

## Goal

Establish baseline scores for unmodified base models on our evaluation suite. These baselines are the floor that every fine-tuned model must beat.

## Models to Evaluate

| Model | Parameters | Why |
|-------|-----------|-----|
| Qwen3.5-9B | 9B | Primary base model for AstroLLM Core |
| Qwen3.5-4B | 4B | Smaller experiment target; informs future Nano tier |
| Gemma 4 E4B | ~4B (E) | Track B cross-family comparison arm |
| Qwen 2.5 7B Instruct | 7B | Prior-generation comparison point |

## Benchmarks

- AstroMLab-1 (subset: ~500 MCQs)
- Custom grounding accuracy (25+ examples)
- Custom tool routing accuracy (25+ examples)
- Custom abstention under weak retrieval (25+ examples)
- Custom pedagogy quality (25+ examples)
- Perplexity on held-out astronomy text

## Results

| Metric | Qwen3.5-9B | Qwen3.5-4B | Gemma 4 E4B | Qwen 2.5 7B |
|--------|------------|------------|-------------|-------------|
| AstroMLab-1 | — | — | — | — |
| Grounding | — | — | — | — |
| Tool routing | — | — | — | — |
| Abstention | — | — | — | — |
| Pedagogy | — | — | — | — |
| Perplexity | — | — | — | — |

*To be filled after running baselines.*

## Notes

- Run on identical hardware for fair comparison
- Use the same prompt templates and retrieval context across all models
- Log all results to W&B under project `astrollm`, tag `baseline`
