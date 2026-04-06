# Experiments

Detailed reports for each experiment. Each experiment gets its own page when results are significant enough to warrant deeper analysis.

## Experiment Index

| ID | Title | Date | Status | Key Finding |
|----|-------|------|--------|-------------|
| EXP-000 | [Baseline Measurements](EXP-000-baselines.md) | TBD | Pending | — |

## How to Add an Experiment

1. Create a new file: `docs/research/experiments/EXP-XXX-short-name.md`
2. Use the template below
3. Add a row to the index table above
4. Link from the [Research Log](../../RESEARCH_LOG.md)

## Experiment Report Template

```markdown
# EXP-XXX: Title

**Date**: YYYY-MM-DD
**W&B Run**: [link to W&B dashboard]
**Config**: `configs/filename.yaml`
**Commit**: `abc1234`

## Hypothesis

What we expect to happen and why. (For baseline experiments, use `## Goal` instead.)

## Setup

| Parameter | Value |
|-----------|-------|
| Base model | Qwen3-8B |
| Method | QLoRA r=64 |
| Dataset | data/sft/train.jsonl (N examples) |
| GPU | RTX 4090 (RunPod spot) |
| Training time | X hours |
| Cost | $X |

## Results

| Metric | Base Model | Previous Best | This Run | Delta |
|--------|-----------|---------------|----------|-------|
| AstroMLab-1 | | | | |
| Grounding accuracy | | | | |
| Tool routing F1 | | | | |
| Abstention recall | | | | |
| Pedagogy score | | | | |

### Loss Curves

[Embed or link W&B charts here]

### Example Outputs

Show 2-3 representative question-answer pairs comparing base vs fine-tuned.

## Analysis

What worked, what didn't, and why.

## Error Analysis

Breakdown by error taxonomy category.

## Next Steps

What to try based on these results.
```
