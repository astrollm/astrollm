# /research-log — Add a Structured Experiment Entry

Add a new experiment entry to `docs/RESEARCH_LOG.md` following the project's standard template.

## Usage
```
/research-log [experiment_title]
```

Examples:
- `/research-log "First QLoRA SFT on Qwen3-4B"`
- `/research-log "Hybrid retrieval vs dense-only comparison"`
- `/research-log "SPECTER2 vs GTE-Qwen2 embedding quality"`

## Workflow

1. Read `docs/RESEARCH_LOG.md` to find the next EXP-XXX number
2. Ask the user for:
   - **Hypothesis**: What do you expect to happen and why?
   - **Setup**: Base model, method, dataset, GPU, estimated training time
   - **Type**: training | retrieval | data | evaluation | infrastructure
3. Generate a structured entry using the template below
4. Append it to `docs/RESEARCH_LOG.md` under the `## Experiments` section
5. If this is a training experiment, suggest creating a config YAML via `/train`

## Entry Template

```markdown
### EXP-{XXX}: {Title}
**Date**: {YYYY-MM-DD}
**Type**: {training | retrieval | data | evaluation | infrastructure}
**W&B Run**: [pending]
**Config**: {configs/filename.yaml or N/A}

**Hypothesis**: {What we expect to happen and why}

**Setup**:
- Base model: {e.g., Qwen3-8B}
- Method: {e.g., QLoRA r=64}
- Dataset: {e.g., data/sft/train.jsonl (5,000 examples)}
- GPU: {e.g., RTX 4090 spot on RunPod}
- Estimated time: {e.g., 6-8 hours}

**Results**:
| Metric | Base Model | Previous Best | This Run | Delta |
|--------|-----------|---------------|----------|-------|
| {metric} | | | | |

**Observations**: {What we learned — fill after experiment completes}

**Next Steps**: {What to try based on these results — fill after experiment completes}
```

## Guidelines
- Every experiment gets an entry BEFORE it runs (hypothesis-first)
- Update the entry AFTER it completes with results, observations, and next steps
- Include W&B run IDs and commit hashes for reproducibility
- Document failures as thoroughly as successes — they're often more informative
- Link to related experiments: "Follow-up to EXP-003" or "Addresses regression in EXP-007"
