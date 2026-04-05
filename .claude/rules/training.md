---
globs: ["packages/training/**", "configs/**"]
---

# Training & Config Rules

- Every training run MUST have a config YAML in `configs/` — no hardcoded hyperparameters
- Config naming: `{model}-{method}-{dataset}-{version}.yaml`
- All training scripts MUST support `--resume_from_checkpoint` (spot instances get preempted)
- Always enable gradient checkpointing to minimize GPU memory
- Use `bf16` precision (not fp16) on modern GPUs
- Log to Weights & Biases — every run gets a W&B run ID
- Add a research log entry via `/research-log` BEFORE starting a training run
- Loss masking: train only on assistant completions, never on user/system tokens
- Save checkpoints to `models/{run_id}/` — these are gitignored
- After training: evaluate against base model AND previous best checkpoint
