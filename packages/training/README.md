# Training Package

Fine-tuning scripts, configurations, and experiment management for AstroLLM.

## Pipeline

```
Base Model → [CPT] → SFT → [Model Merge] → [DPO] → Evaluation → Quantize → Deploy
              ^opt            ^recommended     ^Phase3+
```

> **Note:** Bases are multimodal (Qwen3.5-9B, Gemma 4 E4B) — confirm loader class before first run.

## Quick Start

```bash
# 1. Validate your dataset
python scripts/validate_data.py --input data/sft/train.jsonl

# 2. Dry run (print config, check GPU)
python scripts/train_qlora.py --config configs/qwen3.5-9b-qlora-astro-sft-v001.yaml --dry-run

# 3. Train
python scripts/train_qlora.py --config configs/qwen3.5-9b-qlora-astro-sft-v001.yaml

# 4. Resume from checkpoint (if spot instance preempted)
python scripts/train_qlora.py --config configs/qwen3.5-9b-qlora-astro-sft-v001.yaml \
  --resume models/qwen3.5-9b-qlora-sft-v001/checkpoint-1500

# 5. Merge LoRA adapter with base model
python scripts/merge_model.py \
  --base Qwen/Qwen3.5-9B \
  --adapter models/qwen3.5-9b-qlora-sft-v001/final/ \
  --method slerp --ratio 0.5 \
  --output models/astrollm-9b-v001-merged/
```

## Supported Methods

| Method | GPU Required | Training Time (9B) | Quality | Use Case |
|--------|-------------|-------------------|---------|----------|
| QLoRA | 24GB (RTX 4090) | TBD — re-measure on first 9B run | 80-90% of full | Experimentation |
| LoRA | 80GB (A100) | TBD — re-measure on first 9B run | 90-95% of full | Production |
| Full | 4x80GB | TBD — re-measure on first 9B run | 100% | Research |

## Experiment Workflow

1. Create config YAML with hypothesis
2. Run training with W&B tracking
3. Evaluate against benchmarks
4. Log results in RESEARCH_LOG.md
5. If improved: merge, quantize, deploy
6. If not: analyze, adjust data/hyperparameters, iterate
