# /train — Start a New Training Experiment

Create a new training configuration and launch script for a fine-tuning experiment.

## Usage
```
/train [model] [method] [dataset] [description]
```

Examples:
- `/train qwen3-8b qlora astro-sft-v001 "first astronomy QLoRA experiment"`
- `/train qwen3-4b qlora astro-sft-v001 "smaller model comparison"`

## Workflow
1. Generate a config YAML in `configs/` following the naming convention: `{model}-{method}-{dataset}-{version}.yaml`
2. Create a W&B run name: `{model}-{method}-{short_description}-{timestamp}`
3. Add a new experiment entry to `docs/RESEARCH_LOG.md` with hypothesis and expected outcome (use the EXP-XXX template)
4. Generate the cloud GPU launch commands (RunPod spot instance)
5. Print a pre-flight checklist

## Template Config
```yaml
# configs/{model}-{method}-{dataset}-{version}.yaml
experiment:
  name: "{description}"
  hypothesis: ""  # MUST fill this in before running
  expected_outcome: ""
  date: "YYYY-MM-DD"
  wandb_run: null

model:
  base_model: "Qwen/Qwen3-8B"  # or Qwen/Qwen3-4B
  model_type: "qwen3"
  torch_dtype: "bfloat16"
  attn_implementation: "flash_attention_2"

quantization:
  load_in_4bit: true
  bnb_4bit_quant_type: "nf4"
  bnb_4bit_use_double_quant: true
  bnb_4bit_compute_dtype: "bfloat16"

lora:
  r: 64
  lora_alpha: 128
  lora_dropout: 0.05
  bias: "none"
  target_modules:
    - "q_proj"
    - "k_proj"
    - "v_proj"
    - "o_proj"
    - "gate_proj"
    - "up_proj"
    - "down_proj"
  task_type: "CAUSAL_LM"

training:
  num_train_epochs: 3
  per_device_train_batch_size: 4
  per_device_eval_batch_size: 4
  gradient_accumulation_steps: 4  # Effective batch size: 16
  learning_rate: 2.0e-4
  lr_scheduler_type: "cosine"
  warmup_ratio: 0.05
  weight_decay: 0.01
  max_grad_norm: 1.0
  max_seq_length: 4096
  bf16: true
  gradient_checkpointing: true
  group_by_length: true
  logging_steps: 10
  eval_strategy: "steps"
  eval_steps: 250
  save_strategy: "steps"
  save_steps: 500
  save_total_limit: 3
  load_best_model_at_end: true
  metric_for_best_model: "eval_loss"

data:
  train_file: "data/sft/train.jsonl"
  eval_file: "data/sft/eval.jsonl"
  chat_template: "qwen3"
  train_on_completions_only: true
  system_message: "You are AstroLLM, an astronomy and astrophysics research assistant. Provide accurate, well-sourced answers. When uncertain, acknowledge limitations. Cite relevant papers when possible."

wandb:
  project: "astrollm"
  entity: null
  tags: []
  log_model: "checkpoint"

checkpointing:
  output_dir: "models/{model}-{method}-{dataset}-{version}"
  resume_from_checkpoint: null
  push_to_hub: false
  hub_model_id: null

infrastructure:
  recommended_gpu: "NVIDIA A100 80GB"
  estimated_time_a100: "2-4 hours"
  estimated_time_4090: "6-10 hours"
  estimated_cost_spot: "$3-8"
  provider: "runpod"
```

## Launch Commands
```bash
# Local dry run (verify config loads)
uv run python packages/training/scripts/train_qlora.py --config configs/{config_name}.yaml --dry-run

# RunPod spot instance
runpodctl create pod \
  --name astrollm-{short_name} \
  --gpu "NVIDIA RTX 4090" \
  --image astrollm-train:latest \
  --spot

# On the pod
uv run python packages/training/scripts/train_qlora.py --config configs/{config_name}.yaml
```

## Pre-Flight Checklist
- [ ] Hypothesis documented in config YAML
- [ ] Dataset validated (`uv run python packages/data-pipeline/src/validate_dataset.py`)
- [ ] Base model accessible (check HuggingFace token)
- [ ] W&B API key configured in `.env`
- [ ] Checkpoint resumption enabled (`save_steps` set, `resume_from_checkpoint` ready)
- [ ] Previous best checkpoint noted for comparison
- [ ] Research log entry created with EXP-XXX number
