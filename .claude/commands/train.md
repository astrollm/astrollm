# /train — Start a New Training Experiment

Create a new training configuration and launch script for a fine-tuning experiment.

## Usage
```
/train [model] [method] [dataset] [description]
```

## Workflow
1. Generate a config YAML in `configs/` with the specified parameters
2. Create a W&B run name: `{model}-{method}-{short_description}-{timestamp}`
3. Generate the training launch script with proper cloud GPU setup
4. Add entry to `docs/RESEARCH_LOG.md` with hypothesis and expected outcome
5. Provide the RunPod/Lambda launch command

## Template Config
```yaml
# configs/{model}-{method}-{dataset}-{version}.yaml
experiment:
  name: "{description}"
  hypothesis: ""  # Fill this in before running
  expected_outcome: ""

model:
  base: "meta-llama/Llama-3.1-8B-Instruct"
  method: "qlora"  # qlora | lora | full

qlora:
  bits: 4
  quant_type: "nf4"
  double_quant: true

lora:
  r: 64
  alpha: 128
  dropout: 0.05
  target_modules: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

training:
  epochs: 3
  batch_size: 4
  gradient_accumulation_steps: 4
  learning_rate: 2.0e-4
  lr_scheduler: "cosine"
  warmup_ratio: 0.05
  max_seq_length: 4096
  bf16: true
  gradient_checkpointing: true

data:
  train: "data/sft/train.jsonl"
  eval: "data/sft/eval.jsonl"
  template: "llama3"  # Chat template

wandb:
  project: "astrollm"
  tags: []

checkpointing:
  save_steps: 500
  resume_from: null  # Set for spot instance resumption
```

## Checklist Before Launch
- [ ] Hypothesis documented in config
- [ ] Dataset validated (schema check passed)
- [ ] Base model accessible on cloud instance
- [ ] W&B API key configured
- [ ] Checkpoint resumption enabled (spot instance safety)
- [ ] Evaluation benchmark ready
