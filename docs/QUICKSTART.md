# AstroLLM Quickstart Guide

Get from zero to your first fine-tuned astronomy model.

---

## Prerequisites

- Python 3.11+
- uv (Python package manager: docs.astral.sh/uv)
- Bun (for web packages)
- Docker (for local services)
- RunPod or Lambda Labs account (for GPU training)
- Weights & Biases account (free tier: wandb.ai)
- Anthropic API key (for SFT data generation)
- NASA ADS API key (free: ui.adsabs.harvard.edu/user/settings/token)

## Phase 0: Environment Setup (Day 1)

### 1. Clone and install

```bash
git clone https://github.com/YOUR_USERNAME/astrollm.git
cd astrollm

# Python environment (uv creates and manages the venv automatically)
uv sync

# Bun packages (for web)
bun install

# Environment variables
cp .env.example .env
# Edit .env with your API keys
```

### 2. Start local services

```bash
# PostgreSQL with pgvector + embedding server
docker compose -f docker/docker-compose.yml up -d db embedder

# Verify
docker compose -f docker/docker-compose.yml ps
```

### 3. Verify astronomy tools

```bash
uv run python -c "
from astroquery.simbad import Simbad
result = Simbad.query_object('M31')
print('SIMBAD working:', result['MAIN_ID'][0])
"
```

## Phase 0 (continued): Learn by Building

### Week 1-2: NanoGPT on Astronomy Text

```bash
# Download a small astronomy corpus (10K abstracts)
uv run python packages/data-pipeline/src/download_arxiv.py \
  --category astro-ph \
  --years 2023-2024 \
  --max-papers 10000 \
  --abstracts-only \
  --output data/raw/abstracts/

# Prepare for NanoGPT training
uv run python scripts/prepare_nanogpt_data.py \
  --input data/raw/abstracts/ \
  --output data/processed/nanogpt/

# Train NanoGPT (CPU/Mac is fine for this)
# Follow Karpathy's instructions with your astronomy data
```

### Week 3-4: Baseline Evaluation

```bash
# Evaluate base Qwen3-8B on astronomy benchmarks
# (Requires GPU — use RunPod for this)
uv run python packages/evaluation/src/run_benchmark.py \
  --model Qwen/Qwen3-8B \
  --benchmark astrolab-1 \
  --output docs/baselines/
```

## Phase 1: Data Pipeline + Training

### Download full corpus

```bash
# Full arXiv astro-ph download (this takes hours)
uv run python packages/data-pipeline/src/download_arxiv.py \
  --category astro-ph \
  --years 2020-2024 \
  --output data/raw/papers/
```

### Process and generate SFT data

```bash
# Extract clean text from papers
uv run python packages/data-pipeline/src/process_papers.py \
  --input data/raw/papers/ \
  --output data/processed/

# Generate Q&A pairs (uses Claude API — budget ~$20-50 for 10K pairs)
uv run python packages/data-pipeline/src/generate_sft.py \
  --input data/processed/ \
  --output data/sft/ \
  --type domain_qa \
  --num-pairs 10000

# Validate
uv run python packages/data-pipeline/src/validate_dataset.py \
  --input data/sft/train.jsonl \
  --schema data/sft/schema.json

# Split into train/eval
uv run python packages/data-pipeline/src/split_dataset.py \
  --input data/sft/ \
  --eval-ratio 0.05
```

### Populate RAG database

```bash
# Chunk and embed papers into pgvector
uv run python packages/rag/src/ingest.py \
  --input data/processed/ \
  --db-url postgres://astrollm:astrollm_dev@localhost:5432/astrollm \
  --embedder-url http://localhost:8081
```

## Phase 1 (continued): First Fine-Tune

### Launch cloud training

```bash
# Option A: RunPod (recommended for spot pricing)
# 1. Create a RunPod pod with RTX 4090 or A100
# 2. SSH in and clone your repo
# 3. Run:

uv run python packages/training/scripts/train_qlora.py \
  --config configs/qwen3-8b-qlora-astro-sft-v001.yaml

# Option B: Use the Docker image
docker build -f docker/Dockerfile.train -t astrollm-train .
# Push to registry and launch on cloud GPU
```

### Evaluate your model

```bash
uv run python packages/evaluation/src/run_benchmark.py \
  --model models/qwen3-8b-qlora-sft-v001/final/ \
  --benchmark all \
  --compare Qwen/Qwen3-8B
```

### Merge and quantize

```bash
# Merge LoRA adapter with base model
uv run python packages/training/scripts/merge_model.py \
  --base Qwen/Qwen3-8B \
  --adapter models/qwen3-8b-qlora-sft-v001/final/ \
  --output models/astrollm-8b-v001-merged/

# Quantize to GGUF for efficient serving
uv run python packages/inference/src/quantize.py \
  --input models/astrollm-8b-v001-merged/ \
  --output models/astrollm-8b-v001-q4_k_m.gguf \
  --quant q4_k_m
```

### Test locally

```bash
# Serve with llama.cpp
uv run python packages/inference/src/serve.py \
  --model models/astrollm-8b-v001-q4_k_m.gguf \
  --port 8080

# In another terminal, test:
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Explain the Chandrasekhar limit"}],
    "max_tokens": 500
  }'
```

## Phase 2: Web Interface

```bash
# Start full stack
docker compose -f docker/docker-compose.yml up -d

# Development mode
cd packages/web && bun dev
# Open http://localhost:3000
```

---

## Cloud GPU Cheat Sheet

### RunPod

```bash
# Install CLI
uv tool install runpodctl

# Launch spot instance (cheapest)
runpodctl create pod \
  --name astrollm-train \
  --gpu "NVIDIA RTX 4090" \
  --image astrollm-train:latest \
  --spot

# SSH in
runpodctl ssh <pod_id>
```

### Lambda Labs

```bash
# Launch via API
curl -X POST https://cloud.lambdalabs.com/api/v1/instance-operations/launch \
  -H "Authorization: Bearer $LAMBDA_API_KEY" \
  -d '{
    "region_name": "us-east-1",
    "instance_type_name": "gpu_1x_a100_sxm4",
    "ssh_key_names": ["my-key"]
  }'
```

### Cost Estimates

| GPU | $/hr (spot) | 8B QLoRA time | Cost per run |
|-----|-------------|---------------|--------------|
| RTX 4090 | $0.40-0.80 | 6-10 hrs | $3-8 |
| A100 40GB | $1.00-1.50 | 3-5 hrs | $3-8 |
| A100 80GB | $1.50-2.50 | 2-4 hrs | $3-10 |
| H100 80GB | $2.50-4.00 | 1-3 hrs | $3-12 |

Budget tip: Use RTX 4090 spot instances for experimentation. Save A100/H100 for validated runs.
