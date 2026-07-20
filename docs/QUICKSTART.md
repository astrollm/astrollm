# AstroLLM Quickstart Guide

Get from zero to a working dev environment, the pilot retrieval stack, and — eventually — your
first fine-tuned astronomy model.

!!! note "What exists vs what's planned"
    Sections marked **exists today** run against real code in the repo. Sections marked
    **planned** describe the intended workflow; their scripts are not implemented yet (nothing is
    fine-tuned yet — see the [Research Log](RESEARCH_LOG.md) for current status).

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

> The Phase-0 NanoGPT learning exercise was skipped (see the Phase-0 note in
> [`CLAUDE.md`](https://github.com/astrollm/astrollm/blob/main/CLAUDE.md)); it remains an optional
> backfill — see [LEARNING_PATH_V1](LEARNING_PATH_V1.md) if you want to do it.

## Phase 1: Pilot corpus + retrieval (exists today)

This is the stack the retrieval experiments (EXP-001 – EXP-003) and the SFT authoring harness run
on.

```bash
# Ingest ADS abstracts into a frozen corpus snapshot (needs ADS_API_KEY in .env)
uv run python packages/data-pipeline/src/ingest_ads.py --help

# Bring up the isolated pilot stack (Postgres + pgvector)
docker compose -f docker/docker-compose.pilot.yml up -d

# Build the dense (pgvector) + lexical (SQLite FTS5) indexes from the snapshot
uv run python packages/rag/src/index_corpus.py --help

# Hybrid BM25+dense retrieval with RRF fusion, plus the eval harness
uv run python packages/rag/src/pilot_retrieval.py --help
```

## Phase 1: SFT gold-seed authoring (exists today)

The Phase-A authoring harness (see [sft-pilot.md](research/sft-pilot.md) for the pre-registration):

```bash
# 1. Machine assembles the frozen retrieval context and writes a worksheet
uv run python packages/data-pipeline/src/sft/author.py prepare \
  --query "What did JWST measure for the C/O ratio of WASP-39b?" \
  --family lit_qa --partition eval

# 2. You fill in answer/claims by hand, then validate + append to the gold seed
uv run python packages/data-pipeline/src/sft/author.py commit \
  --worksheet data/sft/worksheets/<id>.yaml

# Progress vs the pre-registered composition targets
uv run python packages/data-pipeline/src/sft/manifest.py status
```

## Phase 1 (continued): First Fine-Tune (planned — not yet implemented)

### Launch cloud training

```bash
# Option A: RunPod (recommended for spot pricing)
# 1. Create a RunPod pod with RTX 4090 or A100
# 2. SSH in and clone your repo
# 3. Run:

uv run python packages/training/scripts/train_qlora.py \
  --config configs/qwen3.5-9b-qlora-astro-sft-v001.yaml

# Option B: Use the Docker image
docker build -f docker/Dockerfile.train -t astrollm-train .
# Push to registry and launch on cloud GPU
```

### Evaluate your model

```bash
uv run python packages/evaluation/src/run_benchmark.py \
  --model models/qwen3.5-9b-qlora-sft-v001/final/ \
  --benchmark all \
  --compare Qwen/Qwen3.5-9B
```

### Merge and quantize

```bash
# Merge LoRA adapter with base model
uv run python packages/training/scripts/merge_model.py \
  --base Qwen/Qwen3.5-9B \
  --adapter models/qwen3.5-9b-qlora-sft-v001/final/ \
  --output models/astrollm-9b-v001-merged/

# Quantize to GGUF for efficient serving
uv run python packages/inference/src/quantize.py \
  --input models/astrollm-9b-v001-merged/ \
  --output models/astrollm-9b-v001-q4_k_m.gguf \
  --quant q4_k_m
```

### Test locally

```bash
# Serve with llama.cpp
uv run python packages/inference/src/serve.py \
  --model models/astrollm-9b-v001-q4_k_m.gguf \
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

| GPU | $/hr (spot) | QLoRA time (9B) | Cost per run |
|-----|-------------|-----------------|--------------|
| RTX 4090 | $0.40-0.80 | TBD — re-measure on first 9B run | TBD — re-measure on first 9B run |
| A100 40GB | $1.00-1.50 | TBD — re-measure on first 9B run | TBD — re-measure on first 9B run |
| A100 80GB | $1.50-2.50 | TBD — re-measure on first 9B run | TBD — re-measure on first 9B run |
| H100 80GB | $2.50-4.00 | TBD — re-measure on first 9B run | TBD — re-measure on first 9B run |

_$/hr spot rates are model-independent._

Budget tip: Use RTX 4090 spot instances for experimentation. Save A100/H100 for validated runs.
