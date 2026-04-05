# AstroLLM

A domain-specialized Large Language Model for Astronomy & Astrophysics.

**Mission**: Build an independent, research-driven astronomy LLM focused on pedagogy, researcher co-piloting, RAG-augmented knowledge, and deep integration with scientific tools astronomers actually use (Astropy, NASA ADS, SIMBAD, VizieR, JWST archives).

**Domain**: astrollm.org

---

## Project Architecture

### System Overview

```
                    ┌──────────────────────────────────┐
                    │         astrollm.org (Web)        │
                    │   TanStack Start + Shadcn/ui      │
                    └──────────┬───────────────────────┘
                               │
                    ┌──────────▼───────────────────────┐
                    │        API Layer (Elysia)         │
                    │   /chat  /search  /tools  /eval   │
                    └──┬───────┬────────┬──────┬───────┘
                       │       │        │      │
              ┌────────▼──┐ ┌─▼────┐ ┌─▼────┐ ┌▼──────────┐
              │ Inference  │ │ RAG  │ │Tools │ │ Evaluation │
              │ vLLM /     │ │pgvec │ │Bridge│ │ Benchmarks │
              │ llama.cpp  │ │embed │ │      │ │            │
              └────────────┘ └──────┘ └──────┘ └────────────┘
                    │                    │
              ┌─────▼────┐      ┌───────▼────────────────┐
              │ Fine-tuned│      │ Scientific Tool APIs    │
              │ Model     │      │ NASA ADS, SIMBAD,       │
              │ (LoRA +   │      │ VizieR, Astropy,        │
              │  Base)    │      │ MAST/JWST, NED          │
              └───────────┘      └────────────────────────┘
```

### Monorepo Structure

```
astrollm/
├── CLAUDE.md                    ← You are here
├── pyproject.toml               ← Python deps (uv)
├── .claude/                     ← Claude Code config
├── packages/
│   ├── data-pipeline/src/       ← arXiv ingestion, processing, dataset creation
│   ├── training/scripts/        ← Fine-tuning scripts, configs, W&B tracking
│   ├── evaluation/src/          ← Benchmark runners (AstroMLab-1, Astro-QA, custom)
│   ├── rag/src/                 ← Vector store, retrieval, embedding pipeline
│   ├── inference/src/           ← Model serving (vLLM, llama.cpp, quantization)
│   ├── tools-integration/src/   ← Bridges to astronomy tools (ADS, SIMBAD, Astropy)
│   └── web/                     ← Chat UI (TanStack Start), API server (Elysia)
├── data/                        ← Training data (gitignored, large files)
│   ├── raw/                     ← Downloaded papers, catalogs (never modified)
│   ├── processed/               ← Cleaned text, chunks (versioned by pipeline hash)
│   ├── sft/                     ← SFT datasets (JSONL + schema)
│   └── eval/                    ← Evaluation datasets
├── models/                      ← Checkpoints and adapters (gitignored)
├── configs/                     ← Training hyperparameters, model configs
├── scripts/                     ← Utility scripts (setup, download, convert)
├── docs/                        ← Architecture docs, research notes
│   └── archive/                 ← Superseded planning documents
└── docker/                      ← Dockerfiles for training, serving, dev
```

### Tech Stack

| Layer | Technology | Notes |
|-------|-----------|-------|
| Runtime | Bun | Package management + scripts (JS/TS) |
| Python | uv | Package management + virtual environments |
| Training | PyTorch + HF Transformers + PEFT + TRL + Unsloth | Fine-tuning |
| Data | Python (arxiv, pdfminer, astropy) | Pipeline |
| RAG | PostgreSQL + pgvector | Vector search |
| Backend | Elysia (Bun) | API server |
| Frontend | TanStack Start + Shadcn/ui + Tailwind | Web UI |
| Tracking | Weights & Biases | Experiments |
| Serving | vLLM or llama.cpp | Inference |
| Cloud GPU | RunPod / Lambda Labs | Training compute |
| CI/CD | GitHub Actions | Testing + deployment |

---

## Development Conventions

### Code Style
- TypeScript for web/API code (strict mode)
- Python for ML/data pipeline code (type hints, ruff for linting)
- All configs in YAML (not JSON) for readability
- Environment variables via `.env` files (never committed)
- Python packages managed with `uv` (pyproject.toml), not pip
- JS/TS packages managed with `bun`, not npm/yarn

### Git Workflow
- `main` — stable, deployable
- `dev` — integration branch
- Feature branches: `feat/data-pipeline-arxiv`, `feat/training-qlora-8b`
- Experiment branches: `exp/lr-sweep-001`, `exp/merge-slerp-v2`
- Every training run gets a W&B run ID linked in the commit message

### Data Management
- Raw data: never modified after download, stored in `data/raw/`
- Processed data: versioned by pipeline hash, stored in `data/processed/`
- SFT datasets: JSONL format with schema validation
- All datasets have a `manifest.json` with provenance metadata
- Large files tracked with Git LFS or stored externally (HuggingFace datasets)

### Training Conventions
- Every training run has a config YAML in `configs/`
- Config naming: `{model}-{method}-{dataset}-{version}.yaml`
  - Example: `qwen3-8b-qlora-astro-sft-v001.yaml`
- Checkpoints saved to `models/{run_id}/`
- W&B project: `astrollm`
- Always log: loss curves, learning rate, GPU memory, eval metrics

### Evaluation Protocol
- Run AstroMLab-1 benchmark after every SFT run
- Run custom pedagogy eval after every SFT run
- Compare against base model AND previous best checkpoint
- Results logged to `docs/RESEARCH_LOG.md` with run IDs

---

## Key Design Decisions

### Why Not Just Use AstroSage?
AstroSage is excellent at benchmark Q&A but:
1. No tool integration — can't query ADS, look up objects in SIMBAD, or run Astropy calculations
2. No RAG — knowledge frozen at training cutoff
3. No pedagogy optimization — not trained for teaching or Socratic dialogue
4. Requires significant compute — 70B model needs serious hardware to serve
5. We want to learn by building, not just consuming

### Researcher-First, Then Broaden
Phase ordering by audience:
1. **Researchers** (Phase 1-2): Tool integration, paper search, data analysis co-pilot
2. **Students** (Phase 2-3): Socratic teaching, concept explanations, problem solving
3. **Enthusiasts** (Phase 3+): Accessible Q&A, sky event explanations, observing guides

### Cloud-Only Strategy
No local GPU. All training on RunPod/Lambda Labs.
- Use spot instances for experimentation ($0.40-0.80/hr for RTX 4090)
- Use reserved instances only for validated training runs
- All training scripts must support checkpoint resumption (spot instances can be preempted)
- Docker images pre-built with all dependencies to minimize setup time

---

## Quick Reference Commands

```bash
# Setup
bun install                              # Install JS dependencies (packages/web)
uv sync                                  # Install Python dependencies (pyproject.toml)

# Data pipeline (Phase 1)
uv run python packages/data-pipeline/src/download_arxiv.py --category astro-ph --years 2020-2024
uv run python packages/data-pipeline/src/process_papers.py --input data/raw/ --output data/processed/
uv run python packages/data-pipeline/src/generate_sft.py --input data/processed/ --output data/sft/

# Training (cloud, Phase 1)
uv run python packages/training/scripts/train_qlora.py --config configs/qwen3-8b-qlora-astro-sft-v001.yaml

# Evaluation (Phase 1)
uv run python packages/evaluation/src/run_benchmark.py --model models/latest/ --benchmark astrolab-1

# Serving (Phase 1)
uv run python packages/inference/src/serve.py --model models/latest/ --quantize q4_k_m

# Web (Phase 2)
cd packages/web && bun dev
```

---

## Phased Roadmap

| Phase | Name | Timeline | Key Deliverables |
|-------|------|----------|-----------------|
| **0** | Foundation & Learning | Pre-v1 | NanoGPT on astro corpus, study transformers, env setup |
| **1 (v1)** | Retrieval-Grounded Copilot | Months 1-3 | Qwen3-4B/8B QLoRA SFT, RAG + ADS/SIMBAD, beta at astrollm.org |
| **2 (v2)** | Serious Astronomy Model | Months 4-8 | Full LoRA 8B, DPO, expanded tools (NED/PDS/Gaia/MAST), TanStack Start web app |
| **3 (v3)** | Scientific Tool Ecosystem | Months 9-18 | Model family (Nano 3B + Core 8B + Pro 32B), continuous learning, tool-use SFT |
| **4+ (v4+)** | Multimodal Knowledge House | Year 2+ | AION-1 vision bridge, Ultra 70B, agent workflows |

See `docs/V1_FINAL_PLAN.md` for Phase 1 execution details. See `docs/MASTER_PLAN.md` for the full long-term vision.

---

## Current Status

**Phase**: 0 — Foundation & Learning
**Next milestone**: Complete NanoGPT implementation on astronomy corpus
**Blocking issues**: None
