# AstroLLM

A domain-specialized Large Language Model for Astronomy & Astrophysics.

> **Status**: Phase 0 — Foundation & Learning. Not yet usable. Follow along as we build it in the open.

## What is AstroLLM?

AstroLLM is an open-source system that connects fine-tuned language models with the astronomical literature, databases, and tools that researchers actually use. It's not just a chatbot that knows astronomy — it's a retrieval-grounded, tool-integrated research assistant that cites real papers and queries real databases.

**What makes it different from general-purpose LLMs:**

- Retrieval-augmented answers grounded in NASA ADS papers, with real citations
- Live tool integration: SIMBAD object lookup, ADS paper search, Astropy calculations
- Audience-adaptive explanations (undergraduate → graduate → researcher)
- Trained to abstain when evidence is thin, rather than hallucinate

**What makes it different from [AstroSage](https://huggingface.co/AstroMLab):**

- Tool integration (AstroSage is text-only, no database queries)
- RAG pipeline (knowledge stays current, not frozen at training cutoff)
- Pedagogy optimization (Socratic teaching, not just Q&A)
- Runs on consumer hardware (8B model, not 70B)

## Architecture

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

## Roadmap

| Phase | Name | Timeline | Key Deliverables |
|-------|------|----------|-----------------|
| **0** | Foundation & Learning | Pre-v1 | NanoGPT on astro corpus, study transformers, env setup |
| **1 (v1)** | Retrieval-Grounded Copilot | Months 1-3 | Qwen3-4B/8B QLoRA SFT, RAG + ADS/SIMBAD, beta at astrollm.org |
| **2 (v2)** | Serious Astronomy Model | Months 4-8 | Full LoRA 8B, DPO, expanded tools (NED/PDS/Gaia/MAST), production web app |
| **3 (v3)** | Scientific Tool Ecosystem | Months 9-18 | Model family (Nano 3B + Core 8B + Pro 32B), continuous learning, tool-use SFT |
| **4+ (v4+)** | Multimodal Knowledge House | Year 2+ | AION-1 vision bridge, Ultra 70B, agent workflows, spectra & light curves |

See [`docs/V1_FINAL_PLAN.md`](docs/V1_FINAL_PLAN.md) for Phase 1 execution details and [`docs/MASTER_PLAN.md`](docs/MASTER_PLAN.md) for the full long-term vision.

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Python | [uv](https://docs.astral.sh/uv/) + PyTorch + HF Transformers + PEFT + TRL |
| JS/TS | [Bun](https://bun.sh/) + TanStack Start + Tailwind |
| RAG | PostgreSQL + pgvector |
| Backend | Elysia |
| Serving | vLLM / llama.cpp |
| Tracking | Weights & Biases |
| Astronomy | Astropy, astroquery (ADS, SIMBAD, VizieR, NED) |
| Cloud GPU | RunPod / Lambda Labs |

## Getting Started

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (`curl -LsSf https://astral.sh/uv/install.sh | sh`)
- [Bun](https://bun.sh/) (`curl -fsSL https://bun.sh/install | bash`)
- Docker (for local pgvector)

### Setup

```bash
git clone https://github.com/astrollm/astrollm.git
cd astrollm

# Python environment
uv sync

# Web frontend
cd packages/web && bun install && cd ../..

# Environment variables
cp .env.example .env
# Edit .env with your API keys (ADS token, W&B, etc.)
```

### Local services

```bash
# Start PostgreSQL with pgvector + embedding server
docker compose -f docker/docker-compose.yml up -d db embedder

# Verify astronomy tools work
uv run python -c "
from astroquery.simbad import Simbad
result = Simbad.query_object('M31')
print('SIMBAD working:', result['MAIN_ID'][0])
"
```

### Web dev server

```bash
cd packages/web && bun dev
# Open http://localhost:3000
```

See [`docs/QUICKSTART.md`](docs/QUICKSTART.md) for the full setup guide including cloud GPU training.

## Project Structure

```
astrollm/
├── packages/
│   ├── data-pipeline/       arXiv ingestion, processing, SFT dataset creation
│   ├── training/            Fine-tuning scripts (QLoRA, LoRA), W&B tracking
│   ├── evaluation/          Benchmark runners (AstroMLab-1, custom evals)
│   ├── rag/                 Vector store, hybrid retrieval, embedding pipeline
│   ├── inference/           Model serving (vLLM, llama.cpp, quantization)
│   ├── tools-integration/   Bridges to ADS, SIMBAD, Astropy, VizieR, NED
│   └── web/                 Chat UI (TanStack Start) + API server (Elysia)
├── configs/                 Training hyperparameters (YAML)
├── data/                    Raw, processed, SFT, eval datasets (gitignored)
├── models/                  Checkpoints and adapters (gitignored)
├── docker/                  Dockerfiles for training, serving, dev stack
└── docs/                    Architecture, research log, planning docs
```

## Data Sources

AstroLLM draws from the open astronomical data ecosystem:

| Source | What it provides | Access |
|--------|-----------------|--------|
| [NASA ADS](https://ui.adsabs.harvard.edu/) | 15M+ papers, citation graphs, co-readership | Free API key |
| [SIMBAD](https://simbad.cds.unistra.fr/) | 20M+ astronomical objects, cross-identifications | Free, no key |
| [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/) | 5,800+ confirmed planets, transit data | Free, no key |
| [NED](https://ned.ipac.caltech.edu/) | Extragalactic objects, SEDs, distances | Free, no key |
| [VizieR](https://vizier.cds.unistra.fr/) | 23,000+ astronomical catalogs | Free, no key |

See [`docs/DATA_SOURCES.md`](docs/DATA_SOURCES.md) for detailed API patterns and collection strategies.

## Documentation

| Document | Purpose |
|----------|---------|
| [`CLAUDE.md`](CLAUDE.md) | Project conventions and quick reference |
| [`docs/V1_FINAL_PLAN.md`](docs/V1_FINAL_PLAN.md) | Phase 1 execution plan (12-week scope) |
| [`docs/MASTER_PLAN.md`](docs/MASTER_PLAN.md) | Long-term vision (Phase 1-5) |
| [`docs/ARCHITECTURE_V2.md`](docs/ARCHITECTURE_V2.md) | Technical deep dive: ADS strategy, model family, multimodal |
| [`docs/DATA_SOURCES.md`](docs/DATA_SOURCES.md) | Per-database API patterns and code examples |
| [`docs/LEARNING_PATH.md`](docs/LEARNING_PATH.md) | Structured study curriculum for LLM engineering |
| [`docs/RESEARCH_LOG.md`](docs/RESEARCH_LOG.md) | Experiment tracking and decision log |

## Acknowledgments

AstroLLM builds on the work of:

- [AstroMLab](https://github.com/AstroMLab) — astronomy LLM benchmarks and AstroSage models
- [Multimodal Universe](https://github.com/MultimodalUniverse/MultimodalUniverse) — 100TB astronomical dataset (NeurIPS 2024)
- [AION-1](https://polymathic-ai.org/) — multimodal astronomy foundation model (Polymathic AI)
- The astronomy open data ecosystem: NASA ADS, SIMBAD (CDS), NED (IPAC), Astropy

## License

Apache 2.0 — see [LICENSE](LICENSE) for details.
