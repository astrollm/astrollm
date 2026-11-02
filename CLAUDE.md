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
  - Example: `qwen3.5-9b-qlora-astro-sft-v001.yaml`
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

# Pilot data + retrieval (what exists today)
uv run python packages/data-pipeline/src/ingest_ads.py --help        # ADS abstract ingestion (data/raw/)
uv run python packages/rag/src/index_corpus.py --help                # build pgvector + FTS indexes
uv run python packages/rag/src/pilot_retrieval.py --help             # hybrid RRF retrieval + eval
docker compose -f docker/docker-compose.pilot.yml up -d              # pilot Postgres/pgvector stack

# SFT gold-seed authoring (Phase A — what exists today)
uv run python packages/data-pipeline/src/sft/author.py prepare --query "..." --family lit_qa --partition eval
uv run python packages/data-pipeline/src/sft/author.py commit --worksheet data/sft/worksheets/<id>.yaml
uv run python packages/data-pipeline/src/sft/manifest.py status      # progress vs pre-registered targets

# Copilot v0 (M1 — what exists today; needs the pilot DB + a local llama-server, see docs/QUICKSTART.md)
uv run python packages/inference/src/copilot.py ask "What has JWST revealed about WASP-39b's atmosphere?"
uv run python packages/inference/src/copilot.py chat --tag citation        # REPL; /simbad <name>, /tag <tag>
uv run python packages/evaluation/src/pool_sweep.py --help                 # --d1-cut (D1 table), --verify-serving (parity gate)

# Training (cloud, Phase 1 — planned, not yet implemented)
uv run python packages/training/scripts/train_qlora.py --config configs/qwen3.5-9b-qlora-astro-sft-v001.yaml

# Evaluation (Phase 1 — planned, not yet implemented)
uv run python packages/evaluation/src/run_benchmark.py --model models/latest/ --benchmark astrolab-1

# Serving (Phase 1 — planned, not yet implemented)
uv run python packages/inference/src/serve.py --model models/latest/ --quantize q4_k_m

# Web (Phase 2)
cd packages/web && bun dev
```

---

## Phased Roadmap

| Phase | Name | Timeline | Key Deliverables |
|-------|------|----------|-----------------|
| **0** | Foundation & Learning | Pre-v1 | Env setup + dev pipeline; NanoGPT learning exercise skipped (jumped to Phase 1) |
| **1 (v1)** | Retrieval-Grounded Copilot | Months 1-3 | Qwen3.5-4B/9B QLoRA SFT + Gemma 4 E4B Track B, RAG + ADS/SIMBAD, beta at astrollm.org |
| **2 (v2)** | Serious Astronomy Model | Months 4-8 | Full LoRA 9B, DPO, expanded tools (NED/PDS/Gaia/MAST), TanStack Start web app |
| **3 (v3)** | Scientific Tool Ecosystem | Months 9-18 | Model family (Nano 3B + Core 9B + Pro 32B), continuous learning, tool-use SFT |
| **4+ (v4+)** | Multimodal Knowledge House | Year 2+ | AION-1 vision bridge, Ultra 70B, agent workflows |

See `docs/V1_PLAN_R2.md` for Phase 1 execution details (Revision 2, adopted 2026-11-02; `docs/V1_FINAL_PLAN.md` is the frozen Revision 1). See `docs/MASTER_PLAN.md` for the full long-term vision.

---

## Claude Code Configuration

### Workflow — Use the Right Tool for the Task
- **Starting a training experiment**: Use `/train` to generate config YAML, then `/research-log` to document the hypothesis before running
- **Evaluating a checkpoint**: Use `/eval` with the model path and benchmark name
- **Checking data health**: Use `/data-status` to scan all data directories
- **Searching for papers**: Use `/ads-search` during development and SFT curation
- **Writing Python code**: Rules in `.claude/rules/python.md` auto-activate — type hints, ruff, uv, httpx
- **Working on the web UI**: Rules in `.claude/rules/web.md` auto-activate — bun, TS strict, Biome
- **Writing docs**: Rules in `.claude/rules/docs.md` auto-activate — never modify V1_FINAL_PLAN, MASTER_PLAN, or a merged `V1_PLAN_R<N>`; plan changes go in the next revision + a Decision Log row
- **Building SFT data generation scripts**: Use the `claude-api` built-in skill (triggers on `import anthropic`)
- **Building the web UI**: Use the `example-skills:frontend-design` built-in skill for high-quality components
- **Testing the web app**: Use the `example-skills:webapp-testing` built-in skill for Playwright testing
- **Writing the arXiv paper or workshop submission**: Use the `example-skills:doc-coauthoring` built-in skill
- **Building MCP servers for astronomy tools**: Use the `example-skills:mcp-builder` built-in skill
- **Writing a weekly lab report**: Use `/lab-report` at the end of each week to document observations and learnings
- **After any significant implementation**: Use the `simplify` built-in skill to review code quality
- **Previewing docs site**: Run `uv sync --group docs && uv run mkdocs serve` to preview at http://localhost:8000

### Active Configuration
- **Personas** (`.claude/personas/`): astronomer, data-engineer, ml-engineer, frontend-dev, retrieval-engineer, devops, technical-writer, mentor, product-strategist
- **Skills** (`.claude/commands/`): `/train`, `/eval`, `/research-log`, `/lab-report`, `/data-status`, `/ads-search`
- **Rules** (`.claude/rules/`): path-scoped for python, web, training, docs, data-pipeline
- **Settings** (`.claude/settings.json`): permissions (safe tools allowed, destructive ops denied)
- **Memory** (`~/.claude/projects/.../memory/`): persistent project context across sessions (user-local, not in repo)

### Deferred — Set Up When Needed
- **MCP Servers** (`.mcp.json`): Decided 2026-11-02 — **adopt** NASA-IMPACT `astroquery-mcp` (pinned; SIMBAD + ADS + 12 more services) rather than build ADS/SIMBAD servers; the one server to *build* is the project's own retriever as an MCP server (M1/M3). Use `example-skills:mcp-builder` for that.
- **Hooks** (`settings.json → hooks`): When Python code exists, add `PostToolUse` hook on `Edit|Write` for `*.py` that runs `ruff check --fix`.
- **Custom Agents** (`.claude/agents/`): Phase 1+ — `benchmark-runner` agent for eval, `data-validator` agent for dataset health checks.
- **Persona: `end-user`** (`.claude/personas/`): at the first-copilot-UI milestone — add a grad student end-user persona representing someone *using* AstroLLM, not building it. Useful for UX decisions, prompt template design, and explanation depth calibration. (The plan's weeks-3-4 copilot demo was skipped in the 2026-06-01 pivot to SFT; the UI milestone now sits after Phase A — see the Decision Log re-anchor, 2026-07-21.)

---

## Current Status

**Phase**: 1 (v1) — Retrieval-Grounded Copilot, under **[Plan Revision 2](docs/V1_PLAN_R2.md)** (adopted 2026-11-02 after the [2026-10 plan review](docs/research/plan-review-2026-10.md); paused 2026-06-02 → 07-21 and 07-21 → 10-23)
**Done**: Retrieval foundation built and ablated end-to-end (EXP-001–003; retrieval thread closed, bottleneck = fusion ranking, beta default hybrid RRF stage-1 @ pool=100). RAG-SFT pilot pre-registered (EXP-004, amended A1 + A2) with its gold-seed authoring harness and a 67-test suite + Python CI. EXP-005 (three-arm training pilot: base / SFT / GRPO-with-verifier-reward) pre-registered. Base-model choice re-verified 2026-10-23 (Qwen3.5-4B/9B; no smaller Qwen3.6/3.8 dense exists). **M1 Copilot v0 built 2026-11-02** (branch `feat/m1-copilot-v0`): **D1 locked — k = 30** of the fused pool-100 ranking under a 16,384-token delivered budget (`K = 30`, `POOL = 100` in `sft/context.py`; `max_seq_length` 20,480 in all configs); **D2 verified** (`enable_thinking=false` renders an empty `<think>` block; thinking is on by default); serving path made identical to the frozen ranking (`pool_sweep.py --verify-serving`, 29/29); grounded chat CLI `packages/inference/src/copilot.py` with the D4 abstention grammar, a bibcode citation guard (one corrective regeneration, then flagged) and JSONL transcripts; local llama-server backend (Qwen3.5-9B Q4_K_M); `astroquery-mcp` pinned in `.mcp.json`; `tools.py` (SIMBAD card display-only, live ADS behind `--live`). See the [Research Log](docs/RESEARCH_LOG.md) and the [M1 lab note](docs/lab/2026-11-02-m1-copilot-v0.md).
**Next — finish the M1 exit**: review 25 logged conversations with error-taxonomy tags (5 logged so far: 3 smoke at k=20, 2 at the locked k=30; all guard `pass`). Then **M2**: author the eval partition under EXP-004-A2 (teacher-drafted, human-verified; `author.py prepare` now delivers the budgeted top-K and locks `delivered_k`) and run EXP-000 baselines through the copilot's grounded path. Milestone-based, no calendar dates until M4 completes.
**Gold seed**: 0 / 170–230 (eval partition authored first, in M2; calibration partition in M3).
**Deferred (post-beta)**: fusion-ranking ladder (RRF_K sweep → weighted fusion → cross-encoder); query widening; multi-call agentic retrieval (flagged experiment only until training contexts match)
**Known gaps**: `train_qlora.py` is a stub; the citation guard checks bibcode membership only, not claim support (the NLI verifier is M3); no LLM is called in the data/training pipeline yet (only the copilot calls one at serve time); exact-scan dense retrieval is a pilot-scale choice to revisit when the corpus widens; Docker DB integrity item still OPEN (Decision Log, 2026-07-21).
**Blocking issues**: None

> **Phase 0 note**: The Phase-0 NanoGPT learning exercise (train a small GPT on the astro corpus) was **not** completed — the project moved directly into the Phase-1 retrieval foundation. Nothing is fine-tuned yet and no SFT data is curated (that is the next milestone); NanoGPT remains an optional learning backfill, not a v1 blocker.
