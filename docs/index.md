# AstroLLM

A domain-specialized Large Language Model for Astronomy & Astrophysics.

!!! info "Current Status: Phase 1 (v1) — Retrieval-Grounded Copilot (in progress)"
    Retrieval foundation built and ablated; the retrieval thread is closed (bottleneck = fusion ranking, not candidate generation). The RAG-SFT pilot is pre-registered ([EXP-004](RESEARCH_LOG.md), amended per EXP-004-A1) and its gold-seed authoring harness is built — **Phase A hand-authoring is the active milestone** (0 / 150–250 examples). Nothing is fine-tuned yet; the Phase-0 NanoGPT exercise was skipped in favor of the retrieval foundation.

## What is AstroLLM?

AstroLLM is an open-source system that connects fine-tuned language models with the astronomical literature, databases, and tools that researchers actually use. It's not just a chatbot — it's a retrieval-grounded, tool-integrated research assistant that cites real papers and queries real databases.

**What makes it different:**

- :material-book-search: **Retrieval-augmented** answers grounded in NASA ADS papers, with real citations
- :material-tools: **Live tool integration**: SIMBAD object lookup, ADS paper search, Astropy calculations
- :material-school: **Audience-adaptive** explanations (undergraduate → graduate → researcher)
- :material-shield-check: Trained to **abstain** when evidence is thin, rather than hallucinate

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
| **0** | Foundation & Learning | Pre-v1 | Env setup + dev pipeline; NanoGPT learning exercise skipped (jumped to Phase 1) |
| **1 (v1)** | Retrieval-Grounded Copilot | Months 1-3 | Qwen3.5-4B/9B QLoRA SFT + Gemma 4 E4B Track B, RAG + ADS/SIMBAD, beta at astrollm.org |
| **2 (v2)** | Serious Astronomy Model | Months 4-8 | Full LoRA 9B, DPO, expanded tools, production web app |
| **3 (v3)** | Scientific Tool Ecosystem | Months 9-18 | Model family (Nano 3B + Core 9B + Pro 32B), continuous learning |
| **4+ (v4+)** | Multimodal Knowledge House | Year 2+ | AION-1 vision bridge, Ultra 70B, agent workflows |

## Quick Links

<div class="grid cards" markdown>

-   :material-rocket-launch:{ .lg .middle } **Getting Started**

    ---

    Set up the project and start building

    [:octicons-arrow-right-24: Quickstart](QUICKSTART.md)

-   :material-flask:{ .lg .middle } **Research**

    ---

    Experiment log, lab reports, findings

    [:octicons-arrow-right-24: Research Hub](research/index.md)

-   :material-map:{ .lg .middle } **Planning**

    ---

    V1 execution plan and long-term vision

    [:octicons-arrow-right-24: V1 Plan](V1_FINAL_PLAN.md)

-   :material-book-open-variant:{ .lg .middle } **Learning**

    ---

    Structured study paths for LLM engineering

    [:octicons-arrow-right-24: Build-Aligned Path](LEARNING_PATH_V1.md)

</div>
