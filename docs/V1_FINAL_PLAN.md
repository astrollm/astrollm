# AstroLLM v1 — Final Plan

*Consolidated from planning conversation + two rounds of peer review (Claude + ChatGPT)*
*Date: March 23, 2026*

---

## Product Definition

**AstroLLM v1** is a retrieval-grounded astronomy copilot for graduate students and early-career researchers, built around cited literature search, object resolution, and adaptive explanation depth.

It is NOT: a benchmark competitor to AstroSage, a model family, a multimodal system, or an autonomous agent.

---

## Scope Lock

### In v1

- One model tier: **Core** (Qwen3-4B first experiment, Qwen3-8B follow-up)
- One retrieval stack: **ADS papers + SIMBAD objects + Exoplanet Archive (thin)**
- Two live tools at orchestration layer: `ads_search` + `simbad_query`
- One UX feature: audience-adaptive explanation depth (undergrad / grad / researcher)
- One evaluation bundle: AstroMLab-1 subset + 4 custom eval tracks
- One primary user: graduate students and early-career researchers

### Explicitly deferred

- Nano / Pro / Ultra model tiers
- NED, PDS, MAST, Gaia full integrations
- Tool-use SFT (until schemas stabilize, post-week 8)
- Multimodal capabilities (AION-1 bridge is a Phase 4+ ambition)
- Autonomous multi-step agent workflows
- Custom MoE architecture

---

## Dual-Track Operating Model

### Track A — Product

Ship a working AstroLLM Core beta at astrollm.org by week 12.

### Track B — Learning

Build deep LLM engineering skill through structured fine-tuning experiments, documented in a research log.

**Week 8 gate**: Decide which model goes into the demo. Continue learning-track experiments regardless.

---

## Technical Architecture (Revised)

### Retrieval Pipeline (Three-Stage)

```
User query
    │
    ├── SIMBAD alias expansion (resolve object names → canonical IDs + aliases)
    │
    ▼
Stage 1: Hybrid recall
    ├── BM25 sparse search (keyword matching)
    ├── Dense search (SPECTER2 or BGE embeddings)
    └── Merge via reciprocal-rank fusion
    │
    ▼
Stage 2: Reranking
    └── Cross-encoder or ColBERTv2 on top 50-100 candidates
    │
    ▼
Stage 3: Astronomy-aware filtering
    ├── Weight title/abstract/conclusion differently
    ├── Filter by year, bibstem, arXiv category
    ├── ADS fielded/object-aware query when target is named
    └── Return top-k with bibcodes, sections, similarity scores
    │
    ▼
Prompt assembly → LLM generates cited answer
```

**Libraries**: Pyserini (hybrid retrieval), SPECTER2 (scientific embeddings), ColBERTv2 (reranking), astroquery (SIMBAD/ADS/NEA access), pgvector (storage)

### Model Strategy

| Phase | Model | Method | Purpose |
|-------|-------|--------|---------|
| Weeks 1-4 | Qwen3-8B (off-the-shelf) | Ollama, no training | RAG prototype backbone |
| Weeks 7-8 | Qwen3-4B | QLoRA SFT | First fine-tune experiment |
| Weeks 7-8 | Qwen3-8B | QLoRA SFT | Parallel comparison |
| Week 8+ | Best checkpoint | Merge + quantize (GGUF) | Deploy in demo |

### SFT Dataset (5,000-8,000 examples)

| Category | Weight | Source |
|----------|--------|--------|
| Literature-grounded Q&A | 30% | ADS papers → Claude API generation |
| Object/property retrieval | 25% | SIMBAD + Exoplanet Archive data |
| Citation-grounded summarization | 20% | ADS abstracts → summary pairs |
| Pedagogy/explanation | 15% | Textbooks, Astrobites, multi-level explanations |
| Tool-call formatting | 10% | Logged real interactions (added later) |

**Quality controls**: Every example tied to provenance (source ID, generation model, timestamp). 100+ manual spot-checks. Schema validation. Tool-call examples under 10-15% and drawn from real logged interactions.

### Evaluation Bundle

| Track | Size | What it measures |
|-------|------|-----------------|
| AstroMLab-1 (subset) | ~500 MCQs | Astronomy knowledge comparability |
| Grounding/citation accuracy | 25+ examples | Does the model cite correctly? |
| Tool routing accuracy | 25+ examples | Does it call the right tool? |
| Abstention under weak retrieval | 25+ examples | Does it say "I don't know" when evidence is thin? |
| Pedagogy quality | 25+ examples | Can it explain at the right level? |

### Astronomy-Specific Error Taxonomy

Track from day one:

- **Citation errors**: wrong citation, right citation wrong synthesis, missed obvious paper
- **Object-identity errors**: alias confusion, host star vs planet, wrong counterpart
- **Unit-system errors**: cgs vs SI, magnitudes vs fluxes, luminosity vs flux
- **Coordinate/epoch errors**: J2000 confusion, equatorial vs galactic
- **Catalog-semantic errors**: measured vs derived parameters, misuse of defaults
- **Literature-timeline errors**: citing superseded results as current consensus
- **Database-boundary errors**: using wrong source type for the question
- **Tool errors**: should have been called but wasn't, called unnecessarily

---

## Pre-Implementation Decisions (Lock Before Week 1)

| Decision | Choice |
|----------|--------|
| Primary user | Graduate students and early-career researchers |
| Citation format | ADS bibcodes, linked to ADS abstract pages |
| Grounding policy | Cite when making specific claims; abstain when retrieval returns <2 relevant papers; express uncertainty when papers disagree |
| Provenance tracking | Source doc ID, bibcode, chunk ID, retrieval score, generation model, timestamp on all synthetic data |
| Answer contract | Always cite sources; never fabricate references; say "I couldn't find relevant papers" rather than hallucinate |

---

## 12-Week Plan

### Budget: $400/month ($1,200 over 12 weeks, ongoing)

### Week 1-2: Corpus + Retrieval Foundation

**Build**:
- ADS ingestion pipeline (metadata + abstracts for 5,000 papers)
- Exoplanet Archive full table download (30 seconds, 5,800 rows)
- Section-aware chunking with metadata schema
- pgvector + BM25 hybrid retrieval
- SIMBAD alias resolution for object queries
- Pilot corpus: 200-500 papers first, then expand

**Lock decisions**: primary user, citation format, grounding policy

**Output**: Working retrieval CLI + gold set of 30 queries with expected relevant papers

**Study**: Raschka Ch. 1-2 (tokenization, embeddings)

**Spend**: $0

### Week 3-4: First Working Copilot

**Build**:
- Q&A pipeline with Qwen3-8B (Ollama, off-the-shelf)
- Prompt templates forcing cited claims
- Teaching mode with audience levels (undergrad / grad / researcher)
- SIMBAD object lookup integration
- Streamlit or simple web UI

**Keep narrow**: No multi-tool chains, no agent loop, no training yet

**Output**: Internal demo + 25-50 manually reviewed conversations + first eval report

**Milestone**: Week 4 demo — "AstroLLM exists and answers astronomy questions with citations"

**Study**: Raschka Ch. 3-4 (attention, GPT architecture)

**Spend**: $0-20

### Week 5-6: SFT Data Curation

**Build**:
- 5,000-8,000 SFT examples via Claude API from harvested data
- Mixture: 30% lit Q&A / 25% object / 20% summarization / 15% pedagogy / 10% tool-call
- Schema validation, provenance tracking
- 100+ manual spot-checks
- Train/eval split (95/5) separated by task family

**Output**: v1 SFT dataset with manifest

**Study**: Raschka Ch. 5-7 (pre-training, fine-tuning) + LoRA paper + QLoRA paper

**Spend**: $50-80 (Claude API)

### Week 7-8: Fine-Tuning Experiments

**Run (Track B — Learning)**:
- QLoRA on Qwen3-4B (conservative run + 2 variants)
- QLoRA on Qwen3-8B (conservative run + 2 variants)
- Small experiment matrix: base size × data mixture × LoRA rank × learning rate
- All tracked in W&B with documented hypotheses

**Evaluate**:
- Custom eval bundle (grounding, tool routing, abstention, pedagogy)
- AstroMLab-1 subset for comparability
- A/B comparison: base model vs fine-tuned on 50 astronomy questions

**Week 8 gate (Track A — Product)**: Which checkpoint goes into the demo?

**Output**: Research log entries, best checkpoint selected, merge + quantize to GGUF

**Study**: AstroSage papers (CPT → SFT → merge pipeline analysis)

**Spend**: $40-100 (10-20 training runs × $3-8 each)

### Week 9-10: Integration + Hardening

**Build**:
- Replace off-the-shelf model with fine-tuned AstroLLM in RAG pipeline
- Retrieval improvements: reranking, query rewriting, metadata filtering
- Clean UI with citation links, object cards, explanation depth toggle
- Logging for errors and failed retrievals
- Track error taxonomy

**Output**: Beta-ready system with tracked error rates

**Study**: Chip Huyen's AI Engineering (RAG + eval chapters)

**Spend**: $20-40

### Week 11-12: Ship the Beta

**Ship at astrollm.org**:
- Public AstroLLM Core beta
- One documented use case with polished example workflow
- Architecture page
- Known limitations page
- Feedback mechanism

**Collect**: Qualitative feedback, failed query logs, top 20 user tasks

**Do NOT add**: more tools, more model tiers, multimodality, agent workflows

**Output**: Live beta + week 12 retrospective

**Spend**: $50-100 (hosting)

### 12-Week Total: $160-340 (well within $1,200 budget)

Remaining budget available for: additional training experiments, retrieval improvements, expanded corpus, or saving for Phase 2.

---

## Success Criteria

**Week 4**: Internal demo retrieves papers, cites them, and resolves objects.

**Week 12**: Public beta where answers are measurably better than prompting a base model, with citations that trace to real papers.

**Measurable targets**:
- Fine-tuned model improves over base on custom eval bundle
- Citation accuracy >80% on grounding eval
- SIMBAD object resolution working for top 1,000 common targets
- At least 5 beta users providing feedback

---

## What Comes After Week 12

With a working beta in hand:

1. **Improve retrieval** — reranking, query rewriting, SPECTER2 embeddings
2. **Expand data** — full ADS corpus, more SFT examples from logged interactions
3. **Scale the model** — Qwen3-8B full LoRA, model merging experiments
4. **Add tools** — NED, PDS, Astropy calculations (tool-use SFT now justified)
5. **Distill down** — Nano (3B) for mobile/lightweight use
6. **Multimodal bridge** — AION-1 encoder + projection layer for FITS images
7. **Publish** — ML4Astro workshop paper on retrieval-grounded astronomy copilot

---

## Key Agreed Principles

1. **Retrieval engineering is a first-class research problem**, not plumbing for the LLM
2. **Ship visible artifacts on schedule** — week 4 demo, week 12 beta (protect these milestones)
3. **Training experiments are justified by the learning mandate**, not just product need
4. **Scope stays narrow for v1** — vision documents can be broad, execution must be focused
5. **Domain expertise is a force multiplier** — the builder is the primary user archetype

---

*This plan has been peer-reviewed by both Claude (Anthropic) and ChatGPT (OpenAI). Both reviewers converged on: Core-only scope, RAG-first sequencing, retrieval as the central technical challenge, and the dual-track operating model. The plan is ready for implementation.*
