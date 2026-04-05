# AstroLLM — Long-Term Master Plan

## A Living Knowledge House for Astronomy & Astrophysics

*"Not a model. A system that thinks in astronomy."*

---

## Vision

AstroLLM is an open, evolving intelligence layer for astronomy and astrophysics — connecting the world's astronomical literature, databases, observations, and instruments through a family of specialized language models that can retrieve, reason, calculate, teach, and discover.

The long-term goal is not "a fine-tuned chatbot that knows astronomy." It is a knowledge system where:

- A graduate student asks a question and gets a cited, adaptive explanation grounded in the actual literature
- A researcher queries across ADS, SIMBAD, NED, and mission archives in natural language, and gets synthesized answers with provenance
- An observer uploads a spectrum or light curve and gets an interpreted analysis
- A teacher generates lecture materials calibrated to their students' level
- The system continuously ingests new papers and updates its knowledge weekly
- The entire stack — models, data, tools, evaluation — is open and reproducible

This is a multi-year vision. What follows is the phased roadmap to get there.

---

## Phase Map

```
Phase 1 (v1)     Phase 2 (v2)       Phase 3 (v3)       Phase 4+
Months 1-3        Months 4-8         Months 9-18         Year 2+
────────────────  ──────────────────  ──────────────────  ──────────────────
RAG copilot       Serious model      Scientific tool     Knowledge house
with citations    + expanded tools   ecosystem           + multimodal

Core (4B/8B)      Core (8B refined)  Core + Nano + Pro   Full family
ADS + SIMBAD      + NED + NEA + PDS  Tool-use SFT        + Ultra
Hybrid retrieval  Advanced retrieval Continuous learning  Multimodal bridge
Custom evals      Formal benchmarks  API for community   AION-1 integration
Streamlit beta    Proper web app     Public service       Agent workflows
```

---

## Phase 1: The Retrieval-Grounded Copilot (v1)

*Months 1-3 | Budget: $400/month | 5-10 hrs/week*

**Already planned in V1_FINAL_PLAN.md. Summarized here for continuity.**

### Deliverables
- AstroLLM Core beta at astrollm.org
- Qwen3-4B/8B fine-tuned on 5,000-8,000 astronomy SFT examples
- Three-stage retrieval pipeline (hybrid recall → reranking → astronomy-aware filtering)
- ADS paper search + SIMBAD object lookup at orchestration layer
- Thin Exoplanet Archive integration
- Custom evaluation suite (grounding, tool routing, abstention, pedagogy)
- Audience-adaptive explanation depth

### Success Criteria
- Fine-tuned model measurably better than base on custom evals
- Citation accuracy >80%
- Public beta with 5+ users providing feedback
- Documented research log of training experiments

### What Phase 1 Proves
- That retrieval-grounded domain specialization adds real value
- That the data pipeline (ADS → SFT → training → evaluation) works end-to-end
- That there's genuine user interest from the astronomy community
- That the builder (you) can execute the full ML engineering loop

---

## Phase 2: The Serious Astronomy Model (v2)

*Months 4-8 | Budget: $400-600/month | 5-15 hrs/week (scaling up as momentum builds)*

### 2.1 Model Improvements

**Larger, better fine-tune:**
- Full LoRA (not QLoRA) on Qwen3-8B with expanded SFT dataset (15,000-25,000 examples)
- Model merging experiments (SLERP/TIES with instruct model) to preserve general capabilities
- DPO alignment using researcher feedback from v1 beta
- Explore continued pre-training (CPT) on astronomy corpus if SFT-only hits a ceiling

**Formal benchmark evaluation:**
- Full AstroMLab-1 (4,425 MCQs) — aim for measurable improvement over base model
- Full Astro-QA (3,082 questions) — broader coverage
- Expanded custom eval suite (100+ examples per track)

### 2.2 Expanded Data Sources

**Add the remaining databases as live tools:**
- NED (extragalactic objects, SEDs, multi-wavelength data)
- Full NASA Exoplanet Archive integration (transit predictions, light curve access)
- PDS (planetary mission data, instrument metadata)
- Gaia (stellar parameters, distances, proper motions)
- MAST (HST, JWST, Kepler, TESS observation search)

**Tool-use SFT now justified:**
- By this phase, tool schemas are stable from months of orchestration-layer use
- Gold dataset curated from logged real interactions in v1
- 500-1,000 tool-use examples per tool added to SFT mix

**Expanded text training data:**
- Astrobites archive (~4,000 accessible paper summaries)
- Astronomy Stack Exchange (~40K Q&A pairs)
- OpenStax Astronomy textbook (CC-BY)
- Broader arXiv astro-ph corpus (full papers, not just abstracts)

### 2.3 Advanced Retrieval

**Upgrade from v1 baseline:**
- SPECTER2 embeddings (scientific-document-specific)
- ColBERTv2 reranking on top-100 candidates
- Query rewriting with SIMBAD alias expansion + ADS fielded search
- Section-aware retrieval weighting (abstract > conclusion > methods)
- Recency weighting for fast-moving subfields
- Cross-database query routing (learn which source answers which question type)

**Retrieval evaluation as a first-class discipline:**
- Dedicated retrieval gold set with Recall@k, MRR, nDCG metrics
- Object-resolution accuracy tracking
- Astronomy-specific negative examples (same topic wrong object, same object wrong phenomenon)

### 2.4 Web Application

**Replace Streamlit with production stack:**
- TanStack Start + Shadcn/ui + Tailwind (your preferred stack)
- Elysia backend with streaming SSE responses
- Proper citation UI with ADS links, object cards, paper previews
- Dark mode default (astronomers work at night)
- Keyboard shortcuts for power users
- Conversation history with local storage
- LaTeX rendering for equations (KaTeX)

### Success Criteria for Phase 2
- AstroMLab-1 score: measurable improvement (target: base model + 5-8 points)
- 50+ active beta users
- 5+ database tools working reliably
- Published blog post or workshop paper on the approach

---

## Phase 3: The Scientific Tool Ecosystem (v3)

*Months 9-18 | Budget: $600-1,000/month (or institutional support)*

### 3.1 Model Family Begins

**Distill Core → Nano (3B):**
- Use Core's outputs to train a smaller student model
- Target: runs on phone/laptop with 4GB RAM
- Quantized to Q4_K_M: ~2GB
- Capabilities: quick lookups, object ID, basic Q&A, glossary
- Distribution: Ollama, HuggingFace, WebLLM for browser

**Scale to Pro (32B):**
- Same SFT data, larger base (Qwen3-32B or equivalent)
- QLoRA on A100 80GB cloud instance
- Capabilities: complex multi-step reasoning, multi-tool chains, long-context paper analysis
- Distribution: cloud API, self-hosted for research groups

### 3.2 Continuous Learning Pipeline

**The model stays current:**
- Weekly arXiv astro-ph ingestion (new papers auto-processed and indexed)
- Monthly RAG index refresh with new embeddings
- Quarterly SFT dataset expansion from new papers and logged interactions
- Semi-annual model re-training on expanded dataset

**Automated pipeline:**
```
arXiv RSS → Download → Extract → Chunk → Embed → Index (pgvector)
                                    ↓
                            Generate Q&A pairs → Add to SFT queue
                                    ↓
                            Quarterly re-train trigger
```

### 3.3 Tool-Use as a First-Class Capability

**The model learns to orchestrate tools natively:**
- Full tool-use SFT with stable schemas from months of logged interactions
- Multi-tool chains: "Find recent JWST papers about TRAPPIST-1e, check what observations exist in MAST, and summarize the atmospheric constraints"
- Astropy calculations: coordinate transforms, cosmological distances, unit conversions
- Error handling: graceful degradation when tools fail or return unexpected results

### 3.4 Community Features

**Open the platform:**
- Public API for astronomy tool developers to build on AstroLLM
- Model weights on HuggingFace for the community
- Evaluation benchmarks published and reproducible
- Contribution pipeline: community can submit SFT examples
- Integration with existing astronomy workflows (Jupyter, Astropy ecosystem)

### Success Criteria for Phase 3
- Three-tier model family shipping (Nano + Core + Pro)
- 500+ monthly active users
- Continuous learning pipeline running automatically
- Workshop or conference paper accepted
- Community contributions flowing in

---

## Phase 4: The Multimodal Knowledge House (v4+)

*Year 2+ | Budget: institutional support or grants | Possible collaboration*

### 4.1 Multimodal Bridge

**Connect AION-1 as a scientific encoder:**
- Use AION-1 (300M, MIT license) pre-trained weights as the vision/spectra encoder
- Add projection layer bridging AION-1 embeddings → AstroLLM text model embedding space
- Fine-tune the bridge with LoRA (lightweight)
- Inspired by "Talking with the Latents" teacher-student approach

**First modality: stellar spectra**
- Upload a 1D FITS spectrum → AstroLLM interprets spectral features
- Training data: SDSS/DESI spectra paired with text descriptions from papers
- Evaluation: spectral classification accuracy, feature identification

**Second modality: galaxy images**
- Upload a FITS cutout → AstroLLM describes morphology, identifies features
- Training data: Legacy Survey/HSC images paired with morphological classifications
- Evaluation: morphology classification, anomaly detection

**Third modality: light curves**
- Upload a Kepler/TESS light curve → AstroLLM interprets variability, identifies transits
- Training data: from Multimodal Universe dataset (HuggingFace subsets)
- Evaluation: period detection, transit identification, variability classification

### 4.2 AstroLLM Ultra (70B+)

**Institutional-grade model:**
- Full LoRA or CPT on 70B base (Qwen3-72B, Llama 3.1 70B, or successor)
- Requires institutional GPU access or cloud partnership
- Target: competitive with AstroSage-70B on AstroMLab-1
- Distribution: shared departmental server, cloud API

### 4.3 Agent Workflows

**Autonomous research assistance:**
- Literature review agent: "Find and synthesize all papers on FRBs from the last 2 years"
- Data analysis agent: query archives, download data, run Astropy analysis, interpret results
- Observation planning agent: predict transits, check visibility, suggest instruments
- Peer review assistant: check citations, flag inconsistencies, suggest related work

### 4.4 Custom Architecture Exploration

**Mixture of Experts for astronomy:**
- 16B total parameters, 4B active per token
- Specialized expert modules: stellar physics, cosmology, planetary science, instrumentation, mathematical reasoning, data analysis, literature/citation, general knowledge
- Router network learns which experts are relevant per query
- Long context (128K+ tokens) for processing multiple papers simultaneously

### Success Criteria for Phase 4
- Working multimodal system (at least one modality)
- Ultra model competitive on benchmarks
- Agent workflows completing real research tasks
- Academic publication in a major venue
- Adoption by astronomy departments or research groups

---

## Phase 5: The Open Astronomy Intelligence Platform

*Year 3+ | The full vision*

### What AstroLLM becomes at maturity

**For researchers:**
- Natural-language interface to the entire astronomical literature and database ecosystem
- Upload observations (spectra, images, light curves) and get AI-assisted interpretation
- Agent that can autonomously conduct literature reviews and data analysis
- Continuously updated with the latest papers and observations

**For students:**
- Adaptive tutor that explains concepts at the right level with real examples
- Homework helper grounded in actual physics (not hallucinated)
- Lab companion that helps interpret observations and calculations
- Career guide that can recommend papers, researchers, and opportunities

**For the public:**
- Accessible astronomy Q&A grounded in real science
- Sky event explanations (eclipses, meteor showers, conjunctions)
- Observing guides for amateur astronomers
- "What's in the sky tonight?" with live data

**For the field:**
- Open-source models, data, and evaluation benchmarks
- Reproducible training pipelines
- Community-governed development
- A demonstration that domain-specialized AI can be built by independent researchers

---

## Technology Evolution Map

| Component | Phase 1 | Phase 2 | Phase 3 | Phase 4+ |
|-----------|---------|---------|---------|----------|
| **Models** | Qwen3-4B/8B QLoRA | Qwen3-8B full LoRA + merge | + Nano (3B) + Pro (32B) | + Ultra (70B) + MoE |
| **Training** | SFT only | SFT + DPO | + CPT + distillation | + multimodal adapters |
| **Retrieval** | BM25+dense hybrid | + SPECTER2 + ColBERTv2 | + continuous indexing | + multimodal retrieval |
| **Tools** | ADS + SIMBAD (orchestration) | + NED + NEA + PDS + Gaia | Tool-use SFT | Agent orchestration |
| **Data** | 5K-8K SFT examples | 25K+ examples | 50K+ + continuous | + multimodal datasets |
| **Modalities** | Text only | Text + structured data | Text + basic images | + spectra + light curves |
| **Frontend** | Streamlit | TanStack Start + Elysia | + API for developers | + Jupyter integration |
| **Eval** | Custom tracks (25 each) | Full benchmarks + 100+ custom | + retrieval eval + community | + multimodal eval |
| **Users** | 5 beta | 50+ active | 500+ monthly | Institutional adoption |
| **Budget** | $400/mo personal | $600-1K/mo personal | Grants / institutional | Collaboration |

---

## Data Strategy Evolution

### Phase 1: Seeds
- 5,000 ADS papers (metadata + abstracts)
- 50,000 SIMBAD objects
- 5,800 confirmed exoplanets
- 5,000-8,000 SFT examples

### Phase 2: Growth
- 100,000+ ADS papers with citation graphs
- 500,000 SIMBAD objects
- Full NED galaxy catalog
- PDS mission metadata
- Gaia stellar parameters
- 25,000+ SFT examples
- Astrobites, Stack Exchange, textbooks

### Phase 3: Living System
- Full arXiv astro-ph corpus (300K+ papers)
- Weekly auto-ingestion of new papers
- Logged user interactions feeding SFT pipeline
- Community-contributed training examples
- 50,000+ SFT examples growing continuously

### Phase 4: Multimodal
- Multimodal Universe subsets (images, spectra, light curves)
- AION-1 encoder embeddings for scientific data
- SDSS/DESI spectra paired with text
- Legacy Survey/HSC images paired with descriptions
- Kepler/TESS light curves paired with interpretations

---

## Open Source Strategy

AstroLLM follows an open-core model:

**Always open:**
- Model weights (all tiers) on HuggingFace
- Training data and SFT datasets
- Evaluation benchmarks and results
- Training scripts and pipeline code
- Data ingestion and processing tools
- Research papers and documentation

**Potentially premium (if needed for sustainability):**
- Hosted inference at astrollm.org (API access)
- High-availability cloud deployment
- Priority access to latest model versions
- Custom fine-tuning service for institutions

The default is open. Premium features exist only if sustainability requires it.

---

## Research Contribution Roadmap

| Timeline | Venue | Topic |
|----------|-------|-------|
| Month 6 | Blog post | "Building a retrieval-grounded astronomy copilot on a budget" |
| Month 9 | ML4Astro workshop (ICML) | "AstroLLM: domain-specialized retrieval and tool use for astronomy" |
| Month 12 | AAS meeting poster | "An open-source LLM assistant for astronomy research and education" |
| Month 18 | Conference paper | "Retrieval-augmented generation for scientific literature: lessons from astronomy" |
| Year 2+ | Journal paper | Full system description with evaluation and community impact |

---

## Risk Management Across Phases

| Risk | Impact | Phase | Mitigation |
|------|--------|-------|-----------|
| Motivation loss from invisible progress | Project death | 1-2 | Visible artifacts at week 4 and 12; research log with tracked deltas |
| Catastrophic forgetting in fine-tuning | Wasted training spend | 1-2 | SFT-only first; model merging; eval after every run |
| Retrieval returning wrong papers | Bad user experience | 1-3 | Three-stage pipeline; dedicated retrieval eval; astronomy-aware filtering |
| Scope creep from exciting new ideas | Diluted execution | All | Scope lock in V1 Final Plan; defer list is explicit |
| Budget insufficient for scaling | Can't reach Phase 3-4 | 3+ | Build community first; apply for grants; seek institutional collaboration |
| AstroSage releases a tool-integrated version | Differentiation eroded | 2-3 | Move fast on v1; build community lock-in; focus on UX not just capability |
| Base model obsolescence (Qwen3 superseded) | Need to re-train | 2+ | Architecture-agnostic design; SFT data is model-independent; keep adapters modular |

---

## The North Star

AstroLLM is not competing to be the smartest astronomy model. It is competing to be the most useful astronomy system.

The difference:
- Smartest model: highest benchmark score, best on MCQs, most parameters
- Most useful system: finds the right paper, resolves the right object, explains at the right level, admits when it doesn't know, gets better every week

AstroSage will likely always have more training compute. AION-1 will likely always have more multimodal data. General models (Claude, GPT) will likely always have broader knowledge.

AstroLLM wins by being the system that's actually integrated into how astronomers work — and by being open, accessible, and continuously improving.

That's the knowledge house.
