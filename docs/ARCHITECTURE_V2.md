# AstroLLM Expanded Architecture — V2

> **Scope note (post peer review):** This document describes the FULL VISION across all phases.
> For v1 execution scope, see **V1_FINAL_PLAN.md** — which narrows to Core-only, ADS + SIMBAD + thin
> Exoplanet Archive, and defers the model family, multimodal, and distillation work to later phases.
> This document remains valid as the long-term technical reference for Phases 2-5.

Addressing: ADS data leverage, model family tiers, multimodal strategy, and distillation architecture.

---

## 1. Leveraging NASA ADS for Data Collection

NASA ADS is arguably the single most valuable data source for AstroLLM — far more structured and complete than raw arXiv scraping. It's a digital library of **15+ million records** covering all astronomy, astrophysics, planetary science, and related physics publications, operated by the Smithsonian Astrophysical Observatory.

### What ADS Gives Us That arXiv Alone Doesn't

| Capability | arXiv | NASA ADS |
|-----------|-------|----------|
| Paper text | LaTeX source (messy) | Clean abstracts + links to full text |
| Citation graph | None | Full citation + reference network |
| "Also-read" data | None | Co-readership patterns (what astronomers read together) |
| Object cross-linking | None | Links to SIMBAD and NED per paper |
| Metadata richness | Basic (title, authors, date) | Keywords, affiliations, grants, bibgroups, metrics |
| Historical coverage | 1991-present | 1741-present (complete astronomical bibliography) |
| Subject classification | Category tags | Detailed astronomical subject hierarchy |
| Usage statistics | Download counts | Read counts, citation velocity, trending papers |

### ADS API Data Collection Strategy

The ADS API (api.adsabs.harvard.edu) is free with registration, rate-limited but generous for research use.

#### Layer 1: Bulk Metadata Harvest

```python
import ads

# Configure
ads.config.token = "YOUR_ADS_TOKEN"

# Harvest ALL astro-ph papers with rich metadata
papers = ads.SearchQuery(
    q="bibstem:ApJ OR bibstem:MNRAS OR bibstem:A&A OR bibstem:AJ",
    fl=[
        "bibcode", "title", "author", "abstract", "keyword",
        "aff", "pubdate", "citation_count", "read_count",
        "reference", "citation", "property", "doctype",
        "identifier", "doi", "arxiv_class"
    ],
    sort="citation_count desc",
    rows=2000,
    max_pages=100  # Paginate through results
)
```

This gives us structured metadata for hundreds of thousands of papers including their full citation graphs.

#### Layer 2: Citation Network for Knowledge Graphs

ADS's citation data enables building a knowledge graph of astronomical research:
- **Citation chains**: Trace the intellectual lineage of ideas (e.g., how dark energy evidence evolved)
- **Co-citation clusters**: Papers frequently cited together reveal conceptual connections
- **Citation velocity**: Identify hot topics and emerging subfields
- **Also-read patterns**: What papers astronomers actually read after a given paper

This network becomes training data: "What papers should I read about [topic]?" or "What is the current consensus on [question]?" can be grounded in actual citation patterns.

#### Layer 3: Object-Paper Mapping

ADS links papers to SIMBAD and NED objects. This means we can build:
- **Object knowledge bases**: "What do we know about M87?" → every paper that studies M87
- **Cross-instrument synthesis**: Papers combining JWST + Chandra + VLA observations of the same object
- **Discovery timelines**: When was exoplanet Proxima b first detected, confirmed, characterized?

#### Layer 4: SFT Data Generation from ADS

ADS metadata enables generating high-quality training pairs at scale:

```json
// Literature review Q&A (grounded in citation data)
{
  "user": "What are the most influential papers on the CMB?",
  "assistant": "The most-cited papers on the CMB include... [generated from actual citation-ranked ADS results]"
}

// Object-centric Q&A (grounded in SIMBAD cross-references)
{
  "user": "Summarize what we know about Cygnus X-1",
  "assistant": "Cygnus X-1 is a stellar-mass black hole in a binary system... [synthesized from papers linked to this object in ADS]"
}

// Trend analysis (grounded in publication/citation velocity)
{
  "user": "What are the hottest topics in astronomy right now?",
  "assistant": "Based on recent publication trends... [generated from ADS read_count and recent publication data]"
}

// Author expertise (grounded in author bibliometrics)
{
  "user": "Who are the leading researchers on Fast Radio Bursts?",
  "assistant": "Key contributors include... [from ADS author search + h-index metrics]"
}
```

#### Layer 5: Full-Text Retrieval for RAG

ADS provides links to full-text PDFs and publisher pages. For RAG:
- Use ADS to identify the most relevant papers for a query
- Fetch full text via arXiv (for preprints) or publisher open-access links
- Chunk and embed for the vector store
- Maintain ADS bibcode as the canonical identifier for citation linking

### ADS Integration in the Live System

Beyond training data, ADS becomes a live tool AstroLLM can call at inference time:

```
User: "What were the key findings from JWST observations of TRAPPIST-1e?"

AstroLLM thinking:
1. Call ADS: search for "JWST TRAPPIST-1e" sorted by date
2. Get top 5 most recent papers with abstracts
3. Cross-reference with MAST for observation metadata
4. Synthesize findings with citations

Response: "Three recent JWST studies have examined TRAPPIST-1e's atmosphere.
[Author et al., 2025] used NIRSpec to find... (2024ApJ...123..456A)
[Author et al., 2025] used MIRI to constrain... (2024MNRAS.789..012B)
..."
```

### Data Collection Budget for ADS

| Task | API Calls | Rate Limit Impact | Time |
|------|-----------|-------------------|------|
| Full astro-ph metadata harvest | ~500K queries | ~1 week (with pacing) | Week 5 |
| Citation graph construction | ~200K queries | ~3 days | Week 6 |
| Object-paper mapping | ~100K queries | ~2 days | Week 6 |
| SFT data generation | ~50K queries | ~1 day | Week 7 |
| Ongoing live queries | ~100/day | Negligible | Continuous |

ADS rate limits: ~5,000 requests/day for registered API users. For bulk access beyond this, contact ADS directly — they are explicitly supportive of research projects and have partnered with AstroMLab.

### Python Tools for ADS

```python
# Option 1: ads package (high-level)
import ads
papers = ads.SearchQuery(q="black hole merger", fl=["title", "abstract", "citation_count"])

# Option 2: astroquery (more astronomy-integrated)
from astroquery import nasa_ads as na
results = na.ADS.query_simple("gravitational wave LIGO")

# Option 3: Direct API via httpx (most control)
import httpx
response = httpx.get(
    "https://api.adsabs.harvard.edu/v1/search/query",
    params={"q": "bibstem:ApJ year:2024", "fl": "bibcode,title,abstract,citation_count", "rows": 100},
    headers={"Authorization": f"Bearer {ADS_TOKEN}"}
)
```

---

## 2. AstroLLM Model Family — Hardware-Tiered Series

This is an excellent strategy. Rather than one model, build a family optimized for different deployment targets.

### The AstroLLM Model Lineup

```
┌──────────────────────────────────────────────────────────────────────┐
│                    AstroLLM Model Family                             │
├──────────┬──────────┬──────────────┬────────────────────────────────┤
│ AstroLLM │ AstroLLM │ AstroLLM     │ AstroLLM                      │
│ Nano     │ Core     │ Pro          │ Ultra                          │
│ (1-3B)   │ (7-8B)   │ (14-32B)     │ (70B+)                        │
├──────────┼──────────┼──────────────┼────────────────────────────────┤
│ Phone    │ Mac M2+  │ Personal     │ Institutional                  │
│ Laptop   │ Gaming PC│ cloud GPU    │ GPU cluster                    │
│ RPi 5    │ RTX 3060+│ A100/H100    │ Multi-node H100                │
├──────────┼──────────┼──────────────┼────────────────────────────────┤
│ 2-4 GB   │ 4-8 GB   │ 10-24 GB     │ 40-80+ GB                     │
│ RAM/VRAM │ VRAM     │ VRAM         │ VRAM                           │
└──────────┴──────────┴──────────────┴────────────────────────────────┘
```

### AstroLLM Nano (1-3B) — "Runs Everywhere"

**Base models**: Llama 3.2 1B/3B, Qwen 2.5 1.5B/3B, Phi-3.5-mini (3.8B), SmolLM2 1.7B

**Target hardware**:
- Smartphones (via llama.cpp or MLC LLM)
- Raspberry Pi 5 (8GB)
- Any laptop with 8GB RAM (CPU inference)
- Browser via WebLLM/WASM

**Capabilities**:
- Quick astronomical fact lookup
- Unit conversions and coordinate transforms
- Object identification and basic properties
- Simple Q&A (what is a neutron star, what is redshift)
- Glossary and terminology assistance

**Limitations**: Limited reasoning depth, no multi-step analysis, no tool calling

**How to build**: Distill from AstroLLM Core or Pro (see Section 4). The 3B Llama 3.2 is surprisingly capable with good SFT data.

**Quantization**: Q4_K_M (GGUF) brings 3B model to ~2GB — fits in phone RAM

**Use case**: Student studying with a textbook who wants quick lookups. Amateur astronomer at the telescope needing object info. Offline-capable astronomy assistant.

### AstroLLM Core (7-8B) — "The Sweet Spot"

**Base models**: Llama 3.1 8B, Qwen 2.5 7B, Mistral 7B v0.3

**Target hardware**:
- Apple Silicon Mac (M1/M2/M3/M4 with 16GB+)
- Gaming PC with RTX 3060 12GB+
- Any machine with RTX 4060 or better
- Google Colab (free tier T4 GPU)

**Capabilities**:
- Full astronomical Q&A with reasoning chains
- Paper summarization and analysis
- Socratic teaching dialogues
- Basic tool calling (ADS search, SIMBAD lookup)
- RAG-augmented knowledge retrieval
- Code generation (Astropy scripts)

**This is the primary development target** — where most effort goes. It's the AstroSage-8B competitor.

**Quantization options**:
| Format | Size | Quality | Hardware |
|--------|------|---------|----------|
| Q4_K_M | ~4.5 GB | Good | 8GB VRAM / 16GB RAM |
| Q5_K_M | ~5.5 GB | Better | 12GB VRAM / 16GB RAM |
| Q8_0 | ~8.5 GB | Best quant | 16GB VRAM / 24GB RAM |
| FP16 | ~16 GB | Full | 24GB VRAM |

**Use case**: Researcher's daily driver. Running locally on a MacBook Pro, querying papers, analyzing data, getting explanations.

### AstroLLM Pro (14-32B) — "Personal Cloud Power"

**Base models**: Qwen 2.5 14B/32B, Llama 3.3 70B (with heavy quantization), Mistral Small 24B, DeepSeek-V3 distills

**Target hardware**:
- RTX 4090 24GB (QLoRA fine-tuning + inference)
- A100 40GB (comfortable inference)
- Cloud VPS with L40S or A10G
- Apple M4 Pro/Max with 64GB+ unified memory

**Capabilities**:
- Everything in Core, plus:
- Complex multi-step reasoning about physical processes
- Multi-tool chains (ADS → SIMBAD → Astropy → synthesis)
- Long-context paper comparison (32K+ tokens)
- Detailed research proposal drafting
- Advanced pedagogical scaffolding (adapting to student level)
- Hypothesis generation from observational data

**Quantization**: Q4_K_M brings 32B model to ~18GB — fits RTX 4090

**Use case**: Research group's shared server. Personal cloud instance for a serious researcher. Graduate student's advisor-in-a-box.

### AstroLLM Ultra (70B+) — "Institutional Grade"

**Base models**: Llama 3.1 70B, Qwen 2.5 72B, DeepSeek-V3 (671B MoE)

**Target hardware**:
- 2-4x A100 80GB or H100 80GB
- University computing cluster
- Cloud: Lambda Labs, CoreWeave, or AWS p5 instances

**Capabilities**:
- Everything in Pro, plus:
- Benchmark-competitive with AstroSage-70B
- Complex literature synthesis across dozens of papers
- Nuanced scientific reasoning with uncertainty quantification
- Peer review assistance
- Multi-modal integration (Phase 4+)
- Agent workflows: autonomous literature review, data analysis pipelines

**This is the long-term ambition**, requiring institutional funding or collaboration.

**Use case**: Department-wide service. Astronomy department deploys this as a shared resource. Or the public-facing astrollm.org inference backend.

### Build Strategy: Bottom-Up with Distillation

```
Phase 2:  Build AstroLLM Core (8B) ─── primary effort
              │
Phase 3:  ├── Distill down → AstroLLM Nano (3B)
          ├── Scale up → AstroLLM Pro (32B)
              │
Phase 4:  └── Full training → AstroLLM Ultra (70B)
              │
Phase 5:  └── Distill Ultra → improved Nano/Core/Pro
```

Each tier shares the same SFT dataset (adapted for model capacity) and the same evaluation benchmarks — enabling direct comparison across the family.

### Naming and Versioning

```
astrollm-nano-3b-v1.0          # Llama 3.2 3B based, QLoRA SFT
astrollm-core-8b-v1.0          # Llama 3.1 8B based, QLoRA SFT
astrollm-core-8b-v1.0-Q4_K_M   # Quantized for Mac/consumer GPU
astrollm-pro-32b-v1.0          # Qwen 2.5 32B based
astrollm-ultra-70b-v1.0        # Llama 3.1 70B based, full LoRA
```

---

## 3. Multimodal AstroLLM — Vision, Spectra, Time Series

Astronomy is inherently multimodal. Researchers work with images (FITS), spectra, light curves, catalog tables, and text simultaneously. A text-only LLM is fundamentally limited.

### The Multimodal Astronomy Landscape

**AION-1** (Polymathic AI, 2025-2026) is the current state-of-the-art: a billion-parameter multimodal foundation model integrating 39 different astronomical data modalities — multi-band images, optical spectra, and various measurements — into a single model. Trained on 200M+ observations from the Multimodal Universe dataset on the Jean Zay supercomputer's H100 partition. Available in 300M, 800M, and 3.1B parameter sizes.

**The Multimodal Universe** (NeurIPS 2024) provides the dataset: 100TB of multi-channel images, hyper-spectral images, spectra, multivariate time series, and associated metadata from 20+ major surveys.

**"Talking with the Latents"** (Kamai et al., Feb 2026) shows a different approach: fusing pre-trained latent physical features with a pre-trained LLM via lightweight adapters and LoRA, achieving strong results on 1B-32B parameter models.

### AstroLLM's Multimodal Strategy

Rather than building a giant multimodal model from scratch (which requires AION-1-level compute), we take a pragmatic, modular approach:

#### Tier 1: Text + Structured Data (Phase 2-3)

Already in the plan — the model understands text and can call tools that return structured data (SIMBAD coordinates, VizieR photometry, catalog values). No vision component needed.

```
User: "What is the spectral type and luminosity of Betelgeuse?"
→ Tool call: simbad_query("Betelgeuse", ["sp_type", "flux_V"])
→ Model interprets: "Betelgeuse is classified as M1-M2 Ia-ab..."
```

#### Tier 2: Image Understanding via Vision-Language Bridge (Phase 4)

Add a vision encoder that can process astronomical images:

**Architecture**: LLaVA-style (vision encoder + projection layer + LLM)
```
FITS Image → Vision Encoder (SigLIP/DINOv2) → Projection MLP → AstroLLM Core (8B)
```

**Training data**: Use images from Legacy Surveys, SDSS, HSC paired with text descriptions from papers that analyze those images.

**Capabilities**:
- "What do you see in this galaxy image?" → morphological classification
- "Is there anything unusual in this field?" → anomaly detection
- "Compare these two images" → transient identification
- Upload a spectrum plot → interpret spectral features

**Base technology**: Fine-tune an existing VLM (e.g., LLaVA-1.6, InternVL) on astronomy image-text pairs, or add a vision adapter to AstroLLM Core.

#### Tier 3: Native Scientific Data Understanding (Phase 5+)

This is the ambitious tier — processing raw scientific data formats natively:

**Spectra**:
```
User: [uploads 1D FITS spectrum]
AstroLLM: "This appears to be an optical spectrum of a Type Ia supernova
near maximum light. I can see the characteristic Si II absorption at 6150Å,
the S II 'W' feature at 5400Å, and Ca II H&K absorption..."
```

**Light curves**:
```
User: [uploads TESS light curve CSV]
AstroLLM: "This light curve shows periodic dips consistent with a transiting
exoplanet. The period is approximately 3.2 days with a transit depth of ~0.8%,
suggesting a planet roughly 1.2 Earth radii..."
```

**FITS images**:
```
User: [uploads multi-band FITS]
AstroLLM: "This is a 3-color composite from what appears to be g, r, i band
imaging. The central source shows an extended disk with spiral structure.
The blue knots suggest active star-forming regions..."
```

**Implementation approach (inspired by "Talking with the Latents")**:
1. Use pre-trained encoders for each modality:
   - Images: DINOv2 or AstroMAE (pre-trained on astronomical images)
   - Spectra: 1D CNN encoder pre-trained on SDSS/DESI spectra
   - Light curves: Time series transformer (e.g., from Multimodal Universe)
2. Add lightweight adapters (LoRA) to project these into LLM's embedding space
3. Teacher-student distillation: use Claude/GPT-4V to generate text descriptions of scientific data, then train AstroLLM to produce similar descriptions from raw data

### Multimodal Data Sources

| Modality | Source | Volume | Format |
|----------|--------|--------|--------|
| Multi-band images | Legacy Surveys DR10 | 1B+ sources | FITS cutouts |
| Multi-band images | HSC SSP | ~800M objects | FITS |
| Optical spectra | SDSS DR18 | ~5M spectra | FITS |
| Optical spectra | DESI | ~40M spectra | FITS |
| Light curves | Kepler/K2 | ~500K targets | FITS |
| Light curves | TESS | ~2M targets | FITS |
| Light curves | ZTF | ~1B alerts | Avro/Parquet |
| X-ray images | Chandra | ~20K observations | FITS |
| IR images | JWST MAST | Growing daily | FITS |
| All of the above | Multimodal Universe | 100TB combined | HuggingFace |

The **Multimodal Universe** dataset (100TB, NeurIPS 2024) is specifically designed for training foundation models and is freely accessible. This is a goldmine.

### Multimodal Phasing

| Phase | Modality | Effort | Hardware |
|-------|----------|--------|----------|
| 2-3 | Text + structured data (tools) | Core effort | Cloud GPU |
| 4 | + Image understanding (VLM) | Add vision encoder | A100 |
| 5 | + Spectra + light curves | Add scientific encoders | Multi-GPU |
| 6 | Native FITS processing | Full multimodal | Cluster |

---

## 4. Distillation & Custom Architecture — Building Bigger

This is the most ambitious direction and deserves a thoughtful strategy. There are three distinct approaches, each with different resource requirements.

### Approach A: Distillation Cascade (Practical, $500-5K/month)

Use large teacher models (Claude, GPT-4, or AstroLLM Ultra) to generate high-quality training data for smaller student models.

```
┌─────────────────────────────┐
│ Teacher (Claude / GPT-4 /   │
│ AstroSage-70B)              │
│ Generate: reasoning chains, │
│ Q&A, explanations           │
└─────────────┬───────────────┘
              │ Synthetic data
              ▼
┌─────────────────────────────┐
│ AstroLLM Ultra (70B)        │
│ Fine-tune on teacher data   │
│ + astronomy corpus          │
└─────────────┬───────────────┘
              │ Distill (logits or data)
              ▼
┌─────────────────────────────┐
│ AstroLLM Pro (32B)          │
│ Distill from Ultra          │
└─────────────┬───────────────┘
              │ Distill
              ▼
┌─────────────────────────────┐
│ AstroLLM Core (8B)          │
│ Distill from Pro            │
└─────────────┬───────────────┘
              │ Distill
              ▼
┌─────────────────────────────┐
│ AstroLLM Nano (3B)          │
│ Distill from Core           │
└─────────────────────────────┘
```

**Techniques**:
- **Logit distillation**: Student learns to match teacher's soft probability distributions (requires access to teacher logits — only possible with open-weight teachers)
- **Chain-of-thought distillation**: Teacher generates reasoning chains; student learns from these (works with any teacher, including API-only models like Claude)
- **Rationale distillation**: Teacher explains WHY answers are correct; student absorbs reasoning patterns
- **Dataset distillation**: Synthesize compact, high-impact datasets that maximize learning per example

**This is DeepSeek's core strategy** — they used large models to generate reasoning traces, then trained smaller models (including DeepSeek-R1-Distill) to follow those traces. The same approach works for astronomy.

**Practical implementation**:
```python
# Generate reasoning chains from teacher
prompt = """
You are an expert astronomer. Answer this question step by step,
showing your complete reasoning process.

Question: {question}

Think through this carefully:
1. What physical principles are relevant?
2. What observational evidence exists?
3. What calculations are needed?
4. What is the answer with appropriate uncertainty?
"""

# Use Claude API to generate 50,000 reasoning chains
# These become SFT data for the student model
```

### Approach B: Progressive Pre-Training (Moderate, $5K-50K total)

Build a stronger base before fine-tuning by continued pre-training at increasing scale.

```
Phase 1: Llama 3.1 8B → CPT on astronomy text → AstroLLM-Base-8B
Phase 2: AstroLLM-Base-8B → SFT → AstroLLM-Core-8B
Phase 3: Use AstroLLM-Core-8B to generate better training data
Phase 4: Repeat with 32B → AstroLLM-Base-32B → AstroLLM-Pro-32B
Phase 5: Use Pro to generate even better data → improve Core and Nano
```

This is the AstroSage approach — and it requires significant compute (they used 43,000 GPU-hours for the 70B model). But for the 8B model, their CPT was more feasible.

**Key insight from AstroSage**: CPT works IF you have enough diverse, high-quality data. They used the complete arXiv astro-ph corpus (2007-2024), not just abstracts.

### Approach C: Custom Architecture (Very Ambitious, $50K+)

Build a custom transformer architecture optimized for astronomical reasoning.

**Inspiration from DeepSeek, Qwen, Kimi**:
- **DeepSeek-V3**: Mixture of Experts (MoE) with 671B total parameters but only 37B active per token. Extremely efficient for its capability level.
- **Qwen 2.5**: Dense models with strong multilingual support and long context.
- **Kimi (Moonshot)**: Extreme long-context (128K-2M tokens) for processing entire paper collections.

**What a custom AstroLLM architecture might look like**:

```
AstroLLM-MoE Architecture (Concept):
- 16B total parameters, 4B active per token
- 8 expert modules:
  - Stellar Physics Expert
  - Cosmology Expert
  - Planetary Science Expert
  - Instrumentation Expert
  - Mathematical Reasoning Expert
  - Data Analysis Expert
  - Literature/Citation Expert
  - General Knowledge Expert
- Router network learns to activate relevant experts per query
- Long context: 128K tokens (process multiple papers at once)
```

**Why MoE for astronomy**: Different subfields require different knowledge. A question about stellar evolution needs different "neural circuits" than one about dark energy. MoE lets the model specialize without growing inference cost.

**Reality check**: Building a custom architecture from scratch requires:
- 6-12 months of engineering
- $100K-1M in compute
- A team of 3-5 ML engineers
- This is a Phase 5+ ambition, not a starting point

### Recommended Distillation Strategy for AstroLLM

```
Phase 2 (Now):
├── Fine-tune AstroLLM Core (8B) with QLoRA
├── Use Claude/GPT-4 as teachers to generate SFT data
└── This is CoT distillation — the most accessible technique

Phase 3 (Months 4-6):
├── Distill Core → Nano (3B) using Core's outputs as training data
├── Scale to Pro (32B) using same SFT data + Core's reasoning chains
└── Add RAG to compensate for Nano's smaller knowledge capacity

Phase 4 (Months 7-12):
├── If budget allows: CPT a 70B model → AstroLLM Ultra
├── Logit distillation: Ultra → improved Pro → improved Core → improved Nano
└── Each generation improves the whole family

Phase 5+ (Year 2):
├── Explore MoE architecture for efficient scaling
├── Integrate multimodal encoders
└── Potentially contribute to or fork AstroSage's open models
```

### Key Distillation Papers to Study

1. **"Distilling Step-by-Step"** (Hsieh et al., 2023): Extract rationales from LLMs to train smaller models. Outperforms standard fine-tuning with 50% less data.
2. **DeepSeek-R1 Distillation** (2025): CoT distillation from 671B MoE to 1.5B-70B dense models. Key technique: generate long reasoning traces, train student to reproduce them.
3. **"Talking with the Latents"** (Kamai et al., 2026): Teacher-student distillation specifically for astronomy. Large LLM generates synthetic Q&A, student learns via LoRA adapters.
4. **Minitron** (NVIDIA, 2025): Combines structured weight pruning with knowledge distillation. Prune a large model, then distill to recover quality.

---

## 5. Updated Project Roadmap

### Phase 0-1: Foundation (Weeks 1-8) — unchanged
Build data pipeline, learn fundamentals, set up ADS integration.

### Phase 2: AstroLLM Core (Weeks 9-14) — primary effort
QLoRA fine-tune 8B model. Evaluate against benchmarks.

### Phase 3: Model Family + RAG (Weeks 15-24) — expanded
- Distill Core → Nano (3B)
- Scale to Pro (32B) with same data
- Build RAG pipeline with ADS-powered retrieval
- Web interface at astrollm.org

### Phase 4: Multimodal + Ultra (Weeks 25-36) — new
- Add vision encoder to Core (LLaVA-style)
- Fine-tune Ultra (70B) if funding available
- Distill Ultra back down to improve Core and Nano
- Integrate Multimodal Universe dataset

### Phase 5: Custom Architecture (Year 2+) — aspirational
- Explore MoE for efficient scaling
- Native scientific data processing (spectra, light curves)
- Potential academic collaboration and publication

---

## 6. Updated Budget Matrix

| Phase | Lean | Serious | Lab |
|-------|------|---------|-----|
| **Core (8B)** | $200 | $1,500 | $5,000 |
| **+ Nano (3B)** | +$50 | +$200 | +$500 |
| **+ Pro (32B)** | +$500 | +$3,000 | +$10,000 |
| **+ Ultra (70B)** | N/A | N/A | +$20,000-50,000 |
| **+ Multimodal** | N/A | +$2,000 | +$15,000 |
| **+ Custom MoE** | N/A | N/A | +$100,000+ |
| **ADS API** | Free | Free | Free |
| **Multimodal Universe data** | Free | Free | Free |

The beauty of the family approach: each tier has independent value. You can ship Nano while still building toward Ultra.
