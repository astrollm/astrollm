# AstroLLM Learning Path — V1 Aligned

Study what you're building, the week you're building it. This path follows the V1_FINAL_PLAN sequencing: RAG first, fine-tune second, ship at week 12.

> For the standalone pedagogical deep-dive (transformers from scratch → fine-tuning → RAG), see `LEARNING_PATH.md`.

---

## Overview

```
Weeks 1-2:   RETRIEVAL        → Build corpus + RAG while studying embeddings & attention
Weeks 3-4:   FIRST COPILOT    → Ship RAG prototype while studying GPT architecture
Weeks 5-6:   DATA CURATION    → Build SFT dataset while studying fine-tuning theory
Weeks 7-8:   FINE-TUNING      → Run training experiments while studying AstroSage
Weeks 9-10:  INTEGRATION      → Combine model + RAG while studying evaluation
Weeks 11-12: SHIP             → Deploy beta while deepening architecture understanding
```

---

## Weeks 1-2: Retrieval Foundation

### You're Building
- ADS ingestion pipeline (5,000 papers metadata + abstracts)
- Exoplanet Archive download (5,800 planets, 30 seconds)
- pgvector + BM25 hybrid retrieval
- SIMBAD alias resolution for object queries
- Gold set of 30 queries with expected relevant papers

### Study Alongside
1. **Raschka — Chapters 1-2** (tokenization, embeddings)
   - You're embedding papers this week — understand what embeddings actually compute
   - Experiment: tokenize astronomy text, see how "Chandrasekhar" or "NGC 4151" gets split

2. **Sentence Transformers documentation** (sbert.net)
   - You're using BGE or SPECTER2 to embed papers — understand the bi-encoder architecture
   - How is a sentence embedding different from a word embedding?

3. **pgvector documentation** (github.com/pgvector/pgvector)
   - You're storing and querying vectors — understand HNSW vs IVFFlat, cosine vs L2 distance

### Key Concepts
- Embeddings: dense vector representations of text
- Cosine similarity: how retrieval finds "similar" documents
- BM25: sparse keyword matching (why hybrid beats either alone)
- Reciprocal-rank fusion: merging results from different retrieval methods

### Checkpoint
Can you: Explain why semantic search finds related papers that keyword search misses? Describe what HNSW is doing at a high level? Write a SQL query against pgvector?

---

## Weeks 3-4: First Working Copilot

### You're Building
- Q&A pipeline with off-the-shelf Qwen3-8B (Ollama, no training)
- Prompt templates forcing cited claims
- Teaching mode with audience levels (undergrad / grad / researcher)
- SIMBAD object lookup integration
- Simple web UI (Streamlit)

### Study Alongside
1. **Raschka — Chapters 3-4** (attention mechanisms, GPT architecture)
   - You're prompting a GPT model — understand what's happening inside when it generates
   - Focus on: self-attention, causal masking, why autoregressive generation works

2. **Karpathy — "Let's build GPT from scratch"** (YouTube, ~2 hrs)
   - Code along — this is the deepest understanding of what the model you're using actually does

3. **RAG survey paper**: "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (Lewis et al., 2020)
   - You just built a RAG system — now understand the theory behind it

4. **Paper**: "Attention Is All You Need" (Vaswani et al., 2017) — arxiv.org/abs/1706.03762
   - The foundational transformer paper — read alongside Raschka Ch. 3 for the implementation perspective

### Key Concepts
- Attention mechanism: what it computes and why it matters
- Autoregressive generation: how the model produces one token at a time
- Prompt engineering: how context injection shapes model behavior
- The retrieval-generation tradeoff: when to retrieve vs when to trust the model

### Checkpoint
Can you: Explain what attention is computing? Describe why your RAG system gives better answers than the bare model? Identify when the model hallucinates vs when retrieval fails?

**Milestone**: Week 4 demo — "AstroLLM exists and answers astronomy questions with citations"

---

## Weeks 5-6: SFT Data Curation

### You're Building
- 5,000-8,000 SFT examples via Claude API
- Mixture: 30% lit Q&A / 25% object / 20% summarization / 15% pedagogy / 10% tool-call
- Schema validation, provenance tracking
- 100+ manual spot-checks
- Train/eval split (95/5)

### Study Alongside
1. **Raschka — Chapters 5-7** (pre-training, fine-tuning for classification, instruction fine-tuning)
   - You're about to fine-tune — understand the full pipeline: pre-training → SFT → alignment
   - Focus on Chapter 7: how instruction fine-tuning changes a model's behavior

2. **LoRA paper** (Hu et al., 2022) — arxiv.org/abs/2106.09685
   - Next week you'll use LoRA — understand the low-rank decomposition now
   - Key insight: why does a rank-64 adapter capture most of the task-specific behavior?

3. **QLoRA paper** (Dettmers et al., 2023) — arxiv.org/abs/2305.14314
   - Understand NormalFloat4 quantization, double quantization, paged optimizers
   - Why does 4-bit quantization barely affect model quality?

### Key Concepts
- SFT data quality > quantity (AstroSage's key finding)
- Loss masking: train only on assistant completions, mask user/system tokens
- Chat templates: how the same text gets formatted differently for different models
- Data contamination: why your eval set must be separated by task family, not random

### Checkpoint
Can you: Explain the LoRA decomposition? Describe why 4-bit quantization works? Look at an SFT example and identify potential quality issues?

---

## Weeks 7-8: Fine-Tuning Experiments

### You're Building
- QLoRA on Qwen3-4B (conservative run + 2 variants)
- QLoRA on Qwen3-8B (conservative run + 2 variants)
- Experiment matrix: base size x data mixture x LoRA rank x learning rate
- All tracked in W&B
- Week 8 gate: which checkpoint goes into the demo?

### Study Alongside
1. **AstroSage papers** (de Haan et al., 2025)
   - "AstroMLab 3: AstroSage-Llama-3.1-8B" — arxiv.org/abs/2411.09012
   - "AstroMLab 4: AstroSage-Llama-3.1-70B" — arxiv.org/abs/2505.17592
   - CRITICAL: Study their CPT → SFT → merge pipeline
   - Understand why early AstroLLaMA failed (catastrophic forgetting from CPT on abstracts only)
   - Note their data strategy: synthetic Q&A + metadata-based Q&A + filtered general instruct data

2. **"Talking with the Latents"** (Kamai et al., 2026) — arxiv.org/abs/2602.09670
   - Teacher-student distillation for scientific LLMs
   - LoRA adapters + frozen base achieves strong domain performance

3. **Model merging**: mergekit documentation (github.com/arcee-ai/mergekit)
   - SLERP, TIES, DARE — techniques you'll use to recover general capabilities after SFT

### Key Concepts
- Hyperparameter sensitivity: learning rate and LoRA rank matter most
- W&B experiment tracking: comparing runs, reading loss curves, diagnosing issues
- Catastrophic forgetting: the specific failure mode and how model merging addresses it
- The eval gap: why a model can have low loss but still give bad answers

### Checkpoint
Can you: Read a W&B dashboard and diagnose a training run? Explain why your fine-tuned model is better (or worse) than the base? Articulate the tradeoff between LoRA rank and training efficiency?

---

## Weeks 9-10: Integration + Hardening

### You're Building
- Replace off-the-shelf model with fine-tuned AstroLLM in RAG pipeline
- Retrieval improvements: reranking, query rewriting, metadata filtering
- Clean UI with citation links, object cards, explanation depth toggle
- Error tracking against the astronomy error taxonomy

### Study Alongside
1. **Chip Huyen — "AI Engineering"** (RAG + evaluation chapters)
   - You're integrating RAG + fine-tuned model — study the production patterns
   - Focus on: evaluation-driven development, failure mode analysis

2. **ColBERTv2 paper** (Santhanam et al., 2022) — arxiv.org/abs/2112.01488
   - You're adding reranking — understand late interaction and how it differs from cross-encoders

3. **SPECTER2 paper** (Singh et al., 2023) — arxiv.org/abs/2211.13308
   - Scientific document embeddings — understand how it differs from general-purpose embedders
   - Consider: should you switch from BGE to SPECTER2 for your paper embeddings?

### Key Concepts
- Reranking: why two-stage retrieval (recall → precision) beats single-stage
- Evaluation as a first-class discipline: measure retrieval independently from generation
- Error taxonomy: tracking specific failure categories (citation errors, object-identity errors, etc.)
- The integration gap: fine-tuned model may be great alone but behave differently with RAG context

### Checkpoint
Can you: Measure whether your fine-tuned model + RAG is better than base model + RAG? Identify the top 3 failure modes? Explain which component (retrieval vs model) is responsible for each failure?

---

## Weeks 11-12: Ship the Beta

### You're Building
- Public AstroLLM Core beta at astrollm.org
- One polished use case with example workflow
- Architecture page, known limitations page
- Feedback mechanism
- Week 12 retrospective

### Study Alongside
1. **Karpathy — NanoGPT** (github.com/karpathy/nanoGPT)
   - Now that you've shipped, go deeper — implement a transformer from scratch on your astronomy corpus
   - This is Phase 0's original milestone, now done with 12 weeks of context

2. **AstroMLab benchmark papers** (2024)
   - "AstroMLab 1: Who Wins Astronomy Jeopardy?" — arxiv.org/abs/2407.11194
   - Understand the benchmark you've been evaluating against — what does it measure? What doesn't it measure?

3. **DPO paper** (Rafailov et al., 2023) — arxiv.org/abs/2305.18290
   - Looking ahead to Phase 2 — understand how preference optimization differs from SFT
   - You'll use feedback from beta users to create DPO training data

### Key Concepts
- Deployment: quantization tradeoffs (Q4_K_M vs Q5_K_M vs Q8_0)
- User feedback as training signal: how beta feedback feeds Phase 2
- Benchmark limitations: what AstroMLab-1 measures vs what your users actually need
- Retrospective: what worked, what didn't, what to prioritize for Phase 2

### Checkpoint
You have a live beta at astrollm.org. You can articulate: what the system does well, where it fails, and what you'd improve with more time/compute.

---

## After Week 12: Continuing Education

With a working beta in hand, study driven by what you're building next:

| If building... | Study this |
|----------------|-----------|
| Full LoRA (Phase 2) | Scaling laws, full fine-tuning vs LoRA tradeoffs |
| DPO alignment | RLHF → DPO → KTO evolution, reward modeling |
| Model merging | SLERP/TIES/DARE theory, mergekit advanced recipes |
| Distillation (Nano 3B) | "Distilling Step-by-Step", DeepSeek-R1 distillation |
| Multimodal (Phase 4) | LLaVA architecture, AION-1, vision encoders |
| Custom MoE | Mixture of Experts theory, DeepSeek-V3 architecture |

---

## Resource Links

### Books
| Book | Why | When |
|------|-----|------|
| Raschka — Build a LLM from Scratch | Foundation (Ch. 1-7 mapped to weeks 1-6) | Weeks 1-6 |
| Chip Huyen — AI Engineering | Production RAG + eval patterns | Weeks 9-10 |
| Alammar & Grootendorst — Hands-On LLMs | Practical guide, broader coverage | After week 12 |

### Video Courses
| Course | Platform | When |
|--------|----------|------|
| Karpathy — Let's build GPT | YouTube (~2h) | Week 3-4 |
| Karpathy — Let's build the Tokenizer | YouTube (~2h) | Week 1 |
| Karpathy — Neural Networks: Zero to Hero | YouTube (~15h) | Ongoing |
| HuggingFace NLP Course | huggingface.co/learn | Weeks 5-8 |

### Essential Papers (in reading order for this path)
1. pgvector HNSW paper (Week 1-2)
2. RAG paper — Lewis et al., 2020 (Week 3-4)
3. Attention Is All You Need — Vaswani et al., 2017 (Week 3-4)
4. LoRA — Hu et al., 2022 (Week 5-6)
5. QLoRA — Dettmers et al., 2023 (Week 5-6)
6. AstroMLab 3: AstroSage 8B — de Haan et al., 2025 (Week 7-8)
7. AstroMLab 4: AstroSage 70B — de Haan et al., 2025 (Week 7-8)
8. Talking with the Latents — Kamai et al., 2026 (Week 7-8)
9. ColBERTv2 — Santhanam et al., 2022 (Week 9-10)
10. SPECTER2 — Singh et al., 2023 (Week 9-10)
11. AstroMLab 1: Benchmarking — 2024 (Week 11-12)
12. DPO — Rafailov et al., 2023 (Week 11-12)

### Communities
- AstroMLab: astromlab.org (models on HuggingFace)
- ML4Astro: ml4astro.github.io
- Hugging Face Discord (PEFT/TRL channels)
- r/LocalLLaMA (Reddit — practical fine-tuning discussions)
- EleutherAI Discord (evaluation, training at scale)
