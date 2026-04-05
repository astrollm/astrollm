# AstroLLM Learning Path

A structured curriculum to go from "reading about LLMs" to "shipping a fine-tuned astronomy model."

---

## Overview

```
Weeks 1-4:   FOUNDATIONS     → Understand transformers & training from scratch
Weeks 5-8:   PRACTITIONER    → Master fine-tuning tools and data engineering
Weeks 9-12:  BUILDER         → Fine-tune, evaluate, and deploy your first model
Weeks 13-16: SPECIALIST      → RAG, tool integration, production systems
Weeks 17+:   RESEARCHER      → Scale up, publish, iterate
```

---

## Weeks 1-2: Transformer Architecture Deep Dive

### Goal
Understand every component of a transformer well enough to implement one.

### Study Material
1. **Raschka — "Build a LLM from Scratch"** Chapters 1-4
   - Chapter 1: Understanding LLMs
   - Chapter 2: Working with text data (tokenization, embeddings)
   - Chapter 3: Coding attention mechanisms
   - Chapter 4: Implementing a GPT model from scratch

2. **Karpathy — "Let's build GPT from scratch"** (YouTube, ~2 hrs)
   - Code along in a Jupyter notebook
   - URL: https://www.youtube.com/watch?v=kCc8FmEb1nY

3. **Paper**: "Attention Is All You Need" (Vaswani et al., 2017)
   - Focus on: multi-head attention, positional encoding, encoder-decoder vs decoder-only
   - arxiv.org/abs/1706.03762

### Hands-On
- [ ] Implement self-attention from scratch in PyTorch (no library calls)
- [ ] Implement multi-head attention
- [ ] Build a small transformer decoder (2-4 layers, 128-256 dim)
- [ ] Train it on a tiny text corpus and verify it learns patterns

### Checkpoint
Can you explain, without notes: What is the attention mechanism computing? Why do we need multiple heads? What is the purpose of positional encoding?

---

## Weeks 3-4: NanoGPT & Training Fundamentals

### Goal
Train a small GPT on astronomy text. Understand the training loop deeply.

### Study Material
1. **Karpathy — NanoGPT** (GitHub + YouTube series)
   - Repository: github.com/karpathy/nanoGPT
   - Video series: "Neural Networks: Zero to Hero"

2. **Raschka — Chapters 5-7**
   - Chapter 5: Pretraining on unlabeled data
   - Chapter 6: Fine-tuning for classification
   - Chapter 7: Fine-tuning to follow instructions

3. **Tokenization deep dive**
   - Karpathy — "Let's build the GPT Tokenizer" (YouTube, ~2 hrs)
   - Understand BPE, SentencePiece, how domain text affects tokenization
   - Experiment: tokenize astronomy text and see how astro-specific terms are split

### Hands-On
- [ ] Clone NanoGPT, reproduce the Shakespeare training
- [ ] Download 10,000 arXiv astro-ph abstracts
- [ ] Prepare data for NanoGPT (character-level or BPE tokenizer)
- [ ] Train NanoGPT on your astronomy corpus
- [ ] Generate text — does it produce astronomy-like output?
- [ ] Experiment with hyperparameters: learning rate, context length, model size
- [ ] Plot training loss, understand overfitting signals

### Key Concepts to Master
- Loss functions (cross-entropy for language modeling)
- Learning rate schedules (warmup + cosine decay)
- Gradient accumulation (simulating larger batch sizes)
- Perplexity as an evaluation metric
- Overfitting vs underfitting in language models

### Checkpoint
Can you: Modify NanoGPT to change context length? Explain why your model generates certain outputs? Read a loss curve and diagnose training issues?

---

## Weeks 5-6: Fine-Tuning Theory — LoRA, QLoRA, and Domain Specialization

### Goal
Understand parameter-efficient fine-tuning deeply, and learn from AstroSage's successes and failures.

### Study Material
1. **Paper**: "LoRA: Low-Rank Adaptation of Large Language Models" (Hu et al., 2022)
   - Focus: Why low-rank works, how adapters are injected, merge at inference
   - arxiv.org/abs/2106.09685

2. **Paper**: "QLoRA: Efficient Finetuning of Quantized LLMs" (Dettmers et al., 2023)
   - Focus: NormalFloat4, double quantization, paged optimizers
   - arxiv.org/abs/2305.14314

3. **Paper**: "AstroMLab 3: AstroSage-Llama-3.1-8B" (de Haan et al., 2025)
   - CRITICAL: Study their CPT → SFT → merge pipeline in detail
   - Understand WHY early AstroLLaMA failed (catastrophic forgetting)
   - Note their SFT data strategy: synthetic Q&A + metadata-based Q&A + filtered general instruct data
   - arxiv.org/abs/2411.09012

4. **Paper**: "AstroMLab 4: AstroSage-Llama-3.1-70B" (de Haan et al., 2025)
   - Scaling lessons, reasoning chain integration, model merging strategy
   - arxiv.org/abs/2505.17592

5. **Paper**: "Talking with the Latents" (Kamai et al., 2026)
   - Novel approach: teacher-student distillation for scientific LLMs
   - arxiv.org/abs/2602.09670

6. **Paper**: "Direct Preference Optimization" (Rafailov et al., 2023)
   - Simpler alternative to RLHF for alignment
   - arxiv.org/abs/2305.18290

### Key Concepts
- Full fine-tuning vs LoRA vs QLoRA: when to use each
- Catastrophic forgetting: what it is and how to prevent it
- Model merging: SLERP, TIES, DARE — preserving capabilities post-SFT
- Loss masking: why you only train on assistant completions
- The AstroSage pipeline: CPT (domain knowledge) → SFT (instruction following) → Merge (capability preservation)

### Checkpoint
Can you: Explain the LoRA decomposition mathematically? Describe why 4-bit quantization doesn't destroy model quality? Articulate the specific failure mode of early AstroLLaMA and how AstroSage fixed it?

---

## Weeks 7-8: Tooling Mastery & Data Engineering

### Goal
Set up the full fine-tuning toolchain and build your data pipeline.

### Study Material
1. **Hugging Face PEFT documentation**: huggingface.co/docs/peft
2. **Hugging Face TRL documentation**: huggingface.co/docs/trl
3. **Unsloth**: github.com/unslothai/unsloth (2-4x faster LoRA training)
4. **LLaMA-Factory**: github.com/hiyouga/LLaMA-Factory (all-in-one fine-tuning)
5. **mergekit**: github.com/arcee-ai/mergekit (model merging)
6. **W&B Quickstart**: docs.wandb.ai/quickstart

### Hands-On
- [ ] Install full toolchain: transformers, PEFT, TRL, unsloth, bitsandbytes
- [ ] Run a tutorial LoRA fine-tune on a tiny model (Llama 3.2 1B) with a toy dataset
- [ ] Set up W&B, run an experiment, explore the dashboard
- [ ] Build the arXiv download pipeline (use `arxiv` Python package)
- [ ] Write LaTeX → clean text extraction (regex for sections, remove figures/tables/bibs)
- [ ] Generate 1,000 Q&A pairs using Claude API as a test batch
- [ ] Validate data format against schema

### Data Engineering Tips
- AstroSage found that extracting ONLY abstracts was insufficient — include intros and conclusions
- Randomize question phrasing styles to avoid model learning a single Q&A pattern
- Include metadata-based questions (paper titles, dates, arXiv IDs) for factual grounding
- Mix in some general instruction-following data to prevent capability regression

### Checkpoint
Can you: Launch a LoRA training run end-to-end? Generate valid SFT data in the correct JSONL format? Navigate W&B to compare two training runs?

---

## Weeks 9-12: Your First Real Fine-Tune

### Goal
Fine-tune Qwen3-8B on your astronomy dataset and evaluate rigorously.

### Week 9-10: Training
- [ ] Prepare final SFT dataset (10K+ pairs minimum)
- [ ] Configure QLoRA training (use the config in `configs/`)
- [ ] Launch on RunPod (RTX 4090 or A100)
- [ ] Monitor training in W&B (loss curves, learning rate, GPU memory)
- [ ] Run 2-3 iterations adjusting: learning rate, LoRA rank, epochs

### Week 11: Evaluation
- [ ] Run AstroMLab-1 benchmark on your model vs base Qwen3-8B
- [ ] Run Astro-QA benchmark
- [ ] Qualitative evaluation: ask 20 astronomy questions, compare responses
- [ ] Check for catastrophic forgetting: test on general knowledge too
- [ ] Document everything in RESEARCH_LOG.md

### Week 12: Merge, Quantize, Deploy
- [ ] Try model merging (SLERP with original instruct model)
- [ ] Evaluate merged model — did general capabilities improve?
- [ ] Quantize to GGUF (Q4_K_M is a good quality/size tradeoff)
- [ ] Deploy locally with llama.cpp or Ollama
- [ ] Test interactive conversation

### Common Pitfalls
- Training loss going to 0: you're overfitting, reduce epochs or increase data
- Model outputs gibberish: learning rate too high, or data quality issues
- Model is worse than base: likely catastrophic forgetting — try model merging
- Memory errors: reduce batch size, enable gradient checkpointing, use QLoRA

### Checkpoint
You have a fine-tuned astronomy model running on your machine that demonstrably outperforms the base model on astronomy questions.

---

## Weeks 13-16: RAG, Tools, and Production

### Goal
Build the full AstroLLM system with RAG and scientific tool integration.

### Week 13-14: RAG Pipeline
- [ ] Set up pgvector (Docker or managed service)
- [ ] Implement section-aware paper chunking
- [ ] Embed and index paper chunks
- [ ] Build hybrid retrieval (semantic + keyword)
- [ ] Test: does RAG improve answer quality and reduce hallucinations?
- [ ] Add citation tracking (link claims back to arXiv papers)

### Week 15: Tool Integration
- [ ] Implement NASA ADS search client
- [ ] Implement SIMBAD query client
- [ ] Implement Astropy calculation bridge
- [ ] Generate tool-use SFT data (500+ examples)
- [ ] Fine-tune with tool-use examples added to dataset

### Week 16: Web Interface
- [ ] Build chat UI with TanStack Start + Shadcn
- [ ] Streaming responses (SSE from Elysia backend)
- [ ] Inline citations with links to papers
- [ ] Tool result rendering (tables, coordinates, calculations)
- [ ] Dark mode (astronomers work at night)

### Checkpoint
astrollm.org is live with a chat interface. Users can ask astronomy questions, get RAG-augmented answers with citations, and the model can call scientific tools.

---

## Weeks 17+: Scale & Specialize

### Ongoing Research Directions
- Fine-tune a 70B model (requires more GPU budget)
- DPO alignment with astronomer feedback
- Multimodal: accept FITS images, spectra, light curves
- Continuous learning: auto-ingest new arXiv papers weekly
- Specialized sub-models: cosmology, exoplanets, stellar physics
- Write and submit a workshop paper (ML4Astro at ICML, or AAS meeting)
- Contribute back to AstroMLab benchmarks

---

## Resource Links

### Books
| Book | Why | Status |
|------|-----|--------|
| Raschka — Build a LLM from Scratch | Foundation | Reading |
| Alammar & Grootendorst — Hands-On LLMs | Practical guide | Queue |

### Video Courses
| Course | Platform | Hours |
|--------|----------|-------|
| Karpathy — Neural Networks: Zero to Hero | YouTube | ~15h |
| Karpathy — Let's build GPT | YouTube | ~2h |
| Karpathy — Let's build the Tokenizer | YouTube | ~2h |
| HuggingFace NLP Course | huggingface.co/learn | ~20h |

### Essential Papers (Read in Order)
1. Attention Is All You Need (2017)
2. LoRA (2022)
3. QLoRA (2023)
4. AstroMLab 1: Benchmarking (2024)
5. AstroMLab 3: AstroSage 8B (2025)
6. AstroMLab 4: AstroSage 70B (2025)
7. Talking with the Latents (2026)
8. DPO (2023)

### Communities
- AstroMLab: astromlab.org (their models are on HuggingFace)
- ML4Astro: ml4astro.github.io
- Hugging Face Discord (PEFT/TRL channels)
- r/LocalLLaMA (Reddit — practical fine-tuning discussions)
- EleutherAI Discord (evaluation, training at scale)
