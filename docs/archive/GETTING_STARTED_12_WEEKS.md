# AstroLLM — Getting Started: 12-Week Plan

**Constraints**: 5-10 hrs/week, $150-400/month, cloud GPU only, learning while building.
**Goal**: Fine-tuned astronomy model + RAG with paper citations = working demo at astrollm.org.

---

## Strategic Principle: RAG First, Fine-Tune Second

This is the critical insight for your constraints:

```
Week 1-4:  RAG prototype (zero GPU cost, immediate visible results)
Week 5-8:  Data pipeline + first fine-tune (first GPU spend)
Week 9-12: Combine fine-tuned model + RAG → demo
```

Why this order:
1. RAG gives you a demoable astronomy assistant in ~2 weeks using an OFF-THE-SHELF model
2. You learn the data pipeline (arXiv/ADS ingestion) while building RAG — same data feeds both
3. Fine-tuning becomes BETTER because you understand your data deeply before training
4. If life gets busy, you still have a working demo at week 4

---

## Model Choice: Start with Qwen3-8B, Fine-tune Qwen3-4B

**For RAG prototype (weeks 1-4)**: Use Qwen3-8B via Ollama or a cheap cloud endpoint.
No fine-tuning yet — just prompt engineering + retrieval. This is your scaffolding.

**For first fine-tune (weeks 5-8)**: Fine-tune Qwen3-4B with QLoRA.
- 4B is the current fine-tuning champion — outperforms models 30x its size after tuning
- QLoRA on 4B fits on RTX 4090 (24GB) with room to spare
- Training cost: ~$3-8 per run on RunPod spot instances
- You can do 10-15 experimental runs for under $100
- Quantized to Q4_K_M: ~2.5GB — runs on any laptop for inference

**Later upgrade path**: Once your pipeline is solid, scale to Qwen3-8B fine-tune (still cheap)
or Llama 3.1 8B (better for model merging with instruct capabilities).

### Why Qwen3 over Llama for the first attempt?

| Factor | Qwen3-4B | Llama 3.2 3B | Llama 3.1 8B |
|--------|----------|--------------|--------------|
| Post-fine-tune quality | Best in class at 4B | Good but older | Excellent |
| Training cost (QLoRA) | ~$3-5/run | ~$2-4/run | ~$6-10/run |
| Reasoning (thinking mode) | Built-in dual mode | No | No |
| Apache 2.0 license | Yes | Llama license | Llama license |
| Fine-tuning ecosystem | Strong (Unsloth, LLaMA-Factory) | Strong | Strongest |
| Quantized inference size | ~2.5 GB | ~2 GB | ~4.5 GB |

Qwen3-4B gives the best bang-for-buck at this stage. Llama 3.1 8B is the upgrade when
you're ready to build AstroLLM Core seriously.

---

## Week-by-Week Plan

### Week 1: Environment + arXiv Data (5-8 hrs)

**Goal**: Set up everything, download your first astronomy data.

**Day 1-2 (3 hrs): Project setup**
```bash
# Clone your scaffold
git init astrollm && cd astrollm
# Copy in the project structure from the scaffold tarball

# Python environment (uv creates and manages the venv automatically)
uv sync
uv add astroquery ads arxiv httpx jsonlines tqdm rich

# Get your API keys
# - NASA ADS: https://ui.adsabs.harvard.edu/user/settings/token (free)
# - Anthropic: for SFT data generation later
# - W&B: https://wandb.ai (free tier)
# - HuggingFace: https://huggingface.co/settings/tokens (for gated models)
```

**Day 3-4 (3-4 hrs): Download astronomy data**
```python
# Quick win: download NASA Exoplanet Archive (tiny, instant)
from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive
planets = NasaExoplanetArchive.query_criteria(
    table="pscomppars", select="*", where="default_flag=1"
)
planets.write("data/raw/exoplanets.csv", format="csv", overwrite=True)
# Result: ~5,800 confirmed exoplanets with full parameters. Takes 30 seconds.

# Download 5,000 recent arXiv astro-ph abstracts via ADS
import ads
ads.config.token = "YOUR_TOKEN"
papers = list(ads.SearchQuery(
    q="bibstem:(ApJ OR MNRAS OR A+A) year:2023-2024",
    fl=["bibcode", "title", "author", "abstract", "keyword",
        "citation_count", "pubdate"],
    sort="citation_count desc",
    rows=2000, max_pages=3
))
# Save to JSONL
```

**Day 5 (1-2 hrs): Explore data**
- Read through 20-30 abstracts to understand the text patterns
- Note: how do astronomers write? What jargon do they use?
- Look at the exoplanet data — what questions would be interesting to answer?

**Spend this week**: $0 (no GPU needed)

---

### Week 2: RAG Prototype — Part 1 (5-8 hrs)

**Goal**: Embed and index your astronomy papers for semantic search.

**Setup pgvector locally with Docker**:
```bash
docker run -d --name astrollm-db \
  -e POSTGRES_DB=astrollm \
  -e POSTGRES_PASSWORD=astrollm \
  -p 5432:5432 \
  pgvector/pgvector:pg16
```

**Chunk and embed papers**:
```python
# Chunk abstracts (they're already ~150-300 words, perfect chunk size)
# For full papers, split at section boundaries

# Embed using a free local model or API
from sentence_transformers import SentenceTransformer
embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")  # 130MB, runs on CPU

# Embed all abstracts
for paper in papers:
    embedding = embedder.encode(paper["abstract"])
    # Insert into pgvector
    cursor.execute(
        "INSERT INTO chunks (arxiv_id, title, content, embedding) VALUES (%s, %s, %s, %s)",
        (paper["bibcode"], paper["title"][0], paper["abstract"], embedding.tolist())
    )
```

**Build basic semantic search**:
```python
def search_papers(query, top_k=5):
    query_embedding = embedder.encode(query)
    cursor.execute("""
        SELECT title, content, arxiv_id,
               1 - (embedding <=> %s::vector) AS similarity
        FROM chunks
        ORDER BY embedding <=> %s::vector
        LIMIT %s
    """, (query_embedding.tolist(), query_embedding.tolist(), top_k))
    return cursor.fetchall()

# Test it!
results = search_papers("habitable zone exoplanets")
for r in results:
    print(f"{r[2]}: {r[0][:80]}... (sim: {r[3]:.3f})")
```

**Spend this week**: $0 (CPU embedding, local Docker)

---

### Week 3: RAG Prototype — Part 2 (5-8 hrs)

**Goal**: Connect search to an LLM for a working astronomy Q&A system.

**Option A — Free/Cheap: Use Ollama locally or on a small VPS**
```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen3:8b  # Downloads ~5GB quantized model

# Or if your machine can't handle 8B:
ollama pull qwen3:4b  # ~2.6GB, runs on most laptops
```

**Option B — API-based: Use a cheap inference provider**
- Together.ai: Qwen3-8B at ~$0.20/M tokens
- Groq: Free tier with rate limits
- OpenRouter: Various models, pay-per-token

**Build RAG pipeline**:
```python
import httpx

def ask_astrollm(question):
    # 1. Retrieve relevant papers
    papers = search_papers(question, top_k=5)

    # 2. Build context
    context = "\n\n".join([
        f"[{p[2]}] {p[0]}\n{p[1][:500]}"
        for p in papers
    ])

    # 3. Prompt the LLM
    prompt = f"""You are AstroLLM, an astronomy research assistant.
Answer the question using the retrieved paper context below.
Always cite papers by their bibcode when referencing specific findings.
If the context doesn't contain relevant information, say so.

Retrieved papers:
{context}

Question: {question}

Answer:"""

    # 4. Call Ollama (or API)
    response = httpx.post("http://localhost:11434/api/generate", json={
        "model": "qwen3:8b",
        "prompt": prompt,
        "stream": False
    })
    return response.json()["response"]

# Test it!
print(ask_astrollm("What recent discoveries have been made about TRAPPIST-1?"))
```

**You now have a working astronomy Q&A system with citations!**
It's rough, but it works. This is your first demo.

**Spend this week**: $0-20 (if using API instead of local Ollama)

---

### Week 4: Polish the RAG + Add More Data (5-8 hrs)

**Goal**: Make the demo presentable, expand the knowledge base.

**Expand data sources**:
```python
# Add SIMBAD object data for 10,000 common objects
from astroquery.simbad import Simbad
custom = Simbad()
custom.add_votable_fields('otype', 'sp', 'flux(V)', 'rv_value', 'plx')
# Query top objects...

# Add Exoplanet Archive data to RAG
# Convert planet properties into natural language chunks:
# "TRAPPIST-1 e is a terrestrial exoplanet orbiting TRAPPIST-1,
#  an M8V dwarf star at 12.4 parsecs. Mass: 0.69 Earth masses,
#  Radius: 0.92 Earth radii, Period: 6.1 days, T_eq: 251K.
#  It orbits within the habitable zone."
```

**Build a simple web interface** (or CLI that's demo-friendly):
```bash
# Quick option: Streamlit (Python, zero frontend work)
uv add streamlit
```

```python
# app.py
import streamlit as st

st.title("🔭 AstroLLM — Astronomy Research Assistant")
st.caption("Ask me anything about astronomy. I'll search the literature and cite my sources.")

question = st.text_input("Your question:")
if question:
    with st.spinner("Searching papers and thinking..."):
        answer = ask_astrollm(question)
    st.markdown(answer)

    st.divider()
    st.caption("Sources retrieved:")
    for p in search_papers(question, top_k=3):
        st.markdown(f"- [{p[2]}] *{p[0]}* (relevance: {p[3]:.2f})")
```

**Milestone**: You can now show someone a web page where they ask astronomy
questions and get answers with paper citations. This is already more useful
than asking ChatGPT, because it's grounded in actual literature.

**Spend this week**: $0-20
**Running total**: $0-40 (well under budget)

---

### Week 5-6: SFT Dataset Creation (5-8 hrs/week)

**Goal**: Build a high-quality training dataset for fine-tuning.

**This is where the ADS data really pays off.** You've already downloaded thousands
of papers. Now use Claude to generate training pairs from them.

**Strategy: Generate 5,000 Q&A pairs for ~$30-50 in API costs**

```python
import anthropic

client = anthropic.Anthropic()

def generate_qa_pairs(abstract, title, bibcode):
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2000,
        messages=[{
            "role": "user",
            "content": f"""Based on this astronomy paper abstract, generate 3 diverse
question-answer pairs that a researcher or student might ask.

Title: {title}
Abstract: {abstract}

Requirements:
- Questions should range from factual to conceptual
- Answers should be 2-4 sentences, accurate, and cite the paper
- Include one question about methodology or implications
- Format as JSON array of {{"question": "...", "answer": "..."}}

Return ONLY valid JSON, no other text."""
        }]
    )
    return response.content[0].text

# Process 2,000 papers × 3 pairs = 6,000 Q&A pairs
# At ~500 tokens per call: 2,000 calls × 500 tokens ≈ 1M tokens
# Claude Sonnet cost: ~$3 input + $15 output ≈ $18-25 total
```

**Also create these specialized dataset types**:

1. **Object knowledge** (from SIMBAD + NED data): 1,000 pairs
   "What type of object is Cygnus X-1?" → "Cygnus X-1 is a high-mass X-ray binary..."

2. **Exoplanet facts** (from NASA Exoplanet Archive): 500 pairs
   "Is TRAPPIST-1 e in the habitable zone?" → "Yes, TRAPPIST-1 e orbits within..."

3. **Pedagogical explanations**: 500 pairs
   "Explain the Hertzsprung-Russell diagram to an undergraduate" → ...

4. **Astrobites-style summaries**: Download and format ~200 Astrobites articles as
   "summarize this paper in accessible language" training examples

**Format everything as chat-template JSONL**:
```json
{"messages": [
  {"role": "system", "content": "You are AstroLLM, an astronomy research assistant..."},
  {"role": "user", "content": "What is the Chandrasekhar limit?"},
  {"role": "assistant", "content": "The Chandrasekhar limit is approximately 1.4 solar masses..."}
]}
```

**Validate**: Run schema checks, spot-check 50 random samples for accuracy.
**Split**: 95% train, 5% eval.

**Spend these 2 weeks**: $30-50 (Claude API for data generation)
**Running total**: $30-90

---

### Week 7-8: First Fine-Tune (5-8 hrs/week)

**Goal**: QLoRA fine-tune Qwen3-4B on your astronomy dataset.

**Week 7: Setup and first run**

```bash
# On RunPod: launch RTX 4090 spot instance (~$0.40-0.60/hr)
# Pre-install: uv pip install unsloth transformers peft trl wandb

# Or use LLaMA-Factory for an even simpler setup:
uv pip install llama-factory
```

**Training with Unsloth (2-4x faster than vanilla)**:
```python
from unsloth import FastLanguageModel
import torch

# Load model with 4-bit quantization
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Qwen/Qwen3-4B",
    max_seq_length=4096,
    load_in_4bit=True,
)

# Add LoRA adapters
model = FastLanguageModel.get_peft_model(
    model,
    r=32,            # LoRA rank (start here, try 64 later)
    lora_alpha=64,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
)

# Train with SFTTrainer
from trl import SFTTrainer
from transformers import TrainingArguments

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset["train"],
    eval_dataset=dataset["eval"],
    args=TrainingArguments(
        output_dir="./astrollm-qwen3-4b-v001",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=10,
        eval_steps=100,
        save_steps=200,
        report_to="wandb",
    ),
    max_seq_length=4096,
)
trainer.train()
```

**Expected training time**: 2-4 hours on RTX 4090
**Expected cost**: $1-3 per run

**Week 8: Iterate and evaluate**

- Run 3-5 training experiments varying: learning rate, LoRA rank, epochs
- Evaluate each against 100 astronomy questions (manual A/B vs base model)
- Pick the best checkpoint
- Merge LoRA adapter: `model.merge_and_unload()`
- Quantize to GGUF: use llama.cpp `convert` + `quantize`

```bash
# Quantize for deployment
python -m llama_cpp.convert --outfile astrollm-4b-v001.gguf
./llama-quantize astrollm-4b-v001.gguf astrollm-4b-v001-Q4_K_M.gguf Q4_K_M
# Result: ~2.5GB file that runs on any laptop
```

**Spend these 2 weeks**: $20-60 (5-15 training runs × $3-5 each)
**Running total**: $50-150

---

### Week 9-10: Combine Fine-Tuned Model + RAG (5-8 hrs/week)

**Goal**: Replace the off-the-shelf Qwen3 in your RAG pipeline with YOUR model.

```bash
# Load your model in Ollama
ollama create astrollm -f Modelfile

# Where Modelfile contains:
# FROM ./astrollm-4b-v001-Q4_K_M.gguf
# SYSTEM "You are AstroLLM, an astronomy and astrophysics research assistant..."
# PARAMETER temperature 0.7
```

**Update RAG pipeline to use your model**:
```python
def ask_astrollm(question):
    papers = search_papers(question, top_k=5)
    context = format_context(papers)
    response = httpx.post("http://localhost:11434/api/generate", json={
        "model": "astrollm",  # YOUR fine-tuned model
        "prompt": build_prompt(question, context),
        "stream": False
    })
    return response.json()["response"]
```

**A/B test**: Run 50 questions through both base Qwen3-4B + RAG and AstroLLM + RAG.
Your fine-tuned model should give noticeably better astronomy-specific answers —
more precise terminology, better reasoning about physical processes, more natural
citations.

**Spend these 2 weeks**: $0-20 (local inference only)
**Running total**: $50-170

---

### Week 11-12: Web Interface + Deploy (5-8 hrs/week)

**Goal**: Ship the demo at astrollm.org.

**Two deployment options**:

**Option A — Simple (Streamlit + VPS, ~$20-40/month)**:
- Streamlit app on a small VPS (DigitalOcean, Hetzner)
- Ollama running your quantized model
- pgvector for RAG
- Good for: personal demo, sharing with friends/colleagues

**Option B — Proper (TanStack Start + Elysia, ~$50-100/month)**:
- Build with your preferred stack
- Deploy model via llama.cpp server or vLLM on a GPU VPS
- Better UX, streaming responses, proper citation UI
- Good for: public beta, portfolio piece

**Either way, you now have**:
- A fine-tuned astronomy LLM running at astrollm.org
- RAG-augmented answers grounded in real papers
- Citations linking back to NASA ADS
- Something you built from scratch, understanding every layer

**Spend these 2 weeks**: $20-100 (hosting + domain)
**Running total**: $70-270 (well within budget)

---

## 12-Week Budget Summary

| Phase | Weeks | GPU Cost | API Cost | Hosting | Total |
|-------|-------|----------|----------|---------|-------|
| RAG prototype | 1-4 | $0 | $0-20 | $0 | $0-20 |
| SFT data generation | 5-6 | $0 | $30-50 | $0 | $30-50 |
| Fine-tuning | 7-8 | $20-60 | $0 | $0 | $20-60 |
| Integration | 9-10 | $0 | $0 | $0-20 | $0-20 |
| Deployment | 11-12 | $0 | $0 | $40-100 | $40-100 |
| **Total** | **12 wks** | **$20-60** | **$30-70** | **$40-120** | **$90-250** |

That's 3 months of work for under $250 total, producing a working fine-tuned
astronomy LLM with RAG. Your monthly budget of $150-400 gives you plenty of room
for extra experiments and iteration.

---

## What to Study While Building

Since you're learning in parallel, here's what to read WHEN:

| Week | What You're Building | What to Study |
|------|---------------------|---------------|
| 1-2 | Data pipeline + embeddings | Raschka Ch. 1-2 (tokenization, embeddings) |
| 3-4 | RAG pipeline | Raschka Ch. 3 (attention) + RAG paper |
| 5-6 | SFT dataset creation | Raschka Ch. 5-6 (pre-training, fine-tuning) |
| 7-8 | QLoRA fine-tuning | LoRA paper + QLoRA paper + AstroSage paper |
| 9-10 | Model evaluation | AstroMLab benchmark papers |
| 11-12 | Deployment | Karpathy NanoGPT (deepen architecture understanding) |

This way you're always reading about what you're DOING that week.
Theory and practice reinforce each other.

---

## After Week 12: What's Next?

With a working demo in hand, you can:

1. **Scale the model**: Fine-tune Qwen3-8B (bigger, better) — same pipeline, bigger GPU
2. **Scale the data**: Harvest full ADS corpus, add SIMBAD/NED/PDS data
3. **Add tool calling**: Teach the model to query databases at inference time
4. **Build the model family**: Distill down to Nano (3B), scale up to Pro (32B)
5. **Go multimodal**: Add vision encoder for FITS images
6. **Publish**: Write up your approach and findings for ML4Astro workshop

The infrastructure you build in weeks 1-12 is the foundation for everything else.
Nothing is thrown away — it just gets bigger and better.
