# AstroLLM Architecture

## Training Pipeline

### Stage 1: Continued Pre-Training (CPT)

**Purpose**: Inject domain knowledge into the base model's weights.

**Data**: Raw astronomy text — arXiv papers, textbook chapters, encyclopedia entries.

**Method**:
- Standard causal language modeling (next-token prediction)
- No chat template — just raw text with document boundaries
- Learning rate: 1e-5 to 5e-5 (much lower than pre-training)
- Warm up for 5-10% of steps, cosine decay to 0
- Monitor perplexity on held-out astro text AND general text (detect forgetting)

**Risk**: Catastrophic forgetting. AstroLLaMA's failure mode:
- They CPT'd on only abstracts → model lost general capabilities
- AstroSage fixed this by using full papers + diverse general data in CPT

**Our approach**: Skip CPT initially. Start with SFT-only on a strong instruct model.
Add CPT only if SFT alone doesn't achieve sufficient domain knowledge.

### Stage 2: Supervised Fine-Tuning (SFT)

**Purpose**: Teach the model to respond helpfully to astronomy queries.

**Data types** (in order of priority):
1. **Domain Q&A pairs**: Questions about astronomy concepts, answered with citations
2. **Reasoning chains**: Multi-step derivations (stellar evolution, orbital mechanics)
3. **Teaching dialogues**: Socratic conversations explaining concepts at different levels
4. **Tool-use examples**: Queries that require calling ADS, SIMBAD, or running Astropy code
5. **Paper analysis**: Summarize a paper, identify key findings, compare with related work

**SFT data format** (JSONL):
```json
{
  "messages": [
    {"role": "system", "content": "You are AstroLLM, an astronomy research assistant..."},
    {"role": "user", "content": "What is the Chandrasekhar limit and why is it important?"},
    {"role": "assistant", "content": "The Chandrasekhar limit is approximately 1.4 solar masses..."}
  ],
  "metadata": {
    "source": "generated",
    "topic": "stellar_physics",
    "difficulty": "undergraduate",
    "requires_tools": false
  }
}
```

**Loss masking**: CRITICAL — train only on assistant tokens. Mask user and system tokens.

### Stage 3: Model Merging

**Purpose**: Recover general capabilities lost during SFT.

**Method**: Merge the SFT checkpoint with the original instruct model.

Techniques to experiment with:
- **SLERP** (Spherical Linear Interpolation): Smooth interpolation between two models
- **TIES** (Trim, Elect, Merge): Prunes small deltas, resolves sign conflicts
- **DARE** (Drop And REscale): Randomly drops delta parameters, rescales rest

Use mergekit library. Evaluate merged model on BOTH astronomy AND general benchmarks.

### Stage 4: Direct Preference Optimization (DPO) — Phase 3+

**Purpose**: Align model with human preferences for response quality.

**Data**: Pairs of (chosen, rejected) responses to the same prompt.
- Generate multiple responses, have astronomers rank them
- Or use AI-as-judge (Claude) with detailed rubric

---

## RAG Architecture

### Embedding Pipeline

```
Paper PDF → Extract text → Chunk (section-aware) → Embed → Store in pgvector
```

**Chunking strategy**:
- Respect paper structure: don't chunk across section boundaries
- Max chunk size: 512 tokens (with 50-token overlap)
- Each chunk preserves metadata: paper_id, arxiv_id, section, authors, date

**Embedding model options**:
- `BAAI/bge-large-en-v1.5` (general, good baseline)
- `nomic-ai/nomic-embed-text-v1.5` (long context)
- Fine-tuned astronomy-specific embedder (Phase 4)

### Retrieval Pipeline

```
User query → Embed → Hybrid search (semantic + keyword) → Rerank → Top-k chunks → Inject into prompt
```

**Hybrid search**: Combine pgvector cosine similarity with full-text search (ts_vector).
Weight: 70% semantic, 30% keyword (tunable).

**Reranking**: Cross-encoder reranker on top-k candidates for precision.

**Prompt injection pattern**:
```
System: You are AstroLLM. Use the following retrieved context to answer.
Always cite papers by their arXiv ID when referencing specific findings.

Context:
[Paper: 2401.12345, Section: Results]
"We measure the stellar mass of NGC 1234 to be 10^11.2 solar masses..."

[Paper: 2312.98765, Section: Discussion]  
"The mass-metallicity relation at z>2 shows significant scatter..."

User: What do we know about the stellar mass of NGC 1234?
```

---

## Tool Integration Architecture

### Function Calling Pattern

AstroLLM uses a tool-calling pattern where the model can request external tool execution:

```json
{
  "tool_call": {
    "name": "simbad_query",
    "arguments": {
      "object_name": "NGC 1234",
      "fields": ["coordinates", "morphtype", "redshift", "flux_V"]
    }
  }
}
```

The inference server intercepts tool calls, executes them, and injects results back:

```json
{
  "tool_result": {
    "name": "simbad_query",
    "result": {
      "ra": "03h 10m 03.1s",
      "dec": "-07d 35m 12s",
      "morphtype": "SAB(s)bc",
      "redshift": 0.003627,
      "flux_V": 11.3
    }
  }
}
```

### Available Tools

| Tool | API | Capability |
|------|-----|-----------|
| `ads_search` | NASA ADS API | Search papers by keyword, author, date, citations |
| `ads_paper` | NASA ADS API | Get full metadata, abstract, citation list for a paper |
| `simbad_query` | SIMBAD TAP | Look up astronomical object properties |
| `vizier_catalog` | VizieR TAP | Query astronomical catalogs |
| `astropy_calc` | Local Python | Coordinate transforms, unit conversions, cosmological calculations |
| `mast_search` | MAST API | Search HST/JWST/Kepler/TESS observations |
| `ned_query` | NED API | Extragalactic object lookup, distances, cross-IDs |

### Training for Tool Use

SFT data includes tool-use examples:
```json
{
  "messages": [
    {"role": "user", "content": "What type of galaxy is M87 and what is its redshift?"},
    {"role": "assistant", "content": null, "tool_calls": [
      {"name": "simbad_query", "arguments": {"object_name": "M87", "fields": ["morphtype", "redshift"]}}
    ]},
    {"role": "tool", "content": "{\"morphtype\": \"cD;E+0-1\", \"redshift\": 0.004283}"},
    {"role": "assistant", "content": "M87 (also known as Virgo A or NGC 4486) is classified as a cD galaxy — a supergiant elliptical typically found at the centers of galaxy clusters. Its redshift of z = 0.004283 corresponds to a recessional velocity of about 1,284 km/s, placing it at a distance of roughly 16.4 Mpc (53.5 million light-years) in the Virgo Cluster."}
  ]
}
```

---

## Deployment Architecture

### Cloud Training (RunPod / Lambda Labs)

```
Docker Image (pre-built) → Cloud GPU Instance → Training Script → W&B Logging
                                                       ↓
                                              Checkpoint → HuggingFace Hub
```

Docker image includes: PyTorch, Transformers, PEFT, TRL, Unsloth, flash-attn, W&B.
Pre-built to minimize startup time on spot instances.

### Inference Serving

**Development**: llama.cpp with GGUF quantized model (runs on CPU/Mac for testing)

**Production**: vLLM on a GPU VPS
- Continuous batching for concurrent users
- LoRA adapter hot-swapping for A/B testing
- Streaming output via SSE

### Web Application

```
Client (TanStack Start) → API (Elysia) → Inference (vLLM) + RAG (pgvector) + Tools (ADS/SIMBAD)
```

Deployed as Docker Compose stack:
- `web`: TanStack Start frontend
- `api`: Elysia backend
- `inference`: vLLM model server
- `db`: PostgreSQL with pgvector
- `embedding`: Embedding model server
