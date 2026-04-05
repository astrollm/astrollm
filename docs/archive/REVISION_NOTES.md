# AstroLLM — Document Revision Notes

*Post peer review (Round 1 + Round 2, March 2026)*

This file tracks what changed in each document after the ChatGPT peer review process.
The V1_FINAL_PLAN.md is the authoritative execution document. These notes explain how
the earlier planning documents relate to it.

---

## Document Status

| Document | Status | Action Needed |
|----------|--------|---------------|
| **V1_FINAL_PLAN.md** | NEW — authoritative v1 execution plan | Follow this for weeks 1-12 |
| **MASTER_PLAN.md** | NEW — long-term vision (Phase 1-5) | Reference for motivation and direction |
| **ARCHITECTURE_V2.md** | Valid as VISION document | Add a header noting v1 scope is narrower than described |
| **DATA_SOURCES.md** | Valid as REFERENCE | All sources are real; v1 uses only ADS + SIMBAD + NEA |
| **GETTING_STARTED_12_WEEKS.md** | SUPERSEDED by V1_FINAL_PLAN.md | Updated version below incorporates peer review |
| **ARCHITECTURE.md** | SUPERSEDED by ARCHITECTURE_V2.md | Keep for historical reference |
| **LEARNING_PATH.md** | Valid | No changes needed |
| **QUICKSTART.md** | Valid as quick reference | Aligns with V1 Final Plan |
| **RESEARCH_LOG.md** | Valid — start using immediately | Add error taxonomy categories from peer review |

---

## Key Changes from Peer Review

### 1. Retrieval pipeline upgraded
**Before**: Simple pgvector semantic search with BGE embeddings
**After**: Three-stage pipeline:
- Stage 1: Hybrid recall (BM25 sparse + dense, merged with reciprocal-rank fusion)
- Stage 2: Cross-encoder or ColBERTv2 reranking on top 50-100
- Stage 3: Astronomy-aware filtering (SIMBAD alias expansion, ADS fielded search, year/bibstem filters)
**New libraries**: Pyserini, SPECTER2, ColBERTv2

### 2. Tool-use SFT deferred
**Before**: Tool-use examples in first SFT dataset (10% allocation)
**After**: Tool calling handled at orchestration/prompt layer in v1. Tool-use SFT begins in Phase 2 from logged real interactions. First SFT tool-call examples kept under 10-15%.

### 3. v1 scope narrowed
**Before**: All databases, all tools, model family implied
**After**: Explicitly Core-only. ADS + SIMBAD + thin Exoplanet Archive. NED/PDS/Gaia/MAST deferred to Phase 2.

### 4. Custom evaluation tracks added
**Before**: AstroMLab-1 and Astro-QA only
**After**: AstroMLab-1 subset + 4 custom tracks (25 examples each):
- Grounding/citation accuracy
- Tool routing accuracy
- Abstention under weak retrieval
- Pedagogy quality

### 5. Astronomy-specific error taxonomy added
Track from day one: citation errors, object-identity errors, unit-system errors, coordinate/epoch errors, catalog-semantic errors, literature-timeline errors, database-boundary errors, tool errors.

### 6. SFT mixture weighted
**Before**: Even distribution across sources
**After**: 30% literature Q&A, 25% object retrieval, 20% summarization, 15% pedagogy, 10% tool-call

### 7. Dual-track operating model
**Before**: Single blended track
**After**: Track A (product — ship the beta) and Track B (learning — training experiments and skill building). Week 8 = deployment decision gate, not training cessation gate.

### 8. Grounding policy defined
**Before**: Implicit
**After**: Explicit — cite when making specific claims; abstain when retrieval returns <2 relevant papers; express uncertainty when papers disagree.

### 9. Primary user locked
**Before**: "Researchers, students, enthusiasts" (too broad)
**After**: Graduate students and early-career researchers.

---

## How to Use These Documents

**Starting the project**: Read V1_FINAL_PLAN.md first. This is your operating plan.

**Understanding the vision**: Read MASTER_PLAN.md for where AstroLLM is going long-term.

**Deep technical reference**: ARCHITECTURE_V2.md for model family, ADS strategy, multimodal approach. DATA_SOURCES.md for per-database API patterns and code examples. Both are valid reference material — just remember v1 scope is narrower than what they describe.

**Learning**: LEARNING_PATH.md is unchanged and aligned with the weekly study schedule.

**Day-to-day**: RESEARCH_LOG.md for tracking experiments. Add error taxonomy categories to the template.
