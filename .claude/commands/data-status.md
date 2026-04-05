# /data-status — Report Dataset Health

Check the state of all data directories and report counts, sizes, and validation status.

## Usage
```
/data-status [--validate]
```

## Workflow

1. Scan all data directories and report:

### `data/raw/`
- Count of downloaded files (papers, catalogs)
- Total size on disk
- Date of most recent download
- Sources present (arXiv abstracts, ADS metadata, exoplanet archive, SIMBAD objects)

### `data/processed/`
- Count of processed documents
- Chunking stats: total chunks, avg tokens per chunk
- Any processing errors or skipped files

### `data/sft/`
- Count of training examples in `train.jsonl`
- Count of eval examples in `eval.jsonl`
- Dataset mixture breakdown (literature Q&A, object retrieval, summarization, pedagogy, tool-call)
- If `--validate` flag: run schema validation against `data/sft/schema.json`

### `data/eval/`
- Count of evaluation examples per track (grounding, tool routing, abstention, pedagogy)
- Retrieval gold set size (if exists)

### `models/`
- List of checkpoints with sizes
- Latest checkpoint date
- Any GGUF quantized models present

2. Report format:

```
AstroLLM Data Status
====================

data/raw/          3,200 files    1.2 GB    Last updated: 2026-04-10
  arXiv abstracts: 3,000
  ADS metadata:    2,800
  Exoplanet table: 1 (5,832 planets)
  SIMBAD objects:  0

data/processed/    2,800 docs     450 MB    Last updated: 2026-04-12
  Total chunks:    14,200
  Avg tokens/chunk: 380

data/sft/          5,200 examples  28 MB    Last updated: 2026-04-15
  train.jsonl:     4,940 (95%)
  eval.jsonl:      260 (5%)
  Mixture: 30% lit-qa | 25% object | 20% summary | 15% pedagogy | 10% tool
  Schema valid:    yes

data/eval/         125 examples    1.2 MB
  Grounding:       30
  Tool routing:    30
  Abstention:      30
  Pedagogy:        35

models/            1 checkpoint    4.5 GB
  qwen3-8b-qlora-sft-v001/final   2026-04-20
  astrollm-8b-v001-q4_k_m.gguf    2.5 GB
```

3. Flag any issues:
   - Missing directories or empty directories that should have data
   - Schema validation failures (if `--validate`)
   - Train/eval split ratio outside 90-95% / 5-10% range
   - Stale data (processed older than raw, suggesting re-processing needed)
