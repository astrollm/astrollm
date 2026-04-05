# Data Pipeline Package

Handles all data ingestion, processing, and dataset creation for AstroLLM training.

## Data Sources

### Primary: arXiv astro-ph
- ~300,000+ papers from 2007 to present
- Access via: arXiv bulk data on S3, OAI-PMH API, or Kaggle mirror
- Content: LaTeX source files → extract clean text
- Sections of interest: abstracts, introductions, methods, results, conclusions

### Secondary Sources
- **NASA ADS metadata**: Paper titles, authors, affiliations, citation counts, keywords
- **Astronomy textbooks** (public domain): Carroll & Ostlie excerpts, Ryden, open textbooks
- **Wikipedia astronomy**: ~5,000 astronomy-related articles (good for pedagogy data)
- **IAU nomenclature**: Official naming conventions, object catalogs

## Processing Pipeline

```
1. Download    → data/raw/arxiv/
2. Extract     → LaTeX → clean text (remove figures, tables, bibliography)
3. Section     → Split into abstract, intro, methods, results, conclusion
4. Clean       → Remove LaTeX artifacts, normalize whitespace, fix encoding
5. Quality     → Filter by length, language detection, topic classification
6. Deduplicate → Fuzzy dedup across papers (many share boilerplate)
7. Chunk       → Section-aware chunking for RAG (512 tokens, 50 overlap)
8. Generate    → Create SFT pairs using Claude API
9. Validate    → Schema check, spot-check samples
10. Package    → JSONL with manifest.json
```

## SFT Dataset Types

### 1. Domain Q&A (target: 50,000+ pairs)
```json
{"messages": [
  {"role": "system", "content": "You are AstroLLM..."},
  {"role": "user", "content": "Explain the Hertzsprung-Russell diagram."},
  {"role": "assistant", "content": "The HR diagram plots stellar luminosity against..."}
]}
```

### 2. Reasoning Chains (target: 10,000+ examples)
```json
{"messages": [
  {"role": "user", "content": "A star has apparent magnitude 5 and is at 100 parsecs. What is its absolute magnitude?"},
  {"role": "assistant", "content": "<think>\nUsing the distance modulus: m - M = 5 log10(d/10)\n5 - M = 5 log10(100/10) = 5 log10(10) = 5\nM = 5 - 5 = 0\n</think>\n\nThe absolute magnitude is 0. Using the distance modulus formula..."}
]}
```

### 3. Socratic Teaching (target: 5,000+ dialogues)
Multi-turn conversations where AstroLLM guides understanding through questions.

### 4. Tool-Use Examples (target: 5,000+ examples)
Queries that require calling external tools (ADS, SIMBAD, Astropy).

### 5. Paper Analysis (target: 10,000+ examples)
Summarize papers, extract key findings, identify methodology, compare with related work.

## Scripts

```bash
# Download arXiv papers
python src/download_arxiv.py --category astro-ph --start 2020 --end 2024 --output data/raw/

# Process LaTeX to clean text
python src/process_papers.py --input data/raw/ --output data/processed/

# Generate SFT Q&A pairs via Claude API
python src/generate_sft.py \
  --input data/processed/ \
  --output data/sft/ \
  --api-key $ANTHROPIC_API_KEY \
  --num-pairs 10000 \
  --type domain_qa

# Validate dataset
python src/validate_dataset.py --input data/sft/train.jsonl

# Create train/eval split
python src/split_dataset.py --input data/sft/ --eval-ratio 0.05
```

## Quality Control Checklist
- [ ] No duplicate Q&A pairs (fuzzy matching on questions)
- [ ] All JSON valid and schema-conformant
- [ ] No leaked LaTeX artifacts in clean text
- [ ] Balanced topic distribution across astronomy subfields
- [ ] Difficulty distribution: 30% intro, 40% intermediate, 30% advanced
- [ ] Tool-use examples cover all supported tools
- [ ] Spot-check 100 random samples for accuracy
