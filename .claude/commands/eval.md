# /eval — Evaluate a Model Checkpoint

Run the evaluation suite against a model checkpoint, compare to baselines, and log results.

## Usage
```
/eval [model_path] [--benchmark all|astrolab|custom|retrieval]
```

Examples:
- `/eval models/qwen3-8b-qlora-sft-v001/final --benchmark all`
- `/eval Qwen/Qwen3-8B --benchmark astrolab` (baseline eval)
- `/eval models/latest --benchmark custom`

## Evaluation Suite

### AstroMLab-1 Benchmark
- 4,425 MCQs from Annual Review of Astronomy and Astrophysics
- Measures: astronomical knowledge recall across subfields
- Baseline targets: Qwen3-8B base (TBD), AstroSage-8B ~80.9%
- Run: `uv run python packages/evaluation/src/run_benchmark.py --model {path} --benchmark astrolab-1`

### Custom Evaluation Tracks (25+ examples each)

| Track | What it measures | Key metrics |
|-------|-----------------|-------------|
| **Grounding/citation accuracy** | Does the model cite correctly? Are citations real? | Citation precision, hallucinated reference rate |
| **Tool routing accuracy** | Does it call the right tool (ADS/SIMBAD/Astropy)? | Tool selection F1, unnecessary call rate |
| **Abstention under weak retrieval** | Does it say "I don't know" when evidence is thin? | Abstention recall, false confidence rate |
| **Pedagogy quality** | Can it explain at the right level? | Depth appropriateness, Socratic engagement |

### Astronomy Error Taxonomy
Track these error categories from day one:
- **Citation errors**: wrong citation, right citation wrong synthesis, missed obvious paper
- **Object-identity errors**: alias confusion, host star vs planet, wrong counterpart
- **Unit-system errors**: cgs vs SI, magnitudes vs fluxes, luminosity vs flux
- **Coordinate/epoch errors**: J2000 confusion, equatorial vs galactic
- **Catalog-semantic errors**: measured vs derived parameters, misuse of defaults
- **Literature-timeline errors**: citing superseded results as current consensus
- **Database-boundary errors**: using wrong source type for the question
- **Tool errors**: should have been called but wasn't, called unnecessarily

### Retrieval Evaluation (independent from LLM)
- Gold set: 100+ queries with known relevant papers
- Metrics: Recall@k, MRR, nDCG at each pipeline stage
- Object-resolution accuracy (SIMBAD alias expansion)
- Run: `uv run python packages/evaluation/src/eval_retrieval.py --gold-set data/eval/retrieval_gold.jsonl`

## Output
1. Results table printed to terminal (base vs this run vs previous best)
2. Entry appended to `docs/RESEARCH_LOG.md` under the experiment's EXP-XXX section
3. W&B metrics updated with run ID
4. Delta from base model highlighted (improvements in green, regressions in red)

## Comparison Template
```
| Metric                  | Base (Qwen3-8B) | Previous Best | This Run | Delta |
|-------------------------|-----------------|---------------|----------|-------|
| AstroMLab-1 (subset)    |                 |               |          |       |
| Grounding accuracy       |                 |               |          |       |
| Tool routing F1          |                 |               |          |       |
| Abstention recall        |                 |               |          |       |
| Pedagogy score           |                 |               |          |       |
```
