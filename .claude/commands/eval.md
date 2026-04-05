# /eval — Evaluate a Model Checkpoint

Run the full evaluation suite against a model checkpoint and log results.

## Usage
```
/eval [model_path] [--benchmark all|astrolab|astroqa|pedagogy|custom]
```

## Evaluation Suite

### AstroMLab-1 Benchmark
- 4,425 MCQs from Annual Review of Astronomy and Astrophysics
- Measures: astronomical knowledge recall across subfields
- Baseline targets: Llama 3.1 8B base ~72%, AstroSage-8B ~80.9%

### Astro-QA Benchmark
- 3,082 questions across 6 types (MCQ, matching, terminology, short-answer)
- Covers: astrophysics, astrometry, celestial mechanics, history, techniques
- Bilingual (English + Chinese)

### Custom Pedagogy Eval
- 200 teaching scenario prompts (explain concept, Socratic dialogue, problem solving)
- Scored by: accuracy, clarity, appropriate depth, engagement
- Compared against base model responses in blind A/B tests

### Tool Integration Eval
- 100 queries requiring tool use (ADS lookup, SIMBAD query, coordinate conversion)
- Measures: correct tool selection, proper API usage, result interpretation

## Output
- Results logged to `docs/RESEARCH_LOG.md`
- W&B metrics updated
- Comparison table against previous checkpoints generated
- Delta from base model highlighted
