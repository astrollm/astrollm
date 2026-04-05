# Evaluation Package

Benchmark runners and custom evaluation suites for AstroLLM.

## Benchmarks

### AstroMLab-1
- 4,425 MCQs from Annual Review of Astronomy and Astrophysics
- Source: github.com/AstroMLab (check for public dataset release)
- If not publicly available, recreate methodology from their paper

### Astro-QA
- 3,082 questions across 6 types
- Source: described in PMC article (check for public release)

### Custom Pedagogy Eval
- Test teaching quality, explanation clarity, Socratic dialogue ability
- 200 hand-crafted scenarios covering intro → graduate level
- Evaluated via Claude-as-judge with detailed rubric

### Tool-Use Eval
- 100 queries requiring correct tool selection and execution
- Tests: ADS search, SIMBAD query, Astropy calculation, multi-tool chains

## Usage

```bash
# Run all benchmarks
python run_benchmark.py --model models/latest/ --benchmark all

# Run specific benchmark
python run_benchmark.py --model models/latest/ --benchmark astrolab-1

# Compare two models
python run_benchmark.py \
  --model models/v002/ \
  --compare meta-llama/Llama-3.1-8B-Instruct \
  --benchmark all
```

## Output
Results saved to `docs/RESEARCH_LOG.md` and logged to W&B.
