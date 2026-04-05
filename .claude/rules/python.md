---
globs: ["packages/**/*.py", "scripts/**/*.py"]
---

# Python Code Rules

- Use type hints on all function signatures
- Follow ruff linting conventions (line length 100, target Python 3.11)
- Use `uv run python` for all script execution, never bare `python` or `pip`
- Add dependencies via `uv add`, not pip install
- Use `pathlib.Path` over `os.path`
- Use `httpx` for HTTP requests, not `requests` (async-ready)
- Use `typer` for CLI interfaces (already in deps)
- Use `rich` for terminal output formatting
- All data files in JSONL format with schema validation against `data/sft/schema.json`
- Every script must support `--dry-run` flag where applicable
- Log to stderr, output data to stdout
