"""Default artifact locations for the gold-seed harness.

All under gitignored ``data/sft/``. Worksheets hold reproduced (copyrighted) abstract text and are
gitignored separately; only the abstract-text-free gold JSONL and manifest are publishable outputs.
"""

from pathlib import Path

DEFAULT_GOLD_PATH = Path("data/sft/gold_seed.jsonl")
DEFAULT_MANIFEST_PATH = Path("data/sft/manifest.json")
DEFAULT_WORKSHEET_DIR = Path("data/sft/worksheets")
