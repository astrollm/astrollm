"""Shared setup for the evaluation-harness tests.

The evaluation modules run as scripts, not as an installed package, so ``packages/evaluation/src``
goes on ``sys.path``. Only pure modules are imported here: no test touches the database, the
retriever, the embedder or a tokenizer download.
"""

from __future__ import annotations

import sys
from pathlib import Path

EVAL_SRC = Path(__file__).resolve().parents[1] / "src"
if str(EVAL_SRC) not in sys.path:
    sys.path.insert(0, str(EVAL_SRC))
