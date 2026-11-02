"""Shared setup for the inference-package tests.

The inference modules run as scripts and import each other as flat siblings, so ``src`` itself
goes on ``sys.path``. Everything here is offline: tests monkeypatch the astroquery clients and
never touch the network.
"""

from __future__ import annotations

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
