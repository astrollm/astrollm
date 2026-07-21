"""Shared fixtures for the sft/ contract-logic tests.

The sft modules import each other as flat siblings (``from schema import ...``) because they run
as scripts, not as an installed package — so the sft directory itself goes on ``sys.path``.

Everything here is offline-only: a tiny in-memory corpus written to ``tmp_path`` stands in for the
frozen 2,500-abstract snapshot, and no test touches the database or the rag/torch stack.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

SFT_DIR = Path(__file__).resolve().parents[1] / "src" / "sft"
if str(SFT_DIR) not in sys.path:
    sys.path.insert(0, str(SFT_DIR))


# ── Tiny corpus ──────────────────────────────────────────────────────────────

CORPUS_RECORDS = [
    {
        "bibcode": "2023A&A...100..001A",
        "title": "JWST transmission spectroscopy of WASP-39b",
        "abstract": "We report the C/O ratio of WASP-39b measured with JWST NIRSpec.",
    },
    {
        "bibcode": "2022ApJ...900..002B",
        "title": "Thermal inversions in ultra-hot Jupiters",
        "abstract": "Evidence for a stratospheric thermal inversion in an ultra-hot Jupiter.",
    },
    {
        "bibcode": "2021MNRAS.500..003C",
        "title": "Cloud formation in warm Neptunes",
        "abstract": "Microphysical cloud models for warm Neptune atmospheres.",
    },
]

IN_POOL = ("2023A&A...100..001A", "2022ApJ...900..002B")
IN_CORPUS_NOT_RETRIEVED = "2021MNRAS.500..003C"
NOT_IN_CORPUS = "2099ZZZZ.999..999Z"


@pytest.fixture()
def corpus_path(tmp_path: Path) -> Path:
    path = tmp_path / "abstracts.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in CORPUS_RECORDS) + "\n")
    return path


@pytest.fixture()
def snapshot(corpus_path: Path):
    from corpus import CorpusSnapshot

    return CorpusSnapshot.load(corpus_path)


# ── Example builders ─────────────────────────────────────────────────────────


def retrieved_context(bibcodes: tuple[str, ...] = IN_POOL) -> list[dict]:
    return [
        {"bibcode": b, "chunk_id": f"{b}:abstract:0", "retrieval_score": 1.0 / (i + 1)}
        for i, b in enumerate(bibcodes)
    ]


def claim(
    text: str = "WASP-39b shows a sub-solar C/O ratio.",
    cited_bibcode: str | None = IN_POOL[0],
    support_span: str | None = "the C/O ratio of WASP-39b measured with JWST",
    supported: bool = True,
) -> dict:
    return {
        "text": text,
        "cited_bibcode": cited_bibcode,
        "support_span": support_span,
        "supported": supported,
    }


def example(**overrides) -> dict:
    """A valid lit_qa/eval example; override fields to construct violations."""
    base = {
        "example_id": "lit_qa-eval-deadbeef",
        "task_family": "lit_qa",
        "partition": "eval",
        "query": "What did JWST measure for the C/O ratio of WASP-39b?",
        "retrieved_context": retrieved_context(),
        "answer": "JWST NIRSpec measured a sub-solar C/O ratio for WASP-39b.",
        "claims": [claim()],
        "abstention_reason": None,
        "is_negative": False,
        "negative_type": None,
        "provenance": {"author": "human", "corpus_snapshot_hash": "sha256:stub"},
    }
    base.update(overrides)
    return base


def abstention_example(**overrides) -> dict:
    base = example(
        example_id="abstention-eval-cafef00d",
        task_family="abstention",
        query="What is the albedo of LHS 475b?",
        answer="The retrieved abstracts do not address the albedo of LHS 475b.",
        claims=[claim(text="No retrieved abstract discusses LHS 475b.",
                      cited_bibcode=None, support_span=None)],
        abstention_reason="absent",
    )
    base.update(overrides)
    return base


def negative_example(**overrides) -> dict:
    base = example(
        example_id="lit_qa-calibration-0badc0de-neg",
        partition="calibration",
        is_negative=True,
        negative_type="claim_not_supported",
        claims=[claim(supported=False)],
    )
    base.update(overrides)
    return base
