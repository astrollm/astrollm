"""Three-way D1 delivery parity: measurement (pool_sweep), authoring (author.prepare) and serving
(copilot grounded mode) must hand the model the same context for the same pool, k and budget.

Offline: an in-memory corpus snapshot, a fake whitespace token counter, no DB, no tokenizer
download. Importing pool_sweep / copilot only puts their sibling directories on ``sys.path``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
for _p in (REPO / "packages" / "data-pipeline" / "src" / "sft",
           REPO / "packages" / "inference" / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import author  # noqa: E402  (data-pipeline/src/sft)
import context  # noqa: E402
import copilot  # noqa: E402  (inference/src)
import pool_sweep  # noqa: E402  (evaluation/src)
from corpus import CorpusSnapshot  # noqa: E402
from schema import Retrieved  # noqa: E402

N_POOL = 12


def _snapshot() -> CorpusSnapshot:
    records = [
        {"bibcode": f"2024ApJ...{i:03d}..001A", "title": f"Paper {i} on exoplanet atmospheres",
         "abstract": " ".join(f"word{i}_{j}" for j in range(20 + 7 * (i % 5)))}
        for i in range(1, N_POOL + 1)
    ]
    return CorpusSnapshot(records, Path("in-memory.jsonl"))


def _assembled(snapshot: CorpusSnapshot) -> context.AssembledContext:
    bibs = sorted(snapshot.bibcodes)
    return context.AssembledContext(
        query="CO2 in WASP-39b?",
        retrieved=[Retrieved(bibcode=b, chunk_id=snapshot.chunk_id(b),
                             retrieval_score=round(1.0 / (60 + i), 6))
                   for i, b in enumerate(bibs, 1)],
        retriever_run_id="rrf-pool100-test", corpus_snapshot_hash=snapshot.snapshot_hash,
        retriever_config={"arm": "hybrid", "k": 100, "pool": 100, "rrf_k": 60},
        assembled_at="T",
    )


def count_tokens(text: str) -> int:
    return len(text.split())


@pytest.mark.parametrize(("k", "budget"), [(5, 10_000), (10, 300), (10, 120), (30, 16_384)])
def test_measurement_authoring_and_serving_deliver_identically(k: int, budget: int) -> None:
    snapshot = _snapshot()
    assembled = _assembled(snapshot)

    measured = pool_sweep._d1_delivery(context, assembled, snapshot, count_tokens, k, budget)
    authored = author.deliver_context(assembled, snapshot, k, budget, count_tokens)
    served = copilot.build_grounded_context(
        copilot.Grounding(assembled=assembled, k=k), snapshot, count_tokens, budget
    )

    assert measured.text == authored.text == served.text
    assert measured.n_delivered == authored.n_delivered == len(served.delivered)
    assert list(measured.delivered) == list(authored.delivered) == served.delivered
    assert list(measured.dropped) == list(authored.dropped) == served.dropped
    assert measured.delivered_tokens == authored.delivered_tokens == served.tokens
    # and the delivered text is render_context's own top-n slice (never re-implemented)
    assert measured.text == context.render_context(assembled, snapshot,
                                                   limit=measured.n_delivered)


def test_parity_cases_exercise_the_overflow_policy() -> None:
    snapshot = _snapshot()
    assembled = _assembled(snapshot)
    tight = author.deliver_context(assembled, snapshot, 10, 120, count_tokens)
    assert tight.overflow and tight.dropped and tight.n_delivered >= 1
