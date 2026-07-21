"""The manifest CLI — the publish gate: a broken seed must exit non-zero with valid=false."""

from __future__ import annotations

import json
from pathlib import Path

import manifest as manifest_mod
import pytest
import yaml  # noqa: F401  (parity with the harness deps; not used directly)
from conftest import example, negative_example
from typer.testing import CliRunner

runner = CliRunner()

STATIC_CONFIG = {"arm": "hybrid", "k": 100, "pool": 100, "pipeline": "pilot-retrieval-0.1.0"}


@pytest.fixture(autouse=True)
def _static_retriever_config(monkeypatch):
    # Keep tests offline/fast: never let the manifest reach for the rag/torch stack.
    monkeypatch.setattr(manifest_mod, "_retriever_config", lambda: STATIC_CONFIG)


def write_gold(tmp_path: Path, snapshot, records: list[dict], *, stamp: bool = True) -> Path:
    """Write records as gold JSONL, stamping the live snapshot hash unless stamp=False."""
    gold = tmp_path / "gold.jsonl"
    stamped = []
    for rec in records:
        rec = dict(rec)
        if stamp:
            rec["provenance"] = {
                **rec.get("provenance", {}),
                "corpus_snapshot_hash": snapshot.snapshot_hash,
            }
        stamped.append(rec)
    gold.write_text("\n".join(json.dumps(r) for r in stamped) + "\n")
    return gold


def run_build(gold: Path, corpus: Path, out: Path, *extra: str):
    return runner.invoke(
        manifest_mod.app,
        ["build", "--gold", str(gold), "--corpus", str(corpus), "--out", str(out), *extra],
    )


def test_build_writes_valid_manifest(tmp_path, snapshot, corpus_path):
    gold = write_gold(tmp_path, snapshot, [example()])
    out = tmp_path / "manifest.json"
    result = run_build(gold, corpus_path, out)
    assert result.exit_code == 0, result.output
    data = json.loads(out.read_text())
    assert data["valid"] is True
    assert data["counts"]["total"] == 1
    assert data["retriever_config"] == STATIC_CONFIG
    assert data["corpus"]["snapshot_hash"] == snapshot.snapshot_hash


def test_build_dry_run_writes_nothing(tmp_path, snapshot, corpus_path):
    gold = write_gold(tmp_path, snapshot, [example()])
    out = tmp_path / "manifest.json"
    result = run_build(gold, corpus_path, out, "--dry-run")
    assert result.exit_code == 0, result.output
    assert not out.exists()
    assert '"valid": true' in result.stdout


def test_build_fails_on_contract_violation(tmp_path, snapshot, corpus_path):
    gold = write_gold(tmp_path, snapshot, [example(answer="")])
    out = tmp_path / "manifest.json"
    result = run_build(gold, corpus_path, out)
    assert result.exit_code == 1
    assert json.loads(out.read_text())["valid"] is False  # written, but marked broken


def test_build_fails_on_shared_calibration_eval_query(tmp_path, snapshot, corpus_path):
    shared = example()["query"]
    records = [
        example(),
        negative_example(example_id="neg-shared-q", query=shared),
    ]
    gold = write_gold(tmp_path, snapshot, records)
    out = tmp_path / "manifest.json"
    result = run_build(gold, corpus_path, out)
    assert result.exit_code == 1
    data = json.loads(out.read_text())
    assert data["valid"] is False
    assert data["disjointness"]["disjoint"] is False
    assert "share queries" in result.output


def test_build_fails_on_corpus_hash_mismatch(tmp_path, snapshot, corpus_path):
    drifted = example()
    drifted["provenance"] = {"corpus_snapshot_hash": "sha256:not-this-corpus"}
    gold = write_gold(tmp_path, snapshot, [drifted], stamp=False)
    out = tmp_path / "manifest.json"
    result = run_build(gold, corpus_path, out)
    assert result.exit_code == 1
    data = json.loads(out.read_text())
    assert data["valid"] is False
    assert data["corpus_consistency"]["all_examples_match"] is False


def test_status_reports_totals(tmp_path, snapshot, corpus_path):
    gold = write_gold(tmp_path, snapshot, [example()])
    result = runner.invoke(
        manifest_mod.app, ["status", "--gold", str(gold), "--corpus", str(corpus_path)]
    )
    assert result.exit_code == 0, result.output
    assert "total=1 valid_errors=0" in result.stdout
