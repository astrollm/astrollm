"""The author CLI's commit path — the gate every hand-authored example passes through.

`prepare` needs the live retriever for context assembly, so only its offline flag-validation is
tested here; `commit` is fully offline (worksheet + corpus snapshot) and is tested end-to-end.
"""

from __future__ import annotations

import json
from pathlib import Path

import author
import yaml
from conftest import example
from typer.testing import CliRunner

runner = CliRunner()


def make_worksheet(tmp_path: Path, snapshot, **overrides) -> Path:
    """A filled worksheet as `prepare` would emit it and the labeler would complete it."""
    data = example(provenance={"author": "human", "corpus_snapshot_hash": snapshot.snapshot_hash})
    data.update(overrides)
    sheet = {
        "_locked": {
            "example_id": data["example_id"],
            "task_family": data["task_family"],
            "partition": data["partition"],
            "query": data["query"],
            "retrieved_context": data["retrieved_context"],
            "provenance": data["provenance"],
            "retriever_config": {"arm": "hybrid", "pool": 100},  # dropped at commit
        },
        "answer": data["answer"],
        "claims": data["claims"],
        "abstention_reason": data["abstention_reason"],
        "is_negative": data["is_negative"],
        "negative_type": data["negative_type"],
    }
    path = tmp_path / f"{data['example_id']}.yaml"
    path.write_text(yaml.safe_dump(sheet, sort_keys=False))
    return path


def commit(worksheet: Path, gold: Path, corpus: Path, *extra: str):
    return runner.invoke(
        author.app,
        ["commit", "--worksheet", str(worksheet), "--gold", str(gold),
         "--corpus", str(corpus), *extra],
    )


# ── commit: happy path ───────────────────────────────────────────────────────


def test_commit_appends_valid_example(tmp_path, snapshot, corpus_path):
    gold = tmp_path / "gold.jsonl"
    sheet = make_worksheet(tmp_path, snapshot)
    result = commit(sheet, gold, corpus_path)
    assert result.exit_code == 0, result.output
    lines = gold.read_text().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["example_id"] == "lit_qa-eval-deadbeef"
    assert "retriever_config" not in record  # stripped from the GoldExample contract
    assert record["provenance"]["corpus_snapshot_hash"] == snapshot.snapshot_hash
    assert "lit_qa-eval-deadbeef" in result.stdout


def test_commit_dry_run_does_not_append(tmp_path, snapshot, corpus_path):
    gold = tmp_path / "gold.jsonl"
    sheet = make_worksheet(tmp_path, snapshot)
    result = commit(sheet, gold, corpus_path, "--dry-run")
    assert result.exit_code == 0, result.output
    assert not gold.exists()


# ── commit: the guards ───────────────────────────────────────────────────────


def test_commit_rejects_corpus_hash_drift(tmp_path, snapshot, corpus_path):
    gold = tmp_path / "gold.jsonl"
    sheet = make_worksheet(
        tmp_path, snapshot, provenance={"corpus_snapshot_hash": "sha256:someone-elses-corpus"}
    )
    result = commit(sheet, gold, corpus_path)
    assert result.exit_code == 1
    assert "corpus hash drift" in result.output
    assert not gold.exists()


def test_commit_rejects_worksheet_without_locked_block(tmp_path, snapshot, corpus_path):
    gold = tmp_path / "gold.jsonl"
    bare = tmp_path / "bare.yaml"
    bare.write_text(yaml.safe_dump({"answer": "x"}))
    result = commit(bare, gold, corpus_path)
    assert result.exit_code == 1
    assert "_locked" in result.output


def test_commit_rejects_unfilled_worksheet(tmp_path, snapshot, corpus_path):
    gold = tmp_path / "gold.jsonl"
    sheet = make_worksheet(tmp_path, snapshot, answer="")
    result = commit(sheet, gold, corpus_path)
    assert result.exit_code == 1
    assert "rejected" in result.output
    assert not gold.exists()


def test_commit_rejects_duplicate_example_id(tmp_path, snapshot, corpus_path):
    gold = tmp_path / "gold.jsonl"
    sheet = make_worksheet(tmp_path, snapshot)
    assert commit(sheet, gold, corpus_path).exit_code == 0
    second = commit(sheet, gold, corpus_path)
    assert second.exit_code == 1
    assert "already in" in second.output
    assert len(gold.read_text().splitlines()) == 1  # no double-append


def test_commit_rejects_citation_outside_frozen_pool(tmp_path, snapshot, corpus_path):
    gold = tmp_path / "gold.jsonl"
    claims = [
        {
            "text": "Clouds form in warm Neptunes.",
            "cited_bibcode": "2021MNRAS.500..003C",  # in corpus, NOT in this query's pool
            "support_span": "cloud models",
            "supported": True,
        }
    ]
    sheet = make_worksheet(tmp_path, snapshot, claims=claims)
    result = commit(sheet, gold, corpus_path)
    assert result.exit_code == 1
    assert "retrieved_context" in result.output
    assert not gold.exists()


# ── prepare: offline flag validation (context assembly needs the DB) ─────────


def prepare(*args: str):
    return runner.invoke(author.app, ["prepare", "--query", "q?", *args])


def test_prepare_negative_requires_calibration_partition():
    result = prepare("--family", "lit_qa", "--partition", "eval", "--negative",
                     "--negative-type", "wrong_paper")
    assert result.exit_code == 1
    assert "calibration" in result.output


def test_prepare_negative_requires_negative_type():
    result = prepare("--family", "lit_qa", "--partition", "calibration", "--negative")
    assert result.exit_code == 1
    assert "negative-type" in result.output


def test_prepare_abstention_cannot_be_negative():
    result = prepare("--family", "abstention", "--partition", "calibration", "--negative",
                     "--negative-type", "wrong_paper")
    assert result.exit_code == 1
    assert "abstention" in result.output
