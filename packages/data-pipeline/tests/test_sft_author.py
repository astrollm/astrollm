"""The author CLI's commit path — the gate every hand-authored example passes through.

`prepare` needs the live retriever and a tokenizer, so here both are stubbed (fake retrieval over
the tiny corpus, whitespace token counter) to test its D1 delivery offline; `commit` is fully
offline (worksheet + corpus snapshot) and is tested end-to-end.
"""

from __future__ import annotations

import json
from pathlib import Path

import author
import yaml
from conftest import CORPUS_RECORDS, IN_POOL, example
from context import AssembledContext, render_context
from schema import Retrieved
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
            "delivered_k": data["delivered_k"],
            # worksheet-only delivery metadata — dropped at commit
            "context_budget_tokens": 16384,
            "context_tokens": 123,
            "context_tokenizer": "Qwen/Qwen3.5-4B",
            "dropped_bibcodes": [],
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
    assert record["delivered_k"] == 2  # read from the worksheet's _locked block
    for key in ("context_budget_tokens", "context_tokens", "context_tokenizer",
                "dropped_bibcodes"):
        assert key not in record  # worksheet audit metadata, not GoldExample fields
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


def test_commit_rejects_citation_below_delivered_k(tmp_path, snapshot, corpus_path):
    # IN_POOL[1] is rank 2 of the retrieved pool, but only the top-1 was delivered to the labeler.
    gold = tmp_path / "gold.jsonl"
    claims = [{"text": "An inversion was found.", "cited_bibcode": IN_POOL[1],
               "support_span": "thermal inversion", "supported": True}]
    sheet = make_worksheet(tmp_path, snapshot, delivered_k=1, claims=claims)
    result = commit(sheet, gold, corpus_path)
    assert result.exit_code == 1
    assert "delivered_k=1" in result.output
    assert not gold.exists()


def test_commit_rejects_worksheet_without_delivered_k(tmp_path, snapshot, corpus_path):
    gold = tmp_path / "gold.jsonl"
    sheet = make_worksheet(tmp_path, snapshot)
    raw = yaml.safe_load(sheet.read_text())
    del raw["_locked"]["delivered_k"]
    sheet.write_text(yaml.safe_dump(raw, sort_keys=False))
    result = commit(sheet, gold, corpus_path)
    assert result.exit_code == 1
    assert "delivered_k" in result.output
    assert not gold.exists()


# ── prepare: D1 delivery with the retriever + tokenizer stubbed ──────────────


POOL3 = tuple(r["bibcode"] for r in CORPUS_RECORDS)  # a 3-doc "pool", best first


def _stub_prepare(monkeypatch) -> None:
    def fake_assemble(query, snapshot, *, timestamp=None):
        return AssembledContext(
            query=query,
            retrieved=[Retrieved(bibcode=b, chunk_id=snapshot.chunk_id(b),
                                 retrieval_score=1.0 / (i + 1)) for i, b in enumerate(POOL3)],
            retriever_run_id="rrf-pool100-test",
            corpus_snapshot_hash=snapshot.snapshot_hash,
            retriever_config={"arm": "hybrid", "k": 100, "pool": 100},
            assembled_at="T",
        )

    monkeypatch.setattr(author, "assemble_context", fake_assemble)
    monkeypatch.setattr(author, "load_token_counter", lambda name: lambda s: len(s.split()))


def _prepare_sheet(tmp_path, corpus_path, *args: str):
    out = tmp_path / "ws.yaml"
    result = runner.invoke(author.app, [
        "prepare", "--query", "JWST WASP-39b?", "--family", "lit_qa", "--partition", "eval",
        "--corpus", str(corpus_path), "--out", str(out), *args,
    ])
    return result, out


def test_prepare_locks_full_pool_and_delivery_metadata(tmp_path, corpus_path, monkeypatch):
    _stub_prepare(monkeypatch)
    result, out = _prepare_sheet(tmp_path, corpus_path, "--render-limit", "3",
                                 "--budget", "1000")
    assert result.exit_code == 0, result.output
    locked = yaml.safe_load(out.read_text())["_locked"]
    assert [r["bibcode"] for r in locked["retrieved_context"]] == list(POOL3)  # full pool kept
    assert locked["delivered_k"] == 3 and locked["dropped_bibcodes"] == []
    assert locked["context_budget_tokens"] == 1000
    assert 0 < locked["context_tokens"] <= 1000
    assert locked["context_tokenizer"] == author.DEFAULT_TOKENIZER


def test_prepare_drops_whole_abstracts_to_fit_budget(tmp_path, corpus_path, snapshot,
                                                     monkeypatch):
    _stub_prepare(monkeypatch)
    # Budget = the exact top-2 rendering, so the rank-3 abstract must be dropped whole.
    assembled = author.assemble_context("JWST WASP-39b?", snapshot)
    top2 = len(render_context(assembled, snapshot, limit=2).split())
    result, out = _prepare_sheet(tmp_path, corpus_path, "--render-limit", "3",
                                 "--budget", str(top2))
    assert result.exit_code == 0, result.output
    text = out.read_text()
    locked = yaml.safe_load(text)["_locked"]
    assert locked["delivered_k"] == 2
    assert locked["dropped_bibcodes"] == [POOL3[2]]
    assert locked["context_tokens"] == top2
    assert len(locked["retrieved_context"]) == 3  # provenance keeps the whole pool
    banner = text.split("_locked:")[0]
    assert POOL3[2] not in banner and "dropped from the bottom" in banner


def test_prepare_dry_run_writes_nothing(tmp_path, corpus_path, monkeypatch):
    _stub_prepare(monkeypatch)
    result, out = _prepare_sheet(tmp_path, corpus_path, "--dry-run")
    assert result.exit_code == 0, result.output
    assert not out.exists()
    assert "lit_qa-eval-" in result.stdout


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
