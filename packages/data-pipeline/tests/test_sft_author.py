"""The author CLI's commit path — the gate every hand-authored example passes through.

`prepare` needs the live retriever and a tokenizer, so here both are stubbed (fake retrieval over
the tiny corpus, whitespace token counter) to test its D1 delivery offline; `commit` is fully
offline (worksheet + corpus snapshot) and is tested end-to-end.
"""

from __future__ import annotations

import json
from pathlib import Path

import author
import pytest
import yaml
from conftest import CORPUS_RECORDS, IN_POOL, example
from context import AssembledContext, render_context
from schema import Retrieved
from typer.testing import CliRunner

runner = CliRunner()


def make_worksheet(tmp_path: Path, snapshot, *, context_k: int | None = None,
                   dropped: list[str] | None = None, show: int | None = None,
                   **overrides) -> Path:
    """A filled worksheet as `prepare` would emit it and the labeler would complete it.

    The banner is the real ``render_context`` output composed by ``author._compose_worksheet``.
    Defaults are self-consistent (context_k = |pool|, dropped = ranks delivered_k+1..context_k,
    banner shows the top delivered_k); ``context_k`` / ``dropped`` / ``show`` tamper with them.
    """
    data = example(provenance={"author": "human", "corpus_snapshot_hash": snapshot.snapshot_hash})
    data.update(overrides)
    pool = [r["bibcode"] for r in data["retrieved_context"]]
    dk = data["delivered_k"]
    ck = len(pool) if context_k is None else context_k
    config = {"arm": "hybrid", "pool": 100}
    sheet = {
        "_locked": {
            "example_id": data["example_id"],
            "task_family": data["task_family"],
            "partition": data["partition"],
            "query": data["query"],
            "retrieved_context": data["retrieved_context"],
            "context_k": ck,
            "delivered_k": dk,
            # worksheet-only delivery metadata — verified, then dropped at commit
            "context_budget_tokens": 16384,
            "context_tokens": 123,
            "context_tokenizer": "Qwen/Qwen3.5-4B",
            "dropped_bibcodes": pool[dk:ck] if dropped is None else dropped,
            "provenance": data["provenance"],
            "retriever_config": config,  # dropped at commit
        },
        "answer": data["answer"],
        "claims": data["claims"],
        "abstention_reason": data["abstention_reason"],
        "is_negative": data["is_negative"],
        "negative_type": data["negative_type"],
    }
    assembled = AssembledContext(
        query=data["query"], retrieved=[Retrieved(**r) for r in data["retrieved_context"]],
        retriever_run_id="rrf-pool100-test", corpus_snapshot_hash=snapshot.snapshot_hash,
        retriever_config=config,
    )
    shown = render_context(assembled, snapshot, limit=dk if show is None else show)
    path = tmp_path / f"{data['example_id']}.yaml"
    path.write_text(author._compose_worksheet(sheet, shown, negative=data["is_negative"]))
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


@pytest.mark.parametrize("field", ["delivered_k", "context_k", "dropped_bibcodes"])
def test_commit_rejects_worksheet_without_delivery_field(tmp_path, snapshot, corpus_path, field):
    gold = tmp_path / "gold.jsonl"
    sheet = make_worksheet(tmp_path, snapshot)
    text = sheet.read_text()
    banner, body = text.split("_locked:", 1)
    raw = yaml.safe_load("_locked:" + body)
    del raw["_locked"][field]
    sheet.write_text(banner + yaml.safe_dump(raw, sort_keys=False))
    result = commit(sheet, gold, corpus_path)
    assert result.exit_code == 1
    assert f"_locked.{field} missing" in result.output
    assert not gold.exists()


# ── commit: the D1 delivery metadata is verified, not trusted ────────────────

CITE_RANK2 = [{"text": "An inversion was found.", "cited_bibcode": IN_POOL[1],
               "support_span": "thermal inversion", "supported": True}]


def _rejected(result, fragment: str, gold: Path) -> None:
    assert result.exit_code == 1, result.output
    assert fragment in result.output, result.output
    assert not gold.exists()


def test_commit_accepts_consistent_delivery_with_a_drop(tmp_path, snapshot, corpus_path):
    # Rank 2 was dropped by the budget; the answer cites rank 1, which was shown.
    gold = tmp_path / "gold.jsonl"
    sheet = make_worksheet(tmp_path, snapshot, delivered_k=1, context_k=2)
    result = commit(sheet, gold, corpus_path)
    assert result.exit_code == 0, result.output
    record = json.loads(gold.read_text())
    assert record["delivered_k"] == 1 and "context_k" not in record


def test_commit_rejects_inflated_delivered_k(tmp_path, snapshot, corpus_path):
    # Codex repro: banner shows rank 1 only, rank 2 listed as dropped, yet delivered_k=2 — an
    # answer citing rank 2 must not commit.
    gold = tmp_path / "gold.jsonl"
    sheet = make_worksheet(tmp_path, snapshot, delivered_k=2, context_k=2,
                           dropped=[IN_POOL[1]], show=1, claims=CITE_RANK2)
    _rejected(commit(sheet, gold, corpus_path), "delivery error", gold)


def test_commit_rejects_delivered_plus_dropped_not_context_k(tmp_path, snapshot, corpus_path):
    # (a) 1 delivered + 1 dropped, but the locked cut was k=1.
    gold = tmp_path / "gold.jsonl"
    sheet = make_worksheet(tmp_path, snapshot, delivered_k=1, context_k=1, dropped=[IN_POOL[1]])
    _rejected(commit(sheet, gold, corpus_path),
              "!= context_k = 1", gold)  # (a) names delivered_k + len(dropped_bibcodes)


def test_commit_rejects_dropped_list_not_the_pool_tail(tmp_path, snapshot, corpus_path):
    # (b) right length, wrong bibcode: dropped must be exactly ranks delivered_k+1..context_k.
    gold = tmp_path / "gold.jsonl"
    sheet = make_worksheet(tmp_path, snapshot, delivered_k=1, context_k=2,
                           dropped=["2021MNRAS.500..003C"])
    _rejected(commit(sheet, gold, corpus_path), "dropped_bibcodes:", gold)


@pytest.mark.parametrize("dk", [0, 3])
def test_commit_rejects_delivered_k_out_of_range(tmp_path, snapshot, corpus_path, dk):
    # (c) 1 <= delivered_k <= len(retrieved_context) (= 2 here).
    gold = tmp_path / "gold.jsonl"
    sheet = make_worksheet(tmp_path, snapshot, delivered_k=dk, context_k=2, dropped=[], show=1)
    _rejected(commit(sheet, gold, corpus_path), "delivered_k: must be", gold)


def test_commit_rejects_rendering_showing_a_dropped_abstract(tmp_path, snapshot, corpus_path):
    # (d) metadata self-consistent (1 delivered, rank 2 dropped) but the banner shows rank 2.
    gold = tmp_path / "gold.jsonl"
    sheet = make_worksheet(tmp_path, snapshot, delivered_k=1, context_k=2, show=2)
    _rejected(commit(sheet, gold, corpus_path), "visible rendering", gold)
    assert IN_POOL[1] in commit(sheet, gold, corpus_path).output  # names the dropped bibcode


def test_commit_rejects_rendering_missing_a_delivered_abstract(tmp_path, snapshot, corpus_path):
    # (d) delivered_k=2 with nothing dropped, but the banner shows only rank 1.
    gold = tmp_path / "gold.jsonl"
    sheet = make_worksheet(tmp_path, snapshot, delivered_k=2, context_k=2, show=1)
    _rejected(commit(sheet, gold, corpus_path), "visible rendering", gold)


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
    monkeypatch.setattr(author, "qwen_token_counter", lambda name: lambda s: len(s.split()))


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
    assert locked["context_k"] == 3
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
    assert locked["context_k"] == 3 and locked["delivered_k"] == 2
    assert locked["dropped_bibcodes"] == [POOL3[2]]
    assert author.verify_delivery(locked, text) == []  # prepare's output passes commit's check
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
