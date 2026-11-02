"""Offline tests for copilot.py turn bookkeeping: failure records, atomic /tag, transcript names,
and the shared (no-fallback) token counter.

No DB, no llama-server, no corpus file, no tokenizer download: the retriever is a fake over an
in-memory snapshot, the LLM endpoint is an ``httpx.MockTransport``, and the token counter is
injected (or ``qwen_token_counter`` is monkeypatched).
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import copilot
import httpx
import pytest

B1, B2, B3 = "2023Natur.614..649J", "2023Natur.614..664A", "2023ApJ...956L..13M"
TIMEOUT = object()  # sentinel: the mock LLM raises httpx.ReadTimeout for this reply


def _snapshot() -> copilot.CorpusSnapshot:
    records = [
        {"bibcode": b, "title": f"Title {i}", "abstract": f"Abstract {i} about CO2."}
        for i, b in enumerate((B1, B2, B3))
    ]
    return copilot.CorpusSnapshot(records, Path("in-memory.jsonl"))


class FakeRetriever:
    """Stands in for GroundedRetriever: fixed ranking, in-memory snapshot."""

    def __init__(self, bibcodes: tuple[str, ...] = (B1, B2)) -> None:
        self.snapshot = _snapshot()
        self.bibcodes = bibcodes

    def retrieve(self, query: str, k: int, pool: int) -> copilot.Grounding:
        assembled = copilot.AssembledContext(
            query=query,
            retrieved=[
                copilot.Retrieved(bibcode=b, chunk_id=f"{b}:abstract:0", retrieval_score=0.5)
                for b in self.bibcodes
            ],
            retriever_run_id="test-run",
            corpus_snapshot_hash=self.snapshot.snapshot_hash,
            retriever_config={"arm": "hybrid", "pool": pool, "k": k},
        )
        return copilot.Grounding(assembled=assembled, k=k)


def _fake_count(text: str) -> int:
    return len(text.split())


def _client(replies: list[Any], seen: list[str] | None = None) -> httpx.Client:
    queue = list(replies)

    def handler(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(request.url.path)
        if request.url.path == "/v1/chat/completions":
            reply = queue.pop(0)
            if reply is TIMEOUT:
                raise httpx.ReadTimeout("timed out", request=request)
            choice = {"message": {"content": reply}, "finish_reason": "stop"}
            return httpx.Response(200, json={"choices": [choice], "usage": {}})
        return httpx.Response(404)

    return httpx.Client(transport=httpx.MockTransport(handler))


def _session(
    tmp_path: Path,
    replies: list[Any],
    retriever: Any = None,
    *,
    count_tokens: Any = _fake_count,
    seen: list[str] | None = None,
) -> copilot.Session:
    settings = copilot.Settings(
        k=2, pool=100, budget=16384, max_tokens=64, temperature=0.2,
        base_url="http://llm.test/v1", model="test-model", live=False, dry_run=False,
        tag=None, corpus=Path("unused.jsonl"),
    )
    return copilot.Session(
        settings, retriever=retriever or FakeRetriever(), client=_client(replies, seen),
        transcript_dir=tmp_path, count_tokens=count_tokens,
    )


def _records(session: copilot.Session) -> list[dict[str, Any]]:
    return [json.loads(line) for line in session.path.read_text().splitlines()]


# --- Finding 6: every turn leaves a record, failures included -------------------------------


def test_empty_retrieval_writes_an_error_record(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(json.dumps({"bibcode": B1, "title": "T", "abstract": "A"}) + "\n")
    retriever = copilot.GroundedRetriever(corpus)
    retriever._pr = SimpleNamespace(retrieve=lambda query, arm, k, pool: [])  # type: ignore[assignment]
    session = _session(tmp_path, [], retriever=retriever)
    with pytest.raises(RuntimeError, match="no results"):
        session.run_turn("WASP-39b CO2")
    (record,) = _records(session)
    assert record["error"]["type"] == "RuntimeError"
    assert record["query"] == "WASP-39b CO2"
    assert record["turn"] == 1
    assert "answer" not in record


def test_first_call_timeout_writes_an_error_record_with_allowlist(tmp_path: Path) -> None:
    session = _session(tmp_path, [TIMEOUT])
    with pytest.raises(httpx.ReadTimeout):
        session.run_turn("WASP-39b CO2")
    (record,) = _records(session)
    assert record["error"]["type"] == "ReadTimeout"
    assert record["allowlist"] == [B1, B2]
    assert "answer" not in record


def test_regeneration_timeout_keeps_first_answer_flagged(tmp_path: Path) -> None:
    first = f"CO2 was detected [{B3}]."  # B3 is outside the top-2 allowlist
    session = _session(tmp_path, [first, TIMEOUT])
    record = session.run_turn("WASP-39b CO2")
    assert record is not None
    assert _records(session) == [record]
    assert record["answer"] == record["first_answer"] == first
    assert record["guard"]["status"] == "flagged"
    assert record["guard"]["violations"] == [B3]
    assert record["guard"]["first_violations"] == [B3]
    assert record["guard"]["regen_failed"] is True
    assert record["guard"]["regenerated"] is False
    assert record["error"]["type"] == "ReadTimeout"


def test_successful_regeneration_is_pass_after_regen(tmp_path: Path) -> None:
    session = _session(tmp_path, [f"CO2 [{B3}].", f"CO2 [{B1}]."])
    record = session.run_turn("WASP-39b CO2")
    assert record is not None
    assert record["guard"]["status"] == "pass_after_regen"
    assert record["guard"]["first_violations"] == [B3]
    assert record["answer"] == f"CO2 [{B1}]."
    assert "error" not in record


# --- Finding 7: /tag rewrites the transcript atomically --------------------------------------


def _transcript(tmp_path: Path) -> Path:
    path = tmp_path / "t.jsonl"
    path.write_text(json.dumps({"turn": 1, "tag": None}) + "\n" + json.dumps({"turn": 2}) + "\n")
    return path


def test_retag_uses_temp_file_in_same_dir_and_os_replace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _transcript(tmp_path)
    calls: list[tuple[str, str]] = []
    real_replace = copilot.os.replace

    def spy(src: str, dst: str) -> None:
        calls.append((str(src), str(dst)))
        real_replace(src, dst)

    monkeypatch.setattr(copilot.os, "replace", spy)
    assert copilot.retag_last_turn(path, "citation")
    (src, dst) = calls[0]
    assert Path(src).parent == path.parent and src != dst and Path(dst) == path
    lines = [json.loads(x) for x in path.read_text().splitlines()]
    assert lines == [{"turn": 1, "tag": None}, {"turn": 2, "tag": "citation"}]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["t.jsonl"]


def test_retag_failure_leaves_original_intact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _transcript(tmp_path)
    before = path.read_text()

    def boom(src: str, dst: str) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(copilot.os, "replace", boom)
    with pytest.raises(OSError):
        copilot.retag_last_turn(path, "citation")
    assert path.read_text() == before
    assert sorted(p.name for p in tmp_path.iterdir()) == ["t.jsonl"]


# --- Finding 8: transcript filenames carry the full session id -------------------------------


def test_transcript_path_uses_full_session_id() -> None:
    a, b = "abcdef01" + "0" * 24, "abcdef01" + "1" * 24  # same 8-char prefix
    pa = copilot.transcript_path(Path("d"), a, "2026-11-02")
    pb = copilot.transcript_path(Path("d"), b, "2026-11-02")
    assert pa != pb
    assert pa.name == f"2026-11-02-{a}.jsonl"


def test_session_path_contains_full_id(tmp_path: Path) -> None:
    session = _session(tmp_path, [])
    assert session.path.parent == tmp_path
    assert session.path.name.endswith(f"-{session.id}.jsonl")
    assert len(session.id) == 32


# --- Finding 9: one shared Qwen counter, no /tokenize, no //4 fallback -----------------------


def test_turn_records_tokenizer_and_never_calls_tokenize(tmp_path: Path) -> None:
    seen: list[str] = []
    session = _session(tmp_path, [f"CO2 [{B1}]."], seen=seen)
    record = session.run_turn("WASP-39b CO2")
    assert record is not None
    assert record["tokenizer"] == copilot.QWEN_TOKENIZER == "Qwen/Qwen3.5-4B"
    assert "token_counter" not in record
    assert seen == ["/v1/chat/completions"]  # no /tokenize request
    assert record["context_tokens"] == _fake_count(
        copilot.render_context(FakeRetriever().retrieve("WASP-39b CO2", 2, 100).assembled,
                               _snapshot(), limit=2)
    )


def test_tokenizer_load_failure_fails_the_turn_loudly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def no_tokenizer(name: str) -> Any:
        raise OSError(f"cannot load {name}")

    monkeypatch.setattr(copilot, "qwen_token_counter", no_tokenizer)
    seen: list[str] = []
    session = _session(tmp_path, [f"CO2 [{B1}]."], count_tokens=None, seen=seen)
    with pytest.raises(OSError, match="cannot load Qwen/Qwen3.5-4B"):
        session.run_turn("WASP-39b CO2")
    (record,) = _records(session)
    assert record["error"]["type"] == "OSError"
    assert "context_tokens" not in record  # nothing was counted by any fallback
    assert seen == []  # and the LLM was never called


def test_shared_counter_is_loaded_once_per_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    loads: list[str] = []

    def factory(name: str) -> Any:
        loads.append(name)
        return _fake_count

    monkeypatch.setattr(copilot, "qwen_token_counter", factory)
    session = _session(tmp_path, [f"A [{B1}].", f"B [{B2}]."], count_tokens=None)
    session.run_turn("first")
    session.run_turn("second")
    assert loads == ["Qwen/Qwen3.5-4B"]


@pytest.mark.parametrize("command", ["ask", "chat"])
def test_cli_exposes_tokenizer_option(command: str) -> None:
    # Inspect the registered click parameters rather than rendered --help text: rich wraps the
    # help box to the detected terminal width, which differs between a developer shell and CI.
    import typer.main

    cmd = typer.main.get_command(copilot.app).commands[command]
    param = next((p for p in cmd.params if "--tokenizer" in getattr(p, "opts", [])), None)
    assert param is not None, f"{command} has no --tokenizer option"
    assert param.default == "Qwen/Qwen3.5-4B"
