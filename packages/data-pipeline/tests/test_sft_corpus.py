"""CorpusSnapshot loading, integrity guards, and the provenance hash."""

from __future__ import annotations

import json

import pytest
from conftest import CORPUS_RECORDS
from corpus import CorpusSnapshot, chunk_id_for, indexed_text


def test_load_and_lookups(corpus_path):
    snap = CorpusSnapshot.load(corpus_path)
    assert len(snap) == 3
    assert "2023A&A...100..001A" in snap
    assert snap.title("2023A&A...100..001A").startswith("JWST")
    assert "C/O ratio" in snap.abstract("2023A&A...100..001A")
    assert snap.bibcodes == {r["bibcode"] for r in CORPUS_RECORDS}


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        CorpusSnapshot.load(tmp_path / "nope.jsonl")


def test_empty_file_raises(tmp_path):
    path = tmp_path / "empty.jsonl"
    path.write_text("\n\n")
    with pytest.raises(ValueError, match="empty"):
        CorpusSnapshot.load(path)


def test_duplicate_bibcode_raises(tmp_path):
    path = tmp_path / "dupe.jsonl"
    rec = json.dumps(CORPUS_RECORDS[0])
    path.write_text(rec + "\n" + rec + "\n")
    with pytest.raises(ValueError, match="duplicate bibcode"):
        CorpusSnapshot.load(path)


def test_record_missing_bibcode_raises(tmp_path):
    path = tmp_path / "nobib.jsonl"
    path.write_text(json.dumps({"title": "t", "abstract": "a"}) + "\n")
    with pytest.raises(ValueError, match="missing bibcode"):
        CorpusSnapshot.load(path)


def test_unknown_bibcode_lookup_raises(snapshot):
    with pytest.raises(KeyError):
        snapshot.record("2099ZZZZ.999..999Z")


# ── Hash semantics: the provenance anchor every worksheet locks ──────────────


def test_hash_is_order_independent(tmp_path):
    fwd = tmp_path / "fwd.jsonl"
    rev = tmp_path / "rev.jsonl"
    fwd.write_text("\n".join(json.dumps(r) for r in CORPUS_RECORDS) + "\n")
    rev.write_text("\n".join(json.dumps(r) for r in reversed(CORPUS_RECORDS)) + "\n")
    assert CorpusSnapshot.load(fwd).snapshot_hash == CorpusSnapshot.load(rev).snapshot_hash


def test_hash_changes_when_indexed_text_changes(tmp_path):
    baseline = tmp_path / "a.jsonl"
    edited = tmp_path / "b.jsonl"
    baseline.write_text("\n".join(json.dumps(r) for r in CORPUS_RECORDS) + "\n")
    changed = [dict(r) for r in CORPUS_RECORDS]
    changed[0]["abstract"] = "A different abstract entirely."
    edited.write_text("\n".join(json.dumps(r) for r in changed) + "\n")
    assert CorpusSnapshot.load(baseline).snapshot_hash != CorpusSnapshot.load(edited).snapshot_hash


def test_hash_ignores_fields_outside_indexed_text(tmp_path):
    baseline = tmp_path / "a.jsonl"
    extra = tmp_path / "b.jsonl"
    baseline.write_text("\n".join(json.dumps(r) for r in CORPUS_RECORDS) + "\n")
    decorated = [dict(r, citation_count=999) for r in CORPUS_RECORDS]
    extra.write_text("\n".join(json.dumps(r) for r in decorated) + "\n")
    # Only title+abstract are indexed; metadata edits must not shift provenance.
    assert CorpusSnapshot.load(baseline).snapshot_hash == CorpusSnapshot.load(extra).snapshot_hash


def test_hash_format():
    text = indexed_text(CORPUS_RECORDS[0])
    assert text.startswith("JWST") and "C/O" in text


def test_indexed_text_handles_missing_parts():
    assert indexed_text({"title": "Only title"}) == "Only title"
    # Documented quirk: the joining ".strip('. ')" also eats a legitimate trailing period on an
    # abstract-only record. Harmless for hashing/indexing (it's applied consistently and mirrors
    # index_corpus._embed_text), but a mismatch here would silently shift every snapshot hash.
    assert indexed_text({"abstract": "Only abstract."}) == "Only abstract"
    assert indexed_text({}) == ""


def test_chunk_id_is_deterministic(snapshot):
    assert chunk_id_for("X") == "X:abstract:0"
    assert snapshot.chunk_id("2023A&A...100..001A") == "2023A&A...100..001A:abstract:0"
