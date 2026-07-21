"""Partition invariants: disjointness, composition accounting, manifest assembly."""

from __future__ import annotations

import json

from conftest import abstention_example, example, negative_example
from partition import (
    build_manifest,
    compose_counts,
    corpus_hash_consistency,
    disjointness_check,
    load_gold,
    target_progress,
)
from schema import GoldExample


def build(data: dict) -> GoldExample:
    return GoldExample.model_validate(data)


# ── load_gold ────────────────────────────────────────────────────────────────


def test_load_gold_missing_file_is_empty(tmp_path):
    examples, errors = load_gold(tmp_path / "none.jsonl")
    assert examples == [] and errors == []


def test_load_gold_reads_valid_and_names_bad_lines(tmp_path, snapshot):
    gold = tmp_path / "gold.jsonl"
    lines = [
        json.dumps(example()),
        "{not json",
        json.dumps(example(example_id="dup-check-2", answer="")),  # contract violation
    ]
    gold.write_text("\n".join(lines) + "\n")
    examples, errors = load_gold(gold, snapshot.bibcodes)
    assert len(examples) == 1
    assert any("line 2" in e and "invalid JSON" in e for e in errors)
    assert any("line 3" in e and "answer" in e for e in errors)


# ── Disjointness: no example calibrates the verifier AND scores the model ────


def test_disjointness_detects_shared_query():
    shared_q = "What did JWST measure for the C/O ratio of WASP-39b?"
    ex_eval = build(example(query=shared_q))
    ex_cal = build(
        example(
            example_id="lit_qa-calibration-same-q",
            partition="calibration",
            query=shared_q,
        )
    )
    result = disjointness_check([ex_eval, ex_cal])
    assert result["disjoint"] is False
    assert result["shared_queries"] == [shared_q]


def test_disjointness_passes_on_distinct_queries():
    result = disjointness_check(
        [build(example()), build(negative_example(query="A different query?"))]
    )
    assert result["disjoint"] is True
    assert result["n_shared_queries"] == 0


# ── compose_counts + target_progress ─────────────────────────────────────────


def test_compose_counts_partitions_families_and_negatives():
    examples = [
        build(example()),
        build(abstention_example(query="q2?")),
        build(negative_example(query="q3?")),
    ]
    counts = compose_counts(examples)
    assert counts["total"] == 3
    assert counts["by_partition"] == {"calibration": 1, "eval": 2}
    assert counts["by_family"] == {"lit_qa": 2, "summarization": 0, "abstention": 1}
    assert counts["positives_negatives"]["calibration"] == {"positive": 0, "negative": 1}
    assert counts["positives_negatives"]["eval"] == {"positive": 2, "negative": 0}
    assert counts["negative_coverage"]["claim_not_supported"] == 1
    assert counts["negative_coverage"]["all_modes_covered"] is False
    assert counts["abstention_coverage"]["absent"] == 1


def test_target_progress_flags_under_met_over():
    counts = compose_counts([build(example())])
    prog = target_progress(counts)
    assert prog["total"]["status"] == "under"  # 1 vs (170, 230)
    assert prog["eval"]["lit_qa"]["status"] == "under"
    # Saturate the calibration-negative band's soft max (40, 60) → "over".
    negatives = [
        build(negative_example(example_id=f"neg-{i}", query=f"nq{i}?")) for i in range(61)
    ]
    prog2 = target_progress(compose_counts(negatives))
    assert prog2["calibration"]["negative"]["status"] == "over"
    # An open-ended band (min, None) can only be under or met, never over.
    assert prog2["eval"]["lit_qa"]["target"][1] is None


# ── Corpus-hash consistency + manifest ───────────────────────────────────────


def test_corpus_hash_consistency_flags_mismatch(snapshot):
    good = build(example(provenance={"corpus_snapshot_hash": snapshot.snapshot_hash}))
    drifted = build(
        example(example_id="drifted", provenance={"corpus_snapshot_hash": "sha256:other"})
    )
    ok = corpus_hash_consistency([good], snapshot)
    assert ok["all_examples_match"] is True
    bad = corpus_hash_consistency([good, drifted], snapshot)
    assert bad["all_examples_match"] is False
    assert bad["mismatched_hashes"] == ["sha256:other"]


def test_build_manifest_carries_all_gates(snapshot):
    examples = [build(example(provenance={"corpus_snapshot_hash": snapshot.snapshot_hash}))]
    manifest = build_manifest(
        examples, snapshot, {"arm": "hybrid", "pool": 100}, source="gold.jsonl", valid=True
    )
    assert manifest["schema_version"] == "sft-gold-seed/1"
    assert manifest["corpus"]["snapshot_hash"] == snapshot.snapshot_hash
    assert manifest["corpus"]["n_abstracts"] == 3
    assert manifest["counts"]["total"] == 1
    assert manifest["disjointness"]["disjoint"] is True
    assert manifest["corpus_consistency"]["all_examples_match"] is True
    assert manifest["target_progress"]["total"]["status"] == "under"
