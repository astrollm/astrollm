"""Partition accounting, query-level disjointness, and manifest construction.

Pure (no DB, no rag import): it operates on already-validated `GoldExample`s plus a corpus snapshot
and a retriever-config dict handed in by the caller. The two invariants it enforces for the gold
seed are the contract's reason for existing:

- **No example both calibrates the verifier and scores the model.** The calibration and eval
  partitions must be disjoint at the *query* level (`disjointness_check`).
- **No negative ever reaches eval/training.** Already enforced per-example by the schema (rule 5);
  the counts here make it auditable in aggregate.

The composition targets from the pre-registration are tracked so the labeler can author *toward*
them; they are reported, not enforced (the seed is filled incrementally).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from corpus import CorpusSnapshot
from schema import (
    AbstentionReason,
    GoldExample,
    NegativeType,
    Partition,
    TaskFamily,
    validate_record,
)

# Composition targets (docs/research/sft-pilot.md). Bands are (min, soft_max); None == open.
TARGETS: dict[str, Any] = {
    "calibration": {"positive": (40, None), "negative": (40, 60)},
    "eval": {"lit_qa": (40, None), "summarization": (30, None), "abstention": (25, None)},
    "total": (170, 230),
}


def load_gold(
    path: Path | str, corpus_bibcodes: set[str] | None = None
) -> tuple[list[GoldExample], list[str]]:
    """Read + validate a gold JSONL. Returns ``(examples, errors)``; errors name the bad line."""
    path = Path(path)
    if not path.exists():
        return [], []
    examples, errors = [], []
    for lineno, raw in enumerate(path.read_text().splitlines(), 1):
        raw = raw.strip()
        if not raw:
            continue
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            errors.append(f"line {lineno}: invalid JSON ({exc})")
            continue
        example, errs, _ = validate_record(data, corpus_bibcodes)
        if errs:
            eid = data.get("example_id", "?")
            errors.extend(f"line {lineno} ({eid}): {e}" for e in errs)
        elif example is not None:
            examples.append(example)
    return examples, errors


def disjointness_check(examples: list[GoldExample]) -> dict[str, Any]:
    """Verify the calibration and eval partitions share no query (the contract's split rule)."""
    cal = {e.query for e in examples if e.partition is Partition.CALIBRATION}
    ev = {e.query for e in examples if e.partition is Partition.EVAL}
    shared = sorted(cal & ev)
    return {
        "disjoint": not shared,
        "n_calibration_queries": len(cal),
        "n_eval_queries": len(ev),
        "n_shared_queries": len(shared),
        "shared_queries": shared,
    }


def compose_counts(examples: list[GoldExample]) -> dict[str, Any]:
    """Per-family, per-partition, positive/negative, and failure-mode coverage counts."""
    by_partition = {p.value: 0 for p in Partition}
    by_family = {f.value: 0 for f in TaskFamily}
    by_partition_family: dict[str, dict[str, int]] = {
        p.value: {f.value: 0 for f in TaskFamily} for p in Partition
    }
    pos_neg = {p.value: {"positive": 0, "negative": 0} for p in Partition}
    negative_modes = {n.value: 0 for n in NegativeType}
    abstention_reasons = {a.value: 0 for a in AbstentionReason}

    for e in examples:
        by_partition[e.partition.value] += 1
        by_family[e.task_family.value] += 1
        by_partition_family[e.partition.value][e.task_family.value] += 1
        pos_neg[e.partition.value]["negative" if e.is_negative else "positive"] += 1
        if e.is_negative and e.negative_type is not None:
            negative_modes[e.negative_type.value] += 1
        if e.task_family is TaskFamily.ABSTENTION and e.abstention_reason is not None:
            abstention_reasons[e.abstention_reason.value] += 1

    modes_covered = sum(1 for v in negative_modes.values() if v > 0)
    return {
        "total": len(examples),
        "by_partition": by_partition,
        "by_family": by_family,
        "by_partition_family": by_partition_family,
        "positives_negatives": pos_neg,
        "negative_coverage": {
            **negative_modes,
            "modes_covered": modes_covered,
            "all_modes_covered": modes_covered == len(NegativeType),
        },
        "abstention_coverage": abstention_reasons,
    }


def target_progress(counts: dict[str, Any]) -> dict[str, Any]:
    """Current counts vs the pre-registered composition targets, with a met/under flag each."""

    def status(current: int, band: tuple[int, int | None]) -> dict[str, Any]:
        lo, hi = band
        if current < lo:
            state = "under"
        elif hi is not None and current > hi:
            state = "over"
        else:
            state = "met"
        return {"current": current, "target": list(band), "status": state}

    cal_pn = counts["positives_negatives"]["calibration"]
    eval_fam = counts["by_partition_family"]["eval"]
    return {
        "calibration": {
            "positive": status(cal_pn["positive"], TARGETS["calibration"]["positive"]),
            "negative": status(cal_pn["negative"], TARGETS["calibration"]["negative"]),
        },
        "eval": {
            fam: status(eval_fam[fam], TARGETS["eval"][fam]) for fam in TARGETS["eval"]
        },
        "total": status(counts["total"], TARGETS["total"]),
    }


def corpus_hash_consistency(
    examples: list[GoldExample], snapshot: CorpusSnapshot
) -> dict[str, Any]:
    """Every example must be grounded on the loaded corpus snapshot (one frozen corpus)."""
    hashes = {
        e.provenance.get("corpus_snapshot_hash") for e in examples if isinstance(e.provenance, dict)
    }
    mismatched = sorted(h for h in hashes if h != snapshot.snapshot_hash)
    return {
        "corpus_snapshot_hash": snapshot.snapshot_hash,
        "all_examples_match": not mismatched,
        "mismatched_hashes": mismatched,
    }


def build_manifest(
    examples: list[GoldExample],
    snapshot: CorpusSnapshot,
    retriever_config: dict[str, Any],
    *,
    source: str,
    valid: bool,
) -> dict[str, Any]:
    """Assemble the full manifest the build commit reports gold-seed composition through."""
    counts = compose_counts(examples)
    return {
        "schema_version": "sft-gold-seed/1",
        "generated_from": source,
        "valid": valid,
        "corpus": {
            "path": str(snapshot.path),
            "snapshot_hash": snapshot.snapshot_hash,
            "n_abstracts": len(snapshot),
        },
        "retriever_config": retriever_config,
        "counts": counts,
        "disjointness": disjointness_check(examples),
        "corpus_consistency": corpus_hash_consistency(examples, snapshot),
        "target_progress": target_progress(counts),
    }
