"""Offline tests for the M1 exit-review tooling: the 25-query YAML and the pure sheet renderer."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import pytest
import yaml
from copilot import Tag
from review_sheet import (
    MARKS,
    render_context_doc,
    render_sheet,
    render_summary,
    select_pending,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
QUERIES_DIR = REPO_ROOT / "packages" / "evaluation" / "queries"
REVIEW = yaml.safe_load((QUERIES_DIR / "m1_review_queries.yaml").read_text())["queries"]
GOLD = {
    q["id"]: q
    for q in yaml.safe_load((QUERIES_DIR / "pilot_exoplanet_atmospheres.yaml").read_text())[
        "queries"
    ]
}

B1, B2, B3 = "2023Natur.614..649J", "2023Natur.614..664A", "2099ApJ...123L..45X"


# --- the review query set ---------------------------------------------------------------------


def test_review_yaml_has_25_unique_ids_in_order() -> None:
    assert [q["id"] for q in REVIEW] == [f"r{i:02d}" for i in range(1, 26)]


def test_review_yaml_class_counts() -> None:
    assert Counter(q["class"] for q in REVIEW) == {
        "known_item": 10,
        "broad": 7,
        "abstain_off_topic": 1,
        "abstain_absent": 1,
        "abstain_thin": 1,
        "contradictory": 1,
        "adversarial_false_premise": 1,
        "adversarial_follow_up": 1,
        "adversarial_alias": 1,
        "adversarial_timeline": 1,
    }


def test_review_yaml_fields_are_non_empty_and_tags_are_taxonomy_tags() -> None:
    tags = {t.value for t in Tag}
    for q in REVIEW:
        assert set(q) == {"id", "class", "query", "expected", "tag"}, q["id"]
        assert all(isinstance(v, str) and v.strip() for v in q.values()), q["id"]
        assert q["tag"] in tags, q["id"]


def test_gold_derived_queries_are_verbatim() -> None:
    gold_ids = [f"q{i:02d}" for i in (*range(1, 11), *range(24, 31))]
    for q, gid in zip(REVIEW[:17], gold_ids, strict=True):
        assert q["query"] == GOLD[gid]["query"], q["id"]
        assert f"gold {gid}" in q["expected"], q["id"]


def test_follow_up_runs_right_after_false_premise() -> None:
    ids = [q["id"] for q in REVIEW]
    assert ids.index("r23") == ids.index("r22") + 1


# --- select_pending ---------------------------------------------------------------------------


def test_select_pending_skips_done_and_honours_start_and_limit() -> None:
    done = {"r01": {}, "r03": {}}
    assert [q["id"] for q in select_pending(REVIEW, done, limit=3)] == ["r02", "r04", "r05"]
    assert [q["id"] for q in select_pending(REVIEW, done, start="r24")] == ["r24", "r25"]
    with pytest.raises(ValueError, match="unknown review id"):
        select_pending(REVIEW, done, start="r99")


# --- the renderer -----------------------------------------------------------------------------


def _record(**overrides: Any) -> dict[str, Any]:
    record: dict[str, Any] = {
        "session_id": "a" * 32,
        "turn": 1,
        "ts": "2026-11-02T05:00:00+00:00",
        "mode": "grounded",
        "model": "qwen3.5-9b",
        "k": 30,
        "query": REVIEW[0]["query"],
        "allowlist": [B2, B1],
        "delivered_n": 2,
        "context_tokens": 1234,
        "answer": f"CO2 was detected [{B1}] and also [{B3}].\nSources: {B1}, {B3}",
        "first_answer": None,
        "parsed": {
            "is_abstain": False,
            "pure_abstention": False,
            "reason": None,
            "mixed": False,
            "marker_misplaced": True,
            "sources_line_malformed": False,
            "all_bibcodes": [B1, B3],
        },
        "guard": {"status": "flagged", "violations": [B3], "first_violations": [B3]},
        "had_think": False,
        "tag": "citation",
        "latency_ms": 4200,
    }
    record.update(overrides)
    return record


def _titles(bibcode: str) -> str | None:
    return {B1: "Identification of carbon dioxide | in an exoplanet atmosphere"}.get(bibcode)


def test_summary_row_and_not_run_rows() -> None:
    rows = render_summary(REVIEW[:2], {"r01": _record()})
    assert rows[2] == (
        "| r01 | known_item | flagged | answer | marker_misplaced | 2 | 1234 |  | citation |"
    )
    assert rows[3] == "| r02 | known_item | not run | — | — | — | — |  | citation |"


def test_turn_section_has_cited_bibcode_table_with_blank_mark_column() -> None:
    sheet = render_sheet(REVIEW[:2], {"r01": {"session_id": "a" * 32, "turn": 1}},
                         {"r01": _record()}, _titles)
    assert "| # | bibcode | title | in context | mark |" in sheet
    # in-context bibcode: rank from the allowlist, pipe in the title escaped, blank mark cell
    assert (
        f"| 1 | `{B1}` | Identification of carbon dioxide \\| in an exoplanet atmosphere "
        "| rank 2 |  |"
    ) in sheet
    assert f"| 2 | `{B3}` | (not in corpus) | NOT IN CONTEXT |  |" in sheet
    assert f"Marks: {' / '.join(MARKS)}" in sheet
    for line in ("- Claims supported: __ / __", "- Verdict: good | minor | bad",
                 "- Tag: citation", "- Note:"):
        assert line in sheet
    assert "`data/eval/review/context/r01.md`" in sheet
    assert "## r02 — known_item" in sheet and "*Not run yet.*" in sheet


def test_answer_is_verbatim_in_a_fence_that_backticks_cannot_close() -> None:
    answer = f"Uses ``` inside [{B1}]."
    sheet = render_sheet(REVIEW[:1], {}, {"r01": _record(answer=answer)}, _titles)
    assert f"````text\n{answer}\n````" in sheet


def test_rejected_first_answer_is_shown_when_regenerated() -> None:
    record = _record(first_answer=f"Off-list [{B3}].", answer=f"Fixed [{B1}].",
                     guard={"status": "pass_after_regen", "first_violations": [B3]})
    sheet = render_sheet(REVIEW[:1], {}, {"r01": record}, _titles)
    assert f"**First answer** (rejected by the guard; off-list: {B3}):" in sheet
    assert "| r01 | known_item | pass_after_regen |" in sheet


def test_error_record_renders_without_answer() -> None:
    record = {"query": "q", "error": {"type": "ReadTimeout", "message": "timed out"},
              "tag": "citation", "allowlist": [B1], "delivered_n": 30}
    sheet = render_sheet(REVIEW[:1], {}, {"r01": record}, _titles)
    assert "| r01 | known_item | error: ReadTimeout |" in sheet
    assert "**Error:** `ReadTimeout`: timed out" in sheet
    assert "**Answer**" not in sheet


def test_sheet_never_contains_abstract_text_but_context_doc_does() -> None:
    abstract = "We report a 26-sigma CO2 feature at 4.3 microns."
    sheet = render_sheet(REVIEW[:1], {}, {"r01": _record()}, _titles)
    assert abstract not in sheet
    doc = render_context_doc("r01", "q", f"[  1] {B1}\n      title\n      {abstract}", "drift")
    assert abstract in doc and "**WARNING:** drift" in doc
