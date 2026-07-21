"""Contract rules 1-6 in schema.py — the validator that gates every gold example."""

from __future__ import annotations

import pytest
from conftest import (
    IN_CORPUS_NOT_RETRIEVED,
    NOT_IN_CORPUS,
    abstention_example,
    claim,
    example,
    negative_example,
)
from schema import validate_record

CORPUS = {"2023A&A...100..001A", "2022ApJ...900..002B", "2021MNRAS.500..003C"}


def assert_valid(data: dict, corpus: set[str] | None = CORPUS) -> None:
    ex, errors, _ = validate_record(data, corpus)
    assert errors == [], errors
    assert ex is not None


def assert_rejected(data: dict, fragment: str, corpus: set[str] | None = CORPUS) -> None:
    ex, errors, _ = validate_record(data, corpus)
    assert any(fragment in e for e in errors), f"expected {fragment!r} in {errors}"


# ── Happy paths ──────────────────────────────────────────────────────────────


def test_valid_lit_qa_example_passes():
    assert_valid(example())


def test_valid_abstention_example_passes():
    assert_valid(abstention_example())


def test_valid_calibration_negative_passes():
    assert_valid(negative_example())


def test_summarization_behaves_like_lit_qa():
    assert_valid(example(task_family="summarization"))
    assert_rejected(
        example(task_family="summarization", claims=[claim(cited_bibcode=None)]),
        "must cite a bibcode",
    )


# ── Unfilled-template guards ─────────────────────────────────────────────────


def test_empty_answer_rejected():
    assert_rejected(example(answer="   "), "answer must not be empty")


def test_empty_claim_text_rejected():
    assert_rejected(example(claims=[claim(text=" ")]), "text must not be empty")


def test_no_claims_rejected_for_lit_qa():
    assert_rejected(example(claims=[]), "must cite a bibcode")


# ── Rule 1 + 2: abstention_reason ⟷ task_family ──────────────────────────────


def test_abstention_without_reason_rejected():
    assert_rejected(abstention_example(abstention_reason=None), "requires abstention_reason")


def test_non_abstention_with_reason_rejected():
    assert_rejected(example(abstention_reason="thin"), "only valid for the abstention")


def test_uncited_claim_rejected_outside_abstention():
    assert_rejected(example(claims=[claim(cited_bibcode=None)]), "must cite a bibcode")


# ── Rules 3-5: negatives ─────────────────────────────────────────────────────


def test_negative_outside_calibration_rejected():
    assert_rejected(
        negative_example(partition="eval", example_id="x-neg"), "calibration partition"
    )


def test_negative_without_type_rejected():
    assert_rejected(negative_example(negative_type=None), "requires a negative_type")


def test_negative_with_all_claims_supported_rejected():
    assert_rejected(
        negative_example(claims=[claim(supported=True)]), "at least one claim with supported=False"
    )


def test_non_negative_with_negative_type_rejected():
    assert_rejected(example(negative_type="contradicts"), "only valid when is_negative")


def test_unsupported_claim_on_non_negative_rejected():
    # Rule 4 is what keeps a mislabeled negative out of eval: supported=False ⟺ negative.
    assert_rejected(example(claims=[claim(supported=False)]), "all claims supported")


def test_abstention_negative_combinations_rejected():
    # author.py forbids abstention negatives at prepare time; the schema still rejects every
    # constructible form at commit time. In eval, rule 3's partition check fires first; in
    # calibration, the missing negative_type fires.
    assert_rejected(abstention_example(is_negative=True), "calibration partition")
    assert_rejected(
        abstention_example(is_negative=True, partition="calibration"), "requires a negative_type"
    )


# ── Rule 6 + citation grounding (needs corpus) ───────────────────────────────


def test_retrieved_bibcode_absent_from_corpus_rejected():
    bad = example(
        retrieved_context=example()["retrieved_context"]
        + [{"bibcode": NOT_IN_CORPUS, "chunk_id": "x", "retrieval_score": 0.1}]
    )
    assert_rejected(bad, "absent from corpus snapshot")


def test_cited_bibcode_absent_from_corpus_rejected():
    assert_rejected(example(claims=[claim(cited_bibcode=NOT_IN_CORPUS)]), "absent from corpus")


def test_citation_outside_retrieved_pool_rejected_for_positive():
    # In the corpus but not in THIS query's frozen pool → breaks the frozen-context guarantee.
    assert_rejected(
        example(claims=[claim(cited_bibcode=IN_CORPUS_NOT_RETRIEVED)]),
        "not in this query's retrieved_context",
    )


def test_citation_outside_pool_is_warning_for_wrong_paper_negative():
    data = negative_example(
        negative_type="wrong_paper",
        claims=[claim(cited_bibcode=IN_CORPUS_NOT_RETRIEVED, supported=False)],
    )
    ex, errors, warnings = validate_record(data, CORPUS)
    assert errors == [], errors
    assert any("not in this query's retrieved_context" in w for w in warnings)


def test_supported_claim_without_span_warns_but_passes():
    ex, errors, warnings = validate_record(example(claims=[claim(support_span=None)]), CORPUS)
    assert errors == []
    assert any("no support_span" in w for w in warnings)


def test_offline_validation_skips_corpus_rules():
    # corpus_bibcodes=None → rule 6 skipped; structural rules still apply.
    data = example(claims=[claim(cited_bibcode=NOT_IN_CORPUS)])
    ex, errors, warnings = validate_record(data, None)
    assert ex is not None and errors == [] and warnings == []
    assert_rejected(example(answer=""), "answer must not be empty", corpus=None)


def test_structural_failure_returns_pydantic_locations():
    ex, errors, _ = validate_record(example(task_family="not_a_family"), CORPUS)
    assert ex is None
    assert any("task_family" in e for e in errors)


@pytest.mark.parametrize("field", ["example_id", "query", "answer"])
def test_missing_required_field_rejected(field):
    data = example()
    del data[field]
    ex, errors, _ = validate_record(data, CORPUS)
    assert ex is None and errors
