"""Offline tests for the D4 parser and the citation guard (``citation.py``, pure module).

Contract under test: ``docs/V1_PLAN_R2.md`` decision D4 — an abstention is ``ABSTAIN: <reason>``
plus at most one sentence of at most 40 words (anything more is *mixed*, scored as non-abstaining);
non-abstaining answers carry inline bibcodes and may end with one bibcode-only ``Sources:`` line.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from citation import (
    ABSTAIN_REASONS,
    BIBCODE_RE,
    check_citations,
    extract_bibcodes,
    parse_answer,
    strip_think,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
PILOT_QUERIES = (
    REPO_ROOT / "packages" / "evaluation" / "queries" / "pilot_exoplanet_atmospheres.yaml"
)

NATURE = "2023Natur.614..670F"
AANDA = "2019A&A...621A..68P"
ARXIV = "2023arXiv230512345S"
B1, B2, B3 = "2023Natur.614..649J", "2023Natur.614..664A", "2023ApJ...956L..13M"


# --- BIBCODE_RE ------------------------------------------------------------------------------


@pytest.mark.parametrize("bibcode", [NATURE, AANDA, ARXIV])
def test_regex_accepts_real_bibcode_shapes(bibcode: str) -> None:
    assert BIBCODE_RE.fullmatch(bibcode)
    assert extract_bibcodes(f"As shown by {bibcode}, the line is real.") == [bibcode]


@pytest.mark.parametrize(
    "near_miss",
    [
        "2023Natur.614..67F",  # 18 chars
        "2023Natur.614..670",  # 18 chars, no author initial
        "2023Natur.614..670FF",  # 20 chars
        "2023Natur.614..6700F",  # 20 chars
    ],
)
def test_regex_rejects_18_and_20_char_near_misses(near_miss: str) -> None:
    assert extract_bibcodes(near_miss) == []


@pytest.mark.parametrize(
    "glued",
    [f"x{NATURE}", f"{NATURE}x", f"{NATURE}7", f"7{NATURE}", f"Ref{AANDA}", f"{ARXIV}z"],
)
def test_regex_rejects_bibcodes_glued_to_letters_or_digits(glued: str) -> None:
    assert extract_bibcodes(glued) == []


def test_regex_accepts_bibcode_in_punctuation() -> None:
    assert extract_bibcodes(f"({NATURE}), [{AANDA}]; {ARXIV}!") == [NATURE, AANDA, ARXIV]


# --- extract_bibcodes ------------------------------------------------------------------------


def test_extract_bibcodes_preserves_first_seen_order() -> None:
    assert extract_bibcodes(f"{B2} then {B1} then {B3}") == [B2, B1, B3]


def test_extract_bibcodes_deduplicates() -> None:
    assert extract_bibcodes(f"{B1} {B2} {B1} {B2} {B1}") == [B1, B2]


def test_extract_bibcodes_empty_when_none() -> None:
    assert extract_bibcodes("No citations here.") == []


# --- parse_answer: abstentions ---------------------------------------------------------------


@pytest.mark.parametrize("reason", ABSTAIN_REASONS)
def test_pure_abstention_for_each_reason(reason: str) -> None:
    parsed = parse_answer(f"ABSTAIN: {reason}. The retrieved abstracts do not cover this.")
    assert parsed.is_abstain
    assert parsed.reason == reason
    assert not parsed.mixed
    assert parsed.pure_abstention
    assert parsed.abstain_sentences == 1


def test_abstention_marker_alone_is_pure() -> None:
    parsed = parse_answer("  ABSTAIN: thin\n")
    assert parsed.pure_abstention
    assert parsed.abstain_sentences == 0
    assert parsed.abstain_words == 0


def test_abstention_with_second_sentence_is_mixed() -> None:
    parsed = parse_answer(
        "ABSTAIN: thin. The abstracts mention WASP-39b only briefly. They report no value."
    )
    assert parsed.is_abstain
    assert parsed.mixed
    assert not parsed.pure_abstention
    assert parsed.abstain_sentences == 2


def test_abstention_over_40_words_in_one_sentence_is_mixed() -> None:
    explanation = " ".join(["word"] * 41)
    parsed = parse_answer(f"ABSTAIN: absent. {explanation}.")
    assert parsed.abstain_sentences == 1
    assert parsed.abstain_words == 41
    assert parsed.mixed
    assert not parsed.pure_abstention


def test_abstention_of_exactly_40_words_is_pure() -> None:
    explanation = " ".join(["word"] * 40)
    parsed = parse_answer(f"ABSTAIN: absent. {explanation}.")
    assert parsed.abstain_words == 40
    assert parsed.pure_abstention


@pytest.mark.parametrize(
    "variant",
    [
        "abstain: thin. Not enough evidence.",  # lowercase marker
        "Abstain: thin. Not enough evidence.",  # title case
        "ABSTAIN: Thin. Not enough evidence.",  # reason is case-sensitive
        "ABSTAIN : thin. Not enough evidence.",  # space before the colon
        "ABSTAIN: unknown. Not enough evidence.",  # not one of the four reasons
        "I must ABSTAIN: thin. Not enough evidence.",  # marker is not the first text
    ],
)
def test_marker_variants_are_not_abstentions(variant: str) -> None:
    parsed = parse_answer(variant)
    assert not parsed.is_abstain
    assert not parsed.pure_abstention
    assert parsed.reason is None


def test_malformed_marker_is_flagged_but_not_an_abstention() -> None:
    parsed = parse_answer("ABSTAIN : thin. Not enough evidence.")
    assert parsed.abstain_marker_malformed
    assert not parsed.is_abstain


# --- parse_answer: non-abstaining answers ----------------------------------------------------


def test_inline_citations_and_trailing_sources_line() -> None:
    text = (
        f"CO2 was detected in WASP-39b ({B1}). Sulfur dioxide followed {B2}.\n\n"
        f"Sources: {B1}, {B2}, {B3}\n"
    )
    parsed = parse_answer(text)
    assert not parsed.is_abstain
    assert parsed.sources_line == f"Sources: {B1}, {B2}, {B3}"
    assert parsed.sources_bibcodes == [B1, B2, B3]
    assert not parsed.sources_line_malformed
    assert parsed.inline_bibcodes == [B1, B2]
    assert parsed.all_bibcodes == [B1, B2, B3]  # inline first, then sources, de-duplicated
    assert "Sources:" not in parsed.body
    assert parsed.body.endswith(f"{B2}.")


def test_sources_line_with_prose_is_malformed_and_stays_in_body() -> None:
    line = f"Sources: see the JWST papers {B1}"
    parsed = parse_answer(f"CO2 was detected ({B2}).\n{line}")
    assert parsed.sources_line == line
    assert parsed.sources_line_malformed
    assert line in parsed.body


def test_sources_line_without_bibcodes_is_malformed() -> None:
    parsed = parse_answer(f"CO2 was detected ({B1}).\nSources: none")
    assert parsed.sources_line_malformed
    assert parsed.sources_bibcodes == []


def test_answer_without_sources_line() -> None:
    parsed = parse_answer(f"CO2 was detected ({B1}).")
    assert parsed.sources_line is None
    assert not parsed.sources_line_malformed
    assert parsed.all_bibcodes == [B1]


# --- check_citations -------------------------------------------------------------------------


def test_check_passes_when_every_citation_is_allowed() -> None:
    parsed = parse_answer(f"CO2 was detected ({B1}); see also {B2}.")
    result = check_citations(parsed, {B1, B2, B3})
    assert result.status == "pass"
    assert result.violations == []
    assert result.uncited_ok
    assert not result.abstain_with_citation


def test_check_flags_bibcode_outside_allowlist() -> None:
    parsed = parse_answer(f"CO2 was detected ({B1}); a later claim cites {B3}.\nSources: {B2}")
    result = check_citations(parsed, {B1, B2})
    assert result.status == "violation"
    assert result.violations == [B3]
    assert result.uncited_ok


def test_check_flags_sources_line_bibcode_outside_allowlist() -> None:
    parsed = parse_answer(f"CO2 was detected ({B1}).\nSources: {B1}, {B3}")
    result = check_citations(parsed, {B1})
    assert result.status == "violation"
    assert result.violations == [B3]


def test_non_abstaining_answer_with_no_citations_is_a_violation() -> None:
    parsed = parse_answer("CO2 was detected in the atmosphere of WASP-39b.")
    result = check_citations(parsed, {B1})
    assert result.status == "violation"
    assert result.violations == []
    assert not result.uncited_ok


def test_mixed_abstention_without_citations_is_a_violation() -> None:
    parsed = parse_answer("ABSTAIN: thin. Only brief mentions. Then a long claim follows.")
    assert parsed.mixed
    result = check_citations(parsed, {B1})
    assert result.status == "violation"
    assert not result.uncited_ok


def test_pure_abstention_with_zero_citations_passes() -> None:
    parsed = parse_answer("ABSTAIN: absent. The corpus does not cover cosmology.")
    result = check_citations(parsed, {B1})
    assert result.status == "pass"
    assert result.uncited_ok
    assert not result.abstain_with_citation


def test_abstention_citing_allowed_bibcode_is_recorded_not_a_violation() -> None:
    parsed = parse_answer(f"ABSTAIN: thin. Only {B1} touches on this.")
    assert parsed.pure_abstention
    result = check_citations(parsed, {B1})
    assert result.status == "pass"
    assert result.abstain_with_citation


def test_abstention_citing_disallowed_bibcode_is_a_violation() -> None:
    parsed = parse_answer(f"ABSTAIN: thin. Only {B3} touches on this.")
    result = check_citations(parsed, {B1})
    assert result.status == "violation"
    assert result.violations == [B3]
    assert result.abstain_with_citation


def test_guard_result_to_dict_has_expected_fields() -> None:
    result = check_citations(parse_answer(f"Claim ({B1})."), {B1})
    assert set(result.to_dict()) == {
        "status",
        "violations",
        "uncited_ok",
        "abstain_with_citation",
    }


# --- strip_think -----------------------------------------------------------------------------


def test_strip_think_removes_closed_block_and_reports_it() -> None:
    clean, had = strip_think(f"<think>\nreasoning about {B1}\n</think>\n\nThe answer ({B2}).")
    assert had
    assert clean == f"The answer ({B2})."
    assert "think" not in clean


def test_strip_think_leaves_clean_text_untouched() -> None:
    text = f"  The answer ({B1}).  \n"
    clean, had = strip_think(text)
    assert not had
    assert clean == text


def test_strip_think_handles_orphan_close_and_unclosed_open() -> None:
    assert strip_think("reasoning</think>Answer.") == ("Answer.", True)
    assert strip_think("Answer.<think>truncated reasoning") == ("Answer.", True)


# --- data: the pilot query set must be citable by the guard ----------------------------------


def test_pilot_expected_bibcodes_all_match_regex_fully() -> None:
    queries = yaml.safe_load(PILOT_QUERIES.read_text(encoding="utf-8"))["queries"]
    bibcodes = [b for q in queries for b in q.get("expected_bibcodes", [])]
    assert bibcodes, "no expected_bibcodes found — wrong file?"
    bad = [b for b in bibcodes if not BIBCODE_RE.fullmatch(b)]
    assert bad == []
