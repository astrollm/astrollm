"""Offline tests for the D1 context-cut logic (fake renderer + fake token counter)."""

from __future__ import annotations

import pytest
from context_cut import (
    ANSWER_ALLOWANCE,
    BUDGET,
    MAX_SEQ_LENGTH,
    PROMPT_RESERVE,
    deliver,
    recall,
    select_k,
    summarize,
)

# Ten abstracts; each word is one "token" under the fake counter.
BIBS = tuple(f"2024ApJ...{i:03d}..001A" for i in range(1, 11))
SIZES = (30, 30, 30, 30, 30, 30, 30, 30, 30, 30)
HEADER = "query: q hdr"  # 3 tokens


def _abstract(i: int) -> str:
    return f"[{i + 1}] {BIBS[i]} " + " ".join(f"w{i}_{j}" for j in range(SIZES[i] - 2))


def render_top(n: int) -> str:
    """Fake render_context: header + whole abstract blocks for the top-n slice."""
    return "\n".join([HEADER, *(_abstract(i) for i in range(n))])


def count_tokens(text: str) -> int:
    return len(text.split())


def test_max_seq_length_matches_plan() -> None:
    assert BUDGET == 16_384
    assert MAX_SEQ_LENGTH == 20_480  # the D1 text's figure, kept verbatim
    assert BUDGET + PROMPT_RESERVE + ANSWER_ALLOWANCE <= MAX_SEQ_LENGTH


def test_no_overflow_delivers_the_full_top_k() -> None:
    d = deliver(BIBS, render_top, count_tokens, k=5, budget=1_000)
    assert d.n_rendered == d.n_delivered == 5
    assert d.delivered == BIBS[:5]
    assert d.dropped == ()
    assert not d.overflow and d.fits
    assert d.rendered_tokens == d.delivered_tokens == 3 + 5 * 30
    assert d.text == render_top(5)


def test_k_larger_than_ranking_renders_what_exists() -> None:
    d = deliver(BIBS[:3], render_top, count_tokens, k=50, budget=1_000)
    assert d.n_rendered == d.n_delivered == 3
    assert d.delivered == BIBS[:3]


def test_overflow_drops_whole_abstracts_from_the_bottom_only() -> None:
    budget = 3 + 6 * 30 + 10  # room for 6 abstracts, not 7
    d = deliver(BIBS, render_top, count_tokens, k=8, budget=budget)
    assert d.overflow and d.fits
    assert d.n_rendered == 8 and d.n_delivered == 6
    assert d.delivered == BIBS[:6]  # a prefix: nothing above the cut point is removed
    assert d.dropped == BIBS[6:8]  # the bottom two, in rank order
    assert d.rendered_tokens == 3 + 8 * 30
    assert d.delivered_tokens == 3 + 6 * 30 <= budget
    # Minimal drop: one more abstract would not have fit.
    assert count_tokens(render_top(7)) > budget


def test_overflow_never_truncates_inside_an_abstract() -> None:
    budget = 3 + 4 * 30 + 29  # 29 spare tokens: one short of a fifth whole abstract
    d = deliver(BIBS, render_top, count_tokens, k=10, budget=budget)
    assert d.text == render_top(d.n_delivered)
    for i in range(d.n_delivered):
        assert _abstract(i) in d.text  # every delivered abstract is whole
    for i in range(d.n_delivered, 10):
        assert BIBS[i] not in d.text  # no fragment of a dropped abstract leaks in
        assert f"w{i}_0" not in d.text
    assert d.delivered_tokens == 3 + 4 * 30  # the 29 spare tokens are left unused


def test_degenerate_budget_drops_everything_but_reports_it() -> None:
    d = deliver(BIBS, render_top, count_tokens, k=3, budget=2)
    assert d.n_delivered == 0 and d.dropped == BIBS[:3]
    assert d.overflow and not d.fits


def test_recall_scored_on_delivered_not_rendered() -> None:
    gold = [BIBS[1], BIBS[7]]  # rank 2 and rank 8
    budget = 3 + 6 * 30
    d = deliver(BIBS, render_top, count_tokens, k=8, budget=budget)
    rendered = BIBS[: d.n_rendered]
    assert recall(gold, rendered) == 1.0
    assert recall(gold, d.delivered) == 0.5  # rank-8 gold was dropped by the overflow policy


def test_recall_matches_rank_cut_definition() -> None:
    gold = [BIBS[0], BIBS[4], "2099ZZZZ.999..999Z"]
    rank = {b: i + 1 for i, b in enumerate(BIBS)}
    for cut in (1, 4, 5, 10):
        by_rank = sum(1 for g in gold if rank.get(g, 1 << 30) <= cut) / len(gold)
        assert recall(gold, BIBS[:cut]) == by_rank
    with pytest.raises(ValueError):
        recall([], BIBS)


def test_summarize_table_readouts() -> None:
    rows = [
        {"delivered_recall": 1.0, "rendered_recall": 1.0, "rendered_tokens": 100,
         "delivered_tokens": 100, "overflow": False, "n_dropped": 0},
        {"delivered_recall": 0.5, "rendered_recall": 1.0, "rendered_tokens": 300,
         "delivered_tokens": 200, "overflow": True, "n_dropped": 2},
        {"delivered_recall": 0.0, "rendered_recall": 0.0, "rendered_tokens": 200,
         "delivered_tokens": 200, "overflow": False, "n_dropped": 0},
    ]
    s = summarize(rows)
    assert s["n"] == 3
    assert s["delivered_recall"] == pytest.approx(0.5)
    assert s["rendered_recall"] == pytest.approx(2 / 3)
    assert s["median_rendered_tokens"] == 200
    assert s["p95_rendered_tokens"] == pytest.approx(290.0)  # numpy-linear on [100, 200, 300]
    assert s["overflow_rate"] == pytest.approx(1 / 3)
    assert s["fit_rate"] == pytest.approx(2 / 3)
    assert s["mean_dropped"] == pytest.approx(2 / 3)
    assert summarize([]) == {"n": 0}


def test_selection_within_tolerance_picks_smallest_qualifying_k() -> None:
    rec = {10: 0.592, 20: 0.70, 30: 0.82, 50: 0.862}
    fit = {10: 1.0, 20: 1.0, 30: 1.0, 50: 0.90}
    s = select_k(rec, fit)
    assert s.branch == "within_tolerance"
    assert s.k == 30 and s.literal_k == 30
    assert s.recall_gap == pytest.approx(0.042)


def test_selection_tolerance_is_inclusive_despite_float_error() -> None:
    rec = {10: 0.5, 20: 0.75, 30: 0.78, 50: 0.8}  # 0.8 - 0.75 = 0.05000000000000004 in floats
    fit = dict.fromkeys(rec, 1.0)
    assert select_k(rec, fit).k == 20


def test_selection_fallback_largest_k_fitting_on_95_percent() -> None:
    rec = {10: 0.5, 20: 0.6, 30: 0.7, 50: 0.9}
    fit = {10: 1.0, 20: 1.0, 30: 0.97, 50: 0.80}
    s = select_k(rec, fit)
    assert s.branch == "fallback_budget_fit"
    assert s.k == 30
    assert s.recall_gap == pytest.approx(0.2)
    assert s.literal_k == 50  # the reading where R@50 wins the first branch trivially


def test_selection_fallback_can_choose_the_reference_cut() -> None:
    rec = {10: 0.5, 20: 0.6, 30: 0.7, 50: 0.9}
    fit = {10: 1.0, 20: 1.0, 30: 1.0, 50: 0.95}  # exactly 95% counts as fitting
    s = select_k(rec, fit)
    assert s.branch == "fallback_budget_fit"
    assert s.k == 50 and s.recall_gap == 0.0


def test_selection_none_when_nothing_fits() -> None:
    rec = {10: 0.5, 20: 0.6, 30: 0.7, 50: 0.9}
    fit = dict.fromkeys(rec, 0.5)
    s = select_k(rec, fit)
    assert s.k is None and s.branch == "none"


def test_selection_rejects_inconsistent_inputs() -> None:
    with pytest.raises(ValueError):
        select_k({10: 0.5, 20: 0.6}, {10: 1.0, 20: 1.0})  # no reference cut
    with pytest.raises(ValueError):
        select_k({10: 0.5, 50: 0.6}, {10: 1.0})
