"""D1 context-cut logic — pure (no DB, no tokenizer, no retrieval imports).

Pre-registered decision D1 (``docs/V1_PLAN_R2.md``, "Pre-registered constants") fixes the serving
context as the top-k of the fused hybrid pool-100 ranking under a 16,384-token context budget. This
module holds the decision logic so it can be tested offline with a fake token counter; the
pool-sweep harness (``pool_sweep.py d1-cut``) supplies the real ranking, the real renderer
(``sft/context.render_context``) and the real Qwen3.5 tokenizer.

Overflow policy (verbatim from D1): k is a fixed count; if a query's top-k rendering exceeds the
budget, drop whole abstracts from the bottom of the ranking until it fits, never truncate inside an
abstract, and log the drop. Recall is scored on the *delivered* context after overflow handling.

Why the renderer is a callable ``render_top(n)`` rather than a list of per-abstract strings: the
rendering's header line depends on how many abstracts are shown ("showing top n"), and BPE token
counts are not additive across block boundaries, so the only faithful count is of the exact string
the model would receive. Bottom-drop of a top-k slice is itself a top-n slice, so the delivered
context is always ``render_top(n_delivered)`` — whole abstracts by construction.

Token counting covers the context rendering only; the system prompt and question are a separate
1,024-token reservation and the answer a separate 2,048. The plan states ``max_seq_length`` =
20,480, which is kept verbatim; note the three components sum to 19,456, leaving 1,024 tokens
unallocated by the D1 text (flagged for the orchestrator, not resolved here).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from statistics import median
from typing import Any

BUDGET = 16_384  # context tokens per query (D1)
PROMPT_RESERVE = 1_024  # system prompt + question (D1), NOT counted against BUDGET
ANSWER_ALLOWANCE = 2_048  # generation allowance (D1)
MAX_SEQ_LENGTH = 20_480  # as stated in D1; components above sum to 19,456 (1,024 slack)
CUTS = (10, 20, 30, 50)  # pre-registered candidate k values
REFERENCE_K = 50  # the rule compares every cut against delivered R@50
TOLERANCE = 0.05  # "within 0.05 of delivered R@50" (inclusive)
MIN_FIT_RATE = 0.95  # fallback: "fits the budget on >= 95% of queries"
_EPS = 1e-9  # float slack so an exact 0.05 gap / 95% rate is not lost to rounding


@dataclass(frozen=True)
class Delivery:
    """One query's context at one cut, before and after whole-abstract bottom-drop."""

    k: int
    n_rendered: int  # abstracts in the top-k slice (min(k, |ranking|))
    n_delivered: int  # abstracts actually delivered after overflow handling
    delivered: tuple[str, ...]  # bibcodes delivered, in rank order (a prefix of the ranking)
    dropped: tuple[str, ...]  # bibcodes dropped from the bottom, in rank order
    rendered_tokens: int  # tokens of the top-k rendering (pre-drop)
    delivered_tokens: int  # tokens of the delivered rendering
    budget: int
    text: str  # the delivered rendering, exactly render_top(n_delivered)

    @property
    def overflow(self) -> bool:
        """True iff the top-k rendering exceeded the budget (i.e. the query did not fit)."""
        return self.rendered_tokens > self.budget

    @property
    def fits(self) -> bool:
        """True iff the delivered rendering is within budget (False only if even 0 abstracts
        overflow — a degenerate header-only case)."""
        return self.delivered_tokens <= self.budget


def deliver(
    bibcodes: Sequence[str],
    render_top: Callable[[int], str],
    count_tokens: Callable[[str], int],
    k: int,
    budget: int = BUDGET,
) -> Delivery:
    """Apply the D1 overflow policy to one query's ranking at cut ``k``.

    ``bibcodes`` is the full fused ranking (best first); ``render_top(n)`` renders its top-``n``
    slice exactly as the model sees it. Abstracts are dropped one at a time from the bottom until
    the rendering fits ``budget``; nothing is ever truncated inside an abstract.
    """
    if k < 1:
        raise ValueError(f"cut k must be >= 1, got {k}")
    if budget < 1:
        raise ValueError(f"budget must be >= 1, got {budget}")
    n_rendered = min(k, len(bibcodes))
    text = render_top(n_rendered)
    rendered_tokens = tokens = count_tokens(text)
    n = n_rendered
    while tokens > budget and n > 0:
        n -= 1
        text = render_top(n)
        tokens = count_tokens(text)
    return Delivery(
        k=k,
        n_rendered=n_rendered,
        n_delivered=n,
        delivered=tuple(bibcodes[:n]),
        dropped=tuple(bibcodes[n:n_rendered]),
        rendered_tokens=rendered_tokens,
        delivered_tokens=tokens,
        budget=budget,
        text=text,
    )


def recall(gold: Sequence[str], context: Sequence[str]) -> float:
    """Fraction of gold bibcodes present in ``context``.

    Same definition as the pool-sweep harness's ``_recall_at`` (fraction of gold ranked within the
    cut): for a context that is the top-n prefix of a ranking the two are identical.
    """
    if not gold:
        raise ValueError("recall is undefined for an empty gold set")
    present = set(context)
    return sum(1 for g in gold if g in present) / len(gold)


def _p95(values: Sequence[int]) -> float:
    """95th percentile, linear interpolation between closest ranks (numpy's default method)."""
    xs = sorted(values)
    pos = 0.95 * (len(xs) - 1)
    lo, hi = math.floor(pos), math.ceil(pos)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate per-query rows of one cut (keys: delivered_recall, rendered_recall,
    rendered_tokens, delivered_tokens, overflow, n_dropped) into the D1 table readouts."""
    n = len(rows)
    if not n:
        return {"n": 0}
    rendered = [int(r["rendered_tokens"]) for r in rows]
    delivered = [int(r["delivered_tokens"]) for r in rows]
    n_over = sum(1 for r in rows if r["overflow"])
    return {
        "n": n,
        "delivered_recall": sum(r["delivered_recall"] for r in rows) / n,
        "rendered_recall": sum(r["rendered_recall"] for r in rows) / n,
        "median_rendered_tokens": median(rendered),
        "p95_rendered_tokens": _p95(rendered),
        "max_rendered_tokens": max(rendered),
        "median_delivered_tokens": median(delivered),
        "p95_delivered_tokens": _p95(delivered),
        "n_overflowing": n_over,
        "overflow_rate": n_over / n,
        "fit_rate": 1.0 - n_over / n,
        "mean_dropped": sum(int(r["n_dropped"]) for r in rows) / n,
        "total_dropped": sum(int(r["n_dropped"]) for r in rows),
    }


@dataclass(frozen=True)
class Selection:
    """Outcome of the pre-registered D1 selection rule (literal), plus the strict reading."""

    k: int | None  # the selected k under the LITERAL rule (the pre-registered selection)
    branch: str  # "within_tolerance" | "fallback_budget_fit" | "none"
    reference_recall: float  # delivered R@REFERENCE_K
    recall_gap: float | None  # reference_recall - delivered recall at k
    reason: str
    # Secondary, reported only: the "k < reference_k" reading, in which the reference cut cannot
    # win the first branch against itself, so the fallback can fire. Never the selection.
    strict_k: int | None
    strict_branch: str
    strict_recall_gap: float | None


def _apply_rule(
    candidates: Sequence[int],
    delivered_recall: Mapping[int, float],
    fit_rate: Mapping[int, float],
    reference_k: int,
    tolerance: float,
    min_fit_rate: float,
) -> tuple[int | None, str, float | None, str]:
    """First branch over ``candidates``; fallback over every cut → (k, branch, gap, reason)."""
    ref = delivered_recall[reference_k]
    for k in sorted(candidates):
        gap = ref - delivered_recall[k]
        if gap <= tolerance + _EPS:
            return k, "within_tolerance", gap, (
                f"smallest k with delivered recall within {tolerance} of delivered "
                f"R@{reference_k} ({ref:.3f}); gap {gap:+.3f}"
            )
    fitting = [k for k in delivered_recall if fit_rate[k] + _EPS >= min_fit_rate]
    if not fitting:
        return None, "none", None, (
            f"no candidate within {tolerance} of delivered R@{reference_k} and no k fits the "
            f"budget on >= {min_fit_rate:.0%} of queries"
        )
    k = max(fitting)
    gap = ref - delivered_recall[k]
    return k, "fallback_budget_fit", gap, (
        f"no candidate within {tolerance} of delivered R@{reference_k} ({ref:.3f}); largest k "
        f"fitting on >= {min_fit_rate:.0%} of queries (fit {fit_rate[k]:.1%}); recall gap "
        f"{gap:+.3f}"
    )


def select_k(
    delivered_recall: Mapping[int, float],
    fit_rate: Mapping[int, float],
    *,
    reference_k: int = REFERENCE_K,
    tolerance: float = TOLERANCE,
    min_fit_rate: float = MIN_FIT_RATE,
) -> Selection:
    """The D1 rule, literally: the smallest k in the cuts (``reference_k`` included) whose
    delivered recall is within ``tolerance`` of delivered R@``reference_k``; if none qualifies,
    the largest k that fits the budget on >= ``min_fit_rate`` of queries, with the recall gap.

    Because the reference cut is itself a candidate (gap 0), the literal fallback is unreachable;
    it is kept for completeness. ``strict_k`` reports the "k < reference_k" reading, whose
    fallback can fire — a secondary readout, never the selection.
    """
    if reference_k not in delivered_recall:
        raise ValueError(f"reference cut {reference_k} missing from delivered_recall")
    if set(delivered_recall) != set(fit_rate):
        raise ValueError("delivered_recall and fit_rate must cover the same cuts")
    args = (delivered_recall, fit_rate, reference_k, tolerance, min_fit_rate)
    k, branch, gap, reason = _apply_rule(list(delivered_recall), *args)
    strict_k, strict_branch, strict_gap, _ = _apply_rule(
        [c for c in delivered_recall if c < reference_k], *args
    )
    return Selection(
        k=k, branch=branch, reference_recall=delivered_recall[reference_k], recall_gap=gap,
        reason=reason, strict_k=strict_k, strict_branch=strict_branch,
        strict_recall_gap=strict_gap,
    )


# ── One token counter for every D1 path (measurement, authoring, serving) ──

QWEN_TOKENIZER = "Qwen/Qwen3.5-4B"


@lru_cache(maxsize=4)
def load_tokenizer(name: str = QWEN_TOKENIZER) -> Any:
    """The HF tokenizer (tokenizer files only, no weights); loaded once per process."""
    from transformers import AutoTokenizer  # deferred: heavy, and the pure logic never needs it

    return AutoTokenizer.from_pretrained(name)


def qwen_token_counter(name: str = QWEN_TOKENIZER) -> Callable[[str], int]:
    """The D1 token counter: ``len(tok.encode(text, add_special_tokens=False))``."""
    tok = load_tokenizer(name)

    def count_tokens(text: str) -> int:
        return len(tok.encode(text, add_special_tokens=False))

    return count_tokens
