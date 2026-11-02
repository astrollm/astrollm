"""M1 exit-review sheet — pure markdown rendering (no I/O, no DB, no network).

Joins the review queries (``packages/evaluation/queries/m1_review_queries.yaml``), the batch run
index (``data/eval/review/m1-review-run.json``: review id → session_id / turn / transcript_path /
ts) and the copilot transcript records into one markdown sheet the reviewer only reads and marks:

* a summary table (one row per review id, verdict column blank);
* one section per turn: query, expected behaviour, the answer verbatim, a table of every cited
  bibcode with its title and a blank mark column, and a marking block.

Titles are bibliographic metadata and may land in ``docs/``; abstract text never does — the
per-turn context files (:func:`render_context_doc`) are written under ``data/eval/`` only.
The CLIs are ``scripts/m1_review_batch.py`` (run) and ``scripts/m1_review_sheet.py`` (render).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

MARKS = ("supported", "partial", "unsupported", "wrong-paper")
VERDICTS = ("good", "minor", "bad")
FLAG_KEYS = ("mixed", "marker_misplaced", "sources_line_malformed")  # from the parsed block

Record = Mapping[str, Any]
TitleOf = Callable[[str], str | None]


def select_pending(
    queries: Sequence[Mapping[str, Any]],
    done: Mapping[str, Any],
    *,
    start: str | None = None,
    limit: int | None = None,
) -> list[Mapping[str, Any]]:
    """Queries from ``start`` (a review id) onward, minus ids already in ``done``, capped."""
    ids = [q["id"] for q in queries]
    if start is not None and start not in ids:
        raise ValueError(f"unknown review id {start!r}; expected one of {ids[0]}..{ids[-1]}")
    begin = ids.index(start) if start is not None else 0
    pending = [q for q in queries[begin:] if q["id"] not in done]
    return pending if limit is None else pending[:limit]


def _cell(text: Any) -> str:
    """A markdown table cell: one line, pipes escaped."""
    return " ".join(str(text).split()).replace("|", "\\|")


def _fence(text: str, info: str = "text") -> str:
    """A fenced block that cannot be closed early by backticks inside ``text``."""
    longest = run = 0
    for ch in text:
        run = run + 1 if ch == "`" else 0
        longest = max(longest, run)
    ticks = "`" * max(3, longest + 1)
    return f"{ticks}{info}\n{text}\n{ticks}"


def abstain_label(record: Record) -> str:
    parsed = record.get("parsed") or {}
    if not parsed:
        return "—"
    if parsed.get("is_abstain"):
        kind = "abstain" if parsed.get("pure_abstention") else "mixed"
        return f"{kind}: {parsed.get('reason')}"
    return "answer"


def flags(record: Record) -> list[str]:
    parsed = record.get("parsed") or {}
    out = [k for k in FLAG_KEYS if parsed.get(k)]
    if record.get("had_think"):
        out.append("had_think")
    return out


def guard_label(record: Record) -> str:
    if record.get("error") and not record.get("guard"):
        return f"error: {record['error'].get('type')}"
    status = (record.get("guard") or {}).get("status", "—")
    return f"{status} (error: {record['error'].get('type')})" if record.get("error") else status


def cited_rows(record: Record, title_of: TitleOf) -> list[tuple[int, str, str, str]]:
    """(n, bibcode, title, context rank) for every bibcode the answer cites, in order."""
    parsed = record.get("parsed") or {}
    allow = list(record.get("allowlist") or [])
    rows = []
    for n, bib in enumerate(parsed.get("all_bibcodes") or [], 1):
        title = title_of(bib)
        rank = f"rank {allow.index(bib) + 1}" if bib in allow else "NOT IN CONTEXT"
        rows.append((n, bib, title if title is not None else "(not in corpus)", rank))
    return rows


def render_summary(
    queries: Sequence[Mapping[str, Any]], records: Mapping[str, Record]
) -> list[str]:
    lines = [
        "| id | class | guard | abstain/reason | flags | delivered_n | context_tokens | verdict "
        "| tag |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for q in queries:
        r = records.get(q["id"])
        if r is None:
            lines.append(f"| {q['id']} | {q['class']} | not run | — | — | — | — |  | {q['tag']} |")
            continue
        lines.append(
            f"| {q['id']} | {q['class']} | {_cell(guard_label(r))} | {_cell(abstain_label(r))} "
            f"| {', '.join(flags(r)) or '—'} | {r.get('delivered_n', '—')} "
            f"| {r.get('context_tokens', '—')} |  | {_cell(r.get('tag') or q['tag'])} |"
        )
    return lines


def render_turn(
    q: Mapping[str, Any],
    record: Record | None,
    entry: Mapping[str, Any] | None,
    title_of: TitleOf,
) -> list[str]:
    lines = [f"## {q['id']} — {q['class']}", "", f"**Query:** {q['query']}", ""]
    lines += [f"**Expected:** {q['expected']}", ""]
    if record is None:
        return lines + ["*Not run yet.*", ""]
    guard = record.get("guard") or {}
    meta = [
        f"guard `{guard_label(record)}`",
        f"{abstain_label(record)}",
        f"delivered {record.get('delivered_n', '—')}",
        f"context tokens {record.get('context_tokens', '—')}",
    ]
    if record.get("latency_ms") is not None:
        meta.append(f"latency {record['latency_ms'] / 1000:.1f} s")
    if entry is not None:
        meta.append(f"session `{str(entry.get('session_id', ''))[:12]}` turn {entry.get('turn')}")
    lines += [" · ".join(meta), ""]
    if f := flags(record):
        lines += [f"**Flags:** {', '.join(f)}", ""]
    if error := record.get("error"):
        lines += [f"**Error:** `{error.get('type')}`: {error.get('message')}", ""]
    if record.get("answer") is not None:
        lines += ["**Answer** (verbatim):", "", _fence(record["answer"]), ""]
    first = record.get("first_answer")
    if first is not None and first != record.get("answer"):
        bad = ", ".join(guard.get("first_violations") or []) or "none (uncited)"
        lines += [
            f"**First answer** (rejected by the guard; off-list: {bad}):",
            "",
            _fence(first),
            "",
        ]
    rows = cited_rows(record, title_of)
    if rows:
        lines += ["| # | bibcode | title | in context | mark |", "|---|---|---|---|---|"]
        lines += [f"| {n} | `{b}` | {_cell(t)} | {rank} |  |" for n, b, t, rank in rows]
        lines += ["", f"Marks: {' / '.join(MARKS)}", ""]
    else:
        lines += ["*No bibcodes cited.*", ""]
    if record.get("delivered_n"):
        lines += [f"Context as the model saw it: `data/eval/review/context/{q['id']}.md`", ""]
    lines += [
        "- Claims supported: __ / __",
        f"- Verdict: {' | '.join(VERDICTS)}",
        f"- Tag: {record.get('tag') or q['tag']}",
        "- Note:",
        "",
    ]
    return lines


def _run_meta(records: Mapping[str, Record]) -> list[str]:
    keys = ("model", "mode", "k", "pool", "budget", "tokenizer", "temperature", "system_prompt_sha")
    out = []
    for key in keys:
        values = sorted({str(r.get(key)) for r in records.values() if r.get(key) is not None})
        if values:
            out.append(f"{key} `{', '.join(values)}`")
    stamps = sorted(str(r["ts"]) for r in records.values() if r.get("ts"))
    if stamps:
        out.append(f"turns `{stamps[0]}` → `{stamps[-1]}`")
    return out


def render_sheet(
    queries: Sequence[Mapping[str, Any]],
    index: Mapping[str, Mapping[str, Any]],
    records: Mapping[str, Record],
    title_of: TitleOf,
) -> str:
    """The whole review sheet. ``records`` is keyed by review id (already joined via ``index``)."""
    done = sum(1 for q in queries if q["id"] in records)
    lines = [
        "# M1 conversation review",
        "",
        f"Exit review for M1 (Copilot v0): {done} / {len(queries)} conversations logged. "
        "Generated by `scripts/m1_review_sheet.py` from the run index and transcripts; mark in "
        "place (verdict column, mark column, marking block). Abstract text is not reproduced "
        "here — each turn's delivered context is under `data/eval/review/context/` (gitignored).",
        "",
    ]
    if meta := _run_meta(records):
        lines += ["Run: " + " · ".join(meta), ""]
    lines += [
        f"Marks per cited bibcode: {', '.join(MARKS)}. Verdict per turn: {', '.join(VERDICTS)}.",
        "",
        "## Summary",
        "",
        *render_summary(queries, records),
        "",
    ]
    for q in queries:
        lines += render_turn(q, records.get(q["id"]), index.get(q["id"]), title_of)
    return "\n".join(lines).rstrip() + "\n"


def render_context_doc(
    review_id: str, query: str, rendered: str, mismatch: str | None = None
) -> str:
    """Per-turn context file (``data/eval/review/context/<id>.md`` only — it holds abstracts)."""
    lines = [
        f"# {review_id} — delivered context",
        "",
        f"**Query:** {query}",
        "",
        "Regenerated with `assemble_context` + `render_context(limit=delivered_n)`: the exact "
        "context block the model saw (in the prompt it is preceded by the copilot's context intro "
        "line and followed by `Question: <query>`).",
        "",
    ]
    if mismatch:
        lines += [f"**WARNING:** {mismatch}", ""]
    lines += [_fence(rendered), ""]
    return "\n".join(lines)
