"""Render the M1 exit-review marking sheet from the batch run index + copilot transcripts.

Thin CLI over ``packages/inference/src/review_sheet.py`` (pure rendering). Writes:

* ``docs/research/m1-conversation-review.md`` — the sheet the reviewer marks. Never overwritten
  unless ``--force`` (it holds human marks);
* ``data/eval/review/sheet.generated.md`` — always refreshed, for diffing against the marked copy;
* with ``--context`` (default), ``data/eval/review/context/<id>.md`` — each turn's delivered
  abstracts exactly as the model saw them, regenerated with ``assemble_context`` +
  ``render_context(limit=delivered_n)`` (needs the pilot DB). Abstract text only ever goes under
  ``data/eval/`` (gitignored), never into ``docs/``.

    uv run python scripts/m1_review_sheet.py
    uv run python scripts/m1_review_sheet.py --no-context --force
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import typer
import yaml
from rich.console import Console

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (
    REPO_ROOT / "packages" / "inference" / "src",
    REPO_ROOT / "packages" / "data-pipeline" / "src" / "sft",
):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from corpus import DEFAULT_CORPUS_PATH, CorpusSnapshot  # noqa: E402
from review_sheet import render_context_doc, render_sheet  # noqa: E402

DEFAULT_QUERIES = REPO_ROOT / "packages" / "evaluation" / "queries" / "m1_review_queries.yaml"
DEFAULT_INDEX = REPO_ROOT / "data" / "eval" / "review" / "m1-review-run.json"
DEFAULT_OUT = REPO_ROOT / "docs" / "research" / "m1-conversation-review.md"
REVIEW_DIR = REPO_ROOT / "data" / "eval" / "review"

err = Console(stderr=True)
app = typer.Typer(add_completion=False, help=__doc__)


def _is_gitignored(path: Path) -> bool:
    rel = path.resolve().relative_to(REPO_ROOT)
    return subprocess.run(
        ["git", "check-ignore", "-q", str(rel)], cwd=REPO_ROOT, check=False
    ).returncode == 0


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def load_records(index: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Join the run index to transcript records by (session_id, turn) → {review id: record}."""
    by_key: dict[tuple[str, int], dict[str, Any]] = {}
    for path in sorted({entry["transcript_path"] for entry in index.values()}):
        full = Path(path) if Path(path).is_absolute() else REPO_ROOT / path
        for line in full.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                by_key[(rec["session_id"], rec["turn"])] = rec
    out: dict[str, dict[str, Any]] = {}
    for rid, entry in index.items():
        rec = by_key.get((entry["session_id"], entry["turn"]))
        if rec is None:
            raise KeyError(f"{rid}: no record for session {entry['session_id']} turn "
                           f"{entry['turn']} in {entry['transcript_path']}")
        out[rid] = rec
    return out


def write_contexts(
    queries: list[dict[str, Any]], records: dict[str, dict[str, Any]], snapshot: CorpusSnapshot
) -> int:
    """Regenerate each turn's delivered context (DB needed) into data/eval/review/context/."""
    from context import assemble_context, render_context  # lazy: retriever stack + DB

    out_dir = REVIEW_DIR / "context"
    if not _is_gitignored(out_dir / "x.md"):
        raise RuntimeError(f"{out_dir} is not gitignored; refusing to write abstract text there")
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for q in queries:
        rec = records.get(q["id"])
        if rec is None or not rec.get("delivered_n"):
            continue
        assembled = assemble_context(rec["query"], snapshot)
        delivered = rec["delivered_n"]
        rendered = render_context(assembled, snapshot, limit=delivered)
        now = [r.bibcode for r in assembled.retrieved]
        mismatch = None
        if now[:delivered] != list(rec.get("allowlist") or [])[:delivered]:
            mismatch = ("the regenerated top-n differs from the transcript allowlist — the index "
                        "or corpus changed since the run; this is NOT what the model saw")
        elif rec.get("pool_bibcodes") and now != rec["pool_bibcodes"]:
            mismatch = "the delivered top-n matches, but the pool-100 ranking below it differs"
        if mismatch:
            err.print(f"[yellow]{q['id']}: {mismatch}[/yellow]")
        (out_dir / f"{q['id']}.md").write_text(
            render_context_doc(q["id"], rec["query"], rendered, mismatch), encoding="utf-8"
        )
        n += 1
    return n


@app.command()
def main(
    queries: Path = typer.Option(DEFAULT_QUERIES, help="Review query YAML."),
    index: Path = typer.Option(DEFAULT_INDEX, help="Run index JSON from m1_review_batch.py."),
    out: Path = typer.Option(DEFAULT_OUT, help="The sheet to mark (refused if present)."),
    force: bool = typer.Option(False, "--force", help="Overwrite an existing (marked) sheet."),
    context: bool = typer.Option(True, "--context/--no-context", help="Write context files."),
    corpus: Path = typer.Option(REPO_ROOT / DEFAULT_CORPUS_PATH, help="Frozen corpus JSONL."),
) -> None:
    """Write the review sheet (+ generated copy and per-turn context files)."""
    review_queries = yaml.safe_load(queries.read_text(encoding="utf-8"))["queries"]
    run_index = json.loads(index.read_text(encoding="utf-8")) if index.exists() else {}
    if not run_index:
        err.print(f"[yellow]{index} is missing or empty — the sheet will show every turn as not "
                  "run.[/yellow]")
    records = load_records(run_index)
    snapshot = CorpusSnapshot.load(corpus)

    def title_of(bibcode: str) -> str | None:
        return snapshot.title(bibcode) if bibcode in snapshot else None

    sheet = render_sheet(review_queries, run_index, records, title_of)
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    generated = REVIEW_DIR / "sheet.generated.md"
    generated.write_text(sheet, encoding="utf-8")
    err.print(f"wrote {_rel(generated)} ({len(records)} turns)")
    if out.exists() and not force:
        err.print(f"[yellow]{out} exists (it may hold marks) — left untouched; use --force to "
                  "overwrite.[/yellow]")
    else:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(sheet, encoding="utf-8")
        err.print(f"wrote {out}")
    if context and records:
        n = write_contexts(review_queries, records, snapshot)
        err.print(f"wrote {n} context file(s) under data/eval/review/context/")
    print(out)


if __name__ == "__main__":
    app()
