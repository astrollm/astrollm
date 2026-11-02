"""Run the M1 exit-review conversations through ONE in-process copilot session (grounded mode).

Each query from ``packages/evaluation/queries/m1_review_queries.yaml`` is one copilot turn, tagged
with the YAML ``tag``. Settings are built exactly as ``copilot.py ask`` builds them, so the
tokenizer and embedder load once for the whole batch. After every turn the run index
``data/eval/review/m1-review-run.json`` (review id → session_id / turn / transcript_path / ts) is
rewritten atomically; ids already in the index are skipped, so an interrupted run resumes.

    uv run python scripts/m1_review_batch.py --dry-run     # print the plan only
    uv run python scripts/m1_review_batch.py               # run every pending query
    uv run python scripts/m1_review_batch.py --start r18 --limit 3

Needs the pilot Postgres (DATABASE_URL) and llama-server (LLM_BASE_URL). Progress → stderr; the
copilot prints each answer to stdout.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import typer
import yaml
from rich.console import Console

REPO_ROOT = Path(__file__).resolve().parents[1]
_INFERENCE_SRC = REPO_ROOT / "packages" / "inference" / "src"
if str(_INFERENCE_SRC) not in sys.path:
    sys.path.insert(0, str(_INFERENCE_SRC))

import copilot  # noqa: E402
from review_sheet import select_pending  # noqa: E402

DEFAULT_QUERIES = REPO_ROOT / "packages" / "evaluation" / "queries" / "m1_review_queries.yaml"
DEFAULT_INDEX = REPO_ROOT / "data" / "eval" / "review" / "m1-review-run.json"

err = Console(stderr=True)
app = typer.Typer(add_completion=False, help=__doc__)


def load_queries(path: Path) -> list[dict[str, Any]]:
    queries = yaml.safe_load(path.read_text(encoding="utf-8"))["queries"]
    tags = {t.value for t in copilot.Tag}
    for q in queries:
        if q["tag"] not in tags:
            raise ValueError(f"{q['id']}: tag {q['tag']!r} is not an error-taxonomy tag")
    return queries


def load_index(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def save_index(path: Path, index: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(index, fh, indent=2, sort_keys=True)
            fh.write("\n")
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def _assert_gitignored(path: Path) -> None:
    """Transcripts and the index reproduce abstract-derived text: they must never be committed."""
    try:
        rel = path.resolve().relative_to(REPO_ROOT)
    except ValueError as exc:
        raise typer.BadParameter(f"{path} must live under data/eval/ in the repo") from exc
    ok = subprocess.run(
        ["git", "check-ignore", "-q", str(rel)], cwd=REPO_ROOT, check=False
    ).returncode == 0
    if not ok:
        raise typer.BadParameter(f"{rel} is not gitignored; refusing to write review data there")


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _progress(rid: str, n: int, total: int, record: dict[str, Any] | None, error: str = "") -> str:
    if record is None:
        return f"[{rid} {n}/{total}] ERROR {error}"
    parsed = record.get("parsed") or {}
    abst = f"{'abstain' if parsed.get('pure_abstention') else 'mixed'}:{parsed.get('reason')}" \
        if parsed.get("is_abstain") else "answer"
    return (
        f"[{rid} {n}/{total}] guard={record['guard']['status']} {abst} "
        f"ctx={record.get('context_tokens')} latency={record.get('latency_ms', 0) / 1000:.1f}s"
    )


@app.command()
def main(
    queries: Path = typer.Option(DEFAULT_QUERIES, help="Review query YAML."),
    index: Path = typer.Option(DEFAULT_INDEX, help="Run index JSON (resumable)."),
    start: str | None = typer.Option(None, help="First review id to consider (e.g. r18)."),
    limit: int | None = typer.Option(None, help="Run at most this many pending queries."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print the plan only; run nothing."),
) -> None:
    """Run the pending review queries in one copilot session and record them in the index."""
    all_queries = load_queries(queries)
    run_index = load_index(index)
    plan = select_pending(all_queries, run_index, start=start, limit=limit)
    done = [q["id"] for q in all_queries if q["id"] in run_index]
    err.print(f"{len(all_queries)} review queries · {len(done)} already in {_rel(index)} · "
              f"{len(plan)} to run", highlight=False)
    if dry_run:
        for q in plan:
            print(f"{q['id']}\t{q['class']}\t{q['tag']}\t{q['query']}")
        err.print("[bold]DRY RUN[/bold] — no session started, nothing written.")
        return
    if not plan:
        return
    ids = [q["id"] for q in plan]
    if "r23" in ids and "r22" not in ids and "r22" in run_index:
        err.print("[yellow]note: r23 runs in a new session, not right after r22 (resumed run; "
                  "turns are memoryless, so only the ordering differs).[/yellow]")

    _assert_gitignored(index)
    _assert_gitignored(copilot.TRANSCRIPT_DIR / "x.jsonl")
    settings = copilot._settings(
        k=copilot.K, pool=copilot.POOL, budget=copilot.BUDGET, tag=None, live=False,
        dry_run=False, max_tokens=copilot.DEFAULT_MAX_TOKENS, temperature=None,
        corpus=copilot.DEFAULT_CORPUS_PATH, tokenizer=copilot.QWEN_TOKENIZER,
    )
    session = copilot.Session(settings)
    err.print(f"session {session.id} → {_rel(session.path)}", highlight=False)
    failed: list[str] = []
    t0 = time.perf_counter()
    for n, q in enumerate(plan, 1):
        try:
            record = session.run_turn(q["query"], q["tag"])
        except Exception as exc:  # the copilot already logged an error record for this turn
            failed.append(q["id"])
            err.print(_progress(q["id"], n, len(plan), None, f"{type(exc).__name__}: {exc}"),
                      markup=False, highlight=False)
            continue
        assert record is not None
        run_index[q["id"]] = {
            "session_id": record["session_id"],
            "turn": record["turn"],
            "transcript_path": _rel(session.path),
            "ts": record["ts"],
            "query": q["query"],
        }
        save_index(index, run_index)
        err.print(_progress(q["id"], n, len(plan), record), markup=False, highlight=False)
    err.print(f"done in {time.perf_counter() - t0:.0f}s: {len(plan) - len(failed)} ok, "
              f"{len(failed)} failed{' (' + ', '.join(failed) + ')' if failed else ''}; "
              "re-run to retry failures", highlight=False)
    if failed:
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
