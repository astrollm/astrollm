"""Manifest CLI — validate the gold seed and emit ``manifest.json`` + a composition report.

    # write/refresh the manifest after authoring
    uv run python packages/data-pipeline/src/sft/manifest.py build

    # quick progress check vs the pre-registered targets (no file written)
    uv run python packages/data-pipeline/src/sft/manifest.py status

`build` re-validates every gold example (schema contract + corpus rule 6), checks query-level
calibration/eval disjointness and corpus-hash consistency, and writes the manifest carrying the
corpus snapshot hash, retriever config, per-family + per-partition counts, negative-failure-mode
coverage, and the disjointness result. A validation failure or a non-disjoint split makes the
manifest's ``valid`` flag False and exits non-zero, so a broken seed can't be silently published.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer
from context import ARM, POOL, K
from corpus import DEFAULT_CORPUS_PATH, CorpusSnapshot
from partition import build_manifest, load_gold, target_progress
from partition import compose_counts as _compose_counts
from paths import DEFAULT_GOLD_PATH, DEFAULT_MANIFEST_PATH
from rich.console import Console
from rich.table import Table

err = Console(stderr=True)
app = typer.Typer(add_completion=False, help=__doc__)


def _retriever_config() -> dict[str, Any]:
    """The frozen retriever config — from the rag module if importable, else the static fallback."""
    try:
        from context import retriever_config

        return retriever_config()
    except Exception as exc:  # rag stack/DB env unavailable — record what we can, note the gap.
        err.log(f"[yellow]rag config unavailable ({type(exc).__name__}); static knobs[/yellow]")
        return {
            "arm": ARM,
            "k": K,
            "pool": POOL,
            "pipeline": "pilot-retrieval-0.1.0",
            "note": "rrf_k/embed_model unread (rag stack unavailable at manifest time)",
        }


def _render_status(counts: dict[str, Any]) -> Table:
    prog = target_progress(counts)
    table = Table(title="Gold-seed composition vs pre-registered targets")
    table.add_column("partition")
    table.add_column("slice")
    table.add_column("current", justify="right")
    table.add_column("target")
    table.add_column("status")
    rows = [
        ("calibration", "positive", prog["calibration"]["positive"]),
        ("calibration", "negative", prog["calibration"]["negative"]),
        ("eval", "lit_qa", prog["eval"]["lit_qa"]),
        ("eval", "summarization", prog["eval"]["summarization"]),
        ("eval", "abstention", prog["eval"]["abstention"]),
        ("all", "total", prog["total"]),
    ]
    palette = {"met": "green", "under": "yellow", "over": "red"}
    for partition, slice_, st in rows:
        lo, hi = st["target"]
        band = f"{lo}–{hi}" if hi is not None else f"≥{lo}"
        color = palette[st["status"]]
        table.add_row(
            partition, slice_, str(st["current"]), band, f"[{color}]{st['status']}[/{color}]"
        )
    return table


def _load_and_validate(gold: Path, corpus: Path) -> tuple[CorpusSnapshot, list, list[str]]:
    snapshot = CorpusSnapshot.load(corpus)
    err.log(f"corpus snapshot: {len(snapshot)} abstracts, {snapshot.snapshot_hash}")
    examples, errors = load_gold(gold, snapshot.bibcodes)
    err.log(f"loaded {len(examples)} valid example(s) from {gold}")
    return snapshot, examples, errors


@app.command()
def build(
    gold: Path = typer.Option(DEFAULT_GOLD_PATH, help="Gold-seed JSONL."),
    corpus: Path = typer.Option(DEFAULT_CORPUS_PATH, help="Frozen corpus snapshot JSONL."),
    out: Path = typer.Option(DEFAULT_MANIFEST_PATH, help="Manifest output path."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Compute + print; do not write."),
) -> None:
    """Validate the gold seed and write the manifest."""
    snapshot, examples, errors = _load_and_validate(gold, corpus)
    for e in errors:
        err.log(f"[red]invalid[/red]: {e}")

    manifest = build_manifest(
        examples, snapshot, _retriever_config(), source=str(gold), valid=not errors
    )
    disjoint = manifest["disjointness"]["disjoint"]
    consistent = manifest["corpus_consistency"]["all_examples_match"]
    ok = not errors and disjoint and consistent
    manifest["valid"] = ok

    err.print(_render_status(manifest["counts"]))
    if not disjoint:
        err.log(
            "[red]calibration/eval share queries[/red]: "
            f"{', '.join(manifest['disjointness']['shared_queries'][:5])}"
        )
    if not consistent:
        err.log("[red]examples grounded on a different corpus snapshot[/red] (hash mismatch)")
    neg = manifest["counts"]["negative_coverage"]
    if examples and not neg["all_modes_covered"]:
        missing = [k for k in ("claim_not_supported", "contradicts", "wrong_paper",
                               "overgeneralization") if neg[k] == 0]
        err.log(f"[yellow]negative failure modes not yet covered: {', '.join(missing)}[/yellow]")

    if dry_run:
        err.log("[bold]DRY RUN[/bold] — manifest not written")
        print(json.dumps(manifest, indent=2, ensure_ascii=False))
    else:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
        err.log(f"[green]manifest written[/green] → {out} (valid={ok})")
        print(str(out))

    if not ok:
        raise typer.Exit(1)


@app.command()
def status(
    gold: Path = typer.Option(DEFAULT_GOLD_PATH, help="Gold-seed JSONL."),
    corpus: Path = typer.Option(DEFAULT_CORPUS_PATH, help="Frozen corpus snapshot JSONL."),
) -> None:
    """Print composition vs targets without writing the manifest."""
    snapshot, examples, errors = _load_and_validate(gold, corpus)
    for e in errors:
        err.log(f"[red]invalid[/red]: {e}")
    counts = _compose_counts(examples)
    err.print(_render_status(counts))
    neg = counts["negative_coverage"]
    err.log(
        f"negative failure modes covered: {neg['modes_covered']}/4"
        + ("" if neg["all_modes_covered"] else " (incomplete)")
    )
    print(f"total={counts['total']} valid_errors={len(errors)}")


if __name__ == "__main__":
    app()
