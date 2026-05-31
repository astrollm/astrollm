"""Retrieval evaluation for the pilot: Recall@10 and MRR over a curated query set.

The query set is a YAML list of {id, query, expected_bibcodes}. Queries whose
``expected_bibcodes`` is empty are skipped and reported (so a draft set under review
still runs without inflating or deflating the score).

Usage (from repo root):
    uv run python packages/evaluation/src/eval_retrieval.py \
        --queries tests/fixtures/pilot/queries.yaml [--k 10]
"""

import sys
from pathlib import Path
from typing import Any

import typer
import yaml
from rich.console import Console
from rich.table import Table

# Import the shared retrieval module from the rag package (pre-packaging monorepo).
_RAG_SRC = Path(__file__).resolve().parents[3] / "packages" / "rag" / "src"
sys.path.insert(0, str(_RAG_SRC))
from pilot_retrieval import hybrid_search  # noqa: E402

err = Console(stderr=True)
app = typer.Typer(add_completion=False, help=__doc__)


def _score_query(expected: list[str], retrieved: list[str], k: int) -> tuple[float, float]:
    """Return (recall@k, reciprocal_rank) for one query."""
    top = retrieved[:k]
    found = sum(1 for b in expected if b in top)
    recall = found / len(expected)
    rr = 0.0
    for rank, bibcode in enumerate(top, 1):
        if bibcode in expected:
            rr = 1.0 / rank
            break
    return recall, rr


@app.command()
def main(
    queries: Path = typer.Option(..., exists=True, help="YAML query set."),
    k: int = typer.Option(10, help="Cutoff for Recall@k and MRR."),
) -> None:
    spec = yaml.safe_load(queries.read_text())
    items: list[dict[str, Any]] = spec["queries"] if isinstance(spec, dict) else spec

    scored, skipped = [], []
    table = Table(title=f"Per-query retrieval (k={k})")
    table.add_column("id")
    table.add_column("query", overflow="fold", max_width=42)
    table.add_column(f"R@{k}", justify="right")
    table.add_column("RR", justify="right")

    for item in items:
        expected = item.get("expected_bibcodes") or []
        if not expected:
            skipped.append(item["id"])
            continue
        retrieved = [r["bibcode"] for r in hybrid_search(item["query"], k=k)]
        recall, rr = _score_query(expected, retrieved, k)
        scored.append((recall, rr))
        table.add_row(str(item["id"]), item["query"], f"{recall:.2f}", f"{rr:.2f}")

    err.print(table)
    if skipped:
        err.log(f"[yellow]skipped {len(skipped)} unscored queries (no expected_bibcodes): "
                f"{', '.join(map(str, skipped))}[/yellow]")

    if not scored:
        err.log("[red]No scorable queries — fill in expected_bibcodes.[/red]")
        raise typer.Exit(1)

    recall_at_k = sum(r for r, _ in scored) / len(scored)
    mrr = sum(rr for _, rr in scored) / len(scored)
    err.log(f"[bold]Scored {len(scored)} queries[/bold]")
    print(f"Recall@{k}: {recall_at_k:.3f}")
    print(f"MRR:       {mrr:.3f}")


if __name__ == "__main__":
    app()
