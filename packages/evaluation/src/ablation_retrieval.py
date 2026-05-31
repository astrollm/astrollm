"""Retrieval-arm ablation for the pilot: dense vs lexical vs hybrid on the FROZEN corpus.

Runs the same 29-query gold set through all three arms of ``retrieve`` and reports Recall@10
and MRR three ways — named-target (q01-18, n=17), broad known-item (q19-30, n=12), all scored
(n=29). Single-variable discipline: identical corpus snapshot, labels, k, pool and embeddings
across arms; the only thing that changes is which arm(s) feed the ranking (see retrieve()).

Two layers of per-query output so the tails are visible at n=29:
  - scored: Recall@10 and the rank of the first gold hit *within the top-k* (the rank that sets RR).
  - diagnostic (single arms only): the full-corpus rank of the best gold paper, depth = |corpus|.
    This is pool-independent (a single arm has no fusion pool) and reveals single-arm-strong tails
    — e.g. q12's ERO sits at dense #338 / lexical #4, which is why hybrid (pool 50) buries it. It
    does NOT touch the scored hybrid config (pool stays fixed); it is annotation only.

Usage (from repo root, with the pilot DB up and indexed):
    uv run python packages/evaluation/src/ablation_retrieval.py \
        --queries packages/evaluation/queries/pilot_exoplanet_atmospheres.yaml \
        --json-out docs/research/pilot-ablation-results.json
"""

import json
import sys
from pathlib import Path
from typing import Any

import typer
import yaml
from rich.console import Console

# Import the shared retrieval module from the rag package (pre-packaging monorepo).
_RAG_SRC = Path(__file__).resolve().parents[3] / "packages" / "rag" / "src"
sys.path.insert(0, str(_RAG_SRC))
from pilot_retrieval import (  # noqa: E402
    ARMS,
    RRF_K,
    db_connect,
    dense_search,
    lexical_search,
    retrieve,
)

err = Console(stderr=True)
app = typer.Typer(add_completion=False, help=__doc__)

# The gold set's own header documents the split: q01-q18 are NAMED-TARGET, q19-q30 BROAD/TOPICAL.
NAMED_MAX_ID = 18


def _kind(qid: str) -> str:
    """named for q01-q18, broad for q19+ (the split documented in the gold-set header)."""
    n = int("".join(c for c in qid if c.isdigit()))
    return "named" if n <= NAMED_MAX_ID else "broad"


def _first_hit_rank(expected: list[str], ordered: list[str]) -> int | None:
    """1-based rank of the first expected bibcode in ``ordered``, or None if absent."""
    for rank, bibcode in enumerate(ordered, 1):
        if bibcode in expected:
            return rank
    return None


def _best_gold_rank(expected: list[str], ranked: list[tuple[str, float]]) -> int | None:
    """Best (smallest) 1-based rank at which any gold paper appears in a full arm ranking."""
    pos = {b: i + 1 for i, (b, _) in enumerate(ranked)}
    ranks = [pos[b] for b in expected if b in pos]
    return min(ranks) if ranks else None


def _corpus_size() -> int:
    with db_connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM papers")
        return int(cur.fetchone()[0])


def run(queries_path: Path, k: int, pool: int) -> dict[str, Any]:
    spec = yaml.safe_load(queries_path.read_text())
    items: list[dict[str, Any]] = spec["queries"] if isinstance(spec, dict) else spec
    corpus_n = _corpus_size()

    per_query: list[dict[str, Any]] = []
    skipped: list[str] = []
    for item in items:
        expected = item.get("expected_bibcodes") or []
        if not expected:
            skipped.append(item["id"])
            continue
        query = item["query"]
        row: dict[str, Any] = {
            "id": item["id"],
            "target": item.get("target", ""),
            "query": query,
            "kind": _kind(item["id"]),
            "n_gold": len(expected),
            "arms": {},
        }
        # Diagnostic full-corpus single-arm rankings (pool-independent annotation only).
        dense_full = dense_search(query, corpus_n)
        lexical_full = lexical_search(query, corpus_n)
        row["diag"] = {
            "dense_best_rank": _best_gold_rank(expected, dense_full),
            "lexical_best_rank": _best_gold_rank(expected, lexical_full),
            "corpus_n": corpus_n,
        }
        for arm in ARMS:
            ordered = [r["bibcode"] for r in retrieve(query, arm=arm, k=k, pool=pool)]
            found = sum(1 for b in expected if b in ordered[:k])
            hit = _first_hit_rank(expected, ordered[:k])
            row["arms"][arm] = {
                "recall": found / len(expected),
                "rr": (1.0 / hit) if hit else 0.0,
                "hit_rank": hit,
            }
        per_query.append(row)

    aggregates = _aggregate(per_query)
    return {
        "config": {
            "k": k,
            "pool": pool,
            "rrf_k": RRF_K,
            "corpus_n": corpus_n,
            "queries_file": str(queries_path),
            "n_scored": len(per_query),
            "n_named": sum(1 for r in per_query if r["kind"] == "named"),
            "n_broad": sum(1 for r in per_query if r["kind"] == "broad"),
            "skipped": skipped,
        },
        "aggregates": aggregates,
        "per_query": per_query,
    }


def _aggregate(per_query: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, float]]]:
    """{arm: {split: {recall, mrr, n}}} for split in named/broad/all."""
    out: dict[str, dict[str, dict[str, float]]] = {}
    for arm in ARMS:
        out[arm] = {}
        for split in ("named", "broad", "all"):
            rows = [r for r in per_query if split == "all" or r["kind"] == split]
            n = len(rows)
            recall = sum(r["arms"][arm]["recall"] for r in rows) / n if n else 0.0
            mrr = sum(r["arms"][arm]["rr"] for r in rows) / n if n else 0.0
            out[arm][split] = {"recall": recall, "mrr": mrr, "n": n}
    return out


# ── Markdown emitters (ready to paste into docs/research/pilot-ablation.md) ──


def _md_aggregate(agg: dict[str, Any]) -> str:
    lines = [
        "| arm | named R@10 | named MRR | broad R@10 | broad MRR | all R@10 | all MRR |",
        "|---|---|---|---|---|---|---|",
    ]
    for arm in ARMS:
        a = agg[arm]
        lines.append(
            f"| {arm} | {a['named']['recall']:.3f} | {a['named']['mrr']:.3f} "
            f"| {a['broad']['recall']:.3f} | {a['broad']['mrr']:.3f} "
            f"| {a['all']['recall']:.3f} | {a['all']['mrr']:.3f} |"
        )
    return "\n".join(lines)


def _cell(arm_res: dict[str, Any]) -> str:
    rank = arm_res["hit_rank"]
    return f"{arm_res['recall']:.2f} (#{rank})" if rank else f"{arm_res['recall']:.2f} (—)"


def _md_per_query(per_query: list[dict[str, Any]]) -> str:
    lines = [
        "| id | target | kind | gold | dense R@10 (rank) | lexical R@10 (rank) "
        "| hybrid R@10 (rank) |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in per_query:
        lines.append(
            f"| {r['id']} | {r['target']} | {r['kind']} | {r['n_gold']} "
            f"| {_cell(r['arms']['dense'])} | {_cell(r['arms']['lexical'])} "
            f"| {_cell(r['arms']['hybrid'])} |"
        )
    return "\n".join(lines)


def _md_diagnostic(per_query: list[dict[str, Any]]) -> str:
    """Decisive queries — where the three arms do not all agree on hit/miss in the top 10.

    Shows the scored top-10 first-gold rank per arm, plus the single arms' full-corpus position
    of the best gold (depth = |corpus|), which is what explains the hybrid hit/miss.
    """
    lines = [
        "| id | target | dense (top10 / full) | lexical (top10 / full) | hybrid (top10) |",
        "|---|---|---|---|---|",
    ]
    for r in per_query:
        hits = {arm: r["arms"][arm]["hit_rank"] is not None for arm in ARMS}
        if len(set(hits.values())) == 1:
            continue  # all agree (all hit or all miss) — not decisive
        n = r["diag"]["corpus_n"]

        def _full(v: int | None) -> str:
            return f"#{v}" if v else "—"

        def _t10(arm: str) -> str:
            hr = r["arms"][arm]["hit_rank"]
            return f"#{hr}" if hr else "miss"

        lines.append(
            f"| {r['id']} | {r['target']} "
            f"| {_t10('dense')} / {_full(r['diag']['dense_best_rank'])} of {n} "
            f"| {_t10('lexical')} / {_full(r['diag']['lexical_best_rank'])} of {n} "
            f"| {_t10('hybrid')} |"
        )
    return "\n".join(lines)


@app.command()
def main(
    queries: Path = typer.Option(..., exists=True, help="YAML query set (the gold labels)."),
    k: int = typer.Option(10, help="Cutoff for Recall@k and MRR."),
    pool: int = typer.Option(50, help="Per-arm candidate depth for RRF (pilot value — keep 50)."),
    json_out: Path = typer.Option(None, "--json-out", help="Write the raw results JSON here."),
) -> None:
    results = run(queries, k=k, pool=pool)
    cfg = results["config"]
    err.log(
        f"corpus={cfg['corpus_n']} | scored={cfg['n_scored']} "
        f"(named={cfg['n_named']}, broad={cfg['n_broad']}) | skipped={cfg['skipped']} "
        f"| k={k} pool={pool} rrf_k={cfg['rrf_k']}"
    )

    print("### Aggregate — Recall@10 / MRR by arm and split\n")
    print(_md_aggregate(results["aggregates"]))
    print("\n### Per-query (all scored, raw)\n")
    print(_md_per_query(results["per_query"]))
    print("\n### Decisive queries — arms disagree on top-10 hit/miss\n")
    print(_md_diagnostic(results["per_query"]))

    if json_out:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(results, indent=2))
        err.log(f"[green]wrote raw results[/green] → {json_out}")


if __name__ == "__main__":
    app()
