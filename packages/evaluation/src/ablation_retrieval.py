"""Retrieval-arm ablation for the pilot: dense vs lexical vs hybrid on the FROZEN corpus.

Runs the same 29-query gold set through all three arms of ``retrieve`` and reports Recall@10
and MRR three ways — named-target (q01-18, n=17), broad known-item (q19-30, n=12), all scored
(n=29). Single-variable discipline: identical corpus snapshot, labels, k, pool and embeddings
across arms; the only thing that changes is which arm(s) feed the ranking (see retrieve()).

Three depth/uncertainty readouts on top of the headline Recall@10/MRR — all ADDITIONAL READOUTS
over the SAME retrieved lists (no re-index, no new arms):
  1. Recall@{pool} (deep cut, =50) per arm, alongside Recall@10 — completes the depth picture.
  2. Candidate-set (union) recall at pool depth + complementarity. For single arms this is the
     arm's top-`pool` recall; for hybrid it is recall of the UNION of both arms' top-`pool`
     candidates (what a stage-2 reranker would receive, >= max(arms) by construction). Reports the
     union recall and the per-document breakdown: dense-only / lexical-only / both / neither.
  3. Bootstrap 95% CIs (B, fixed seed). Marginal per-arm CIs for Recall@10 and MRR per slice for
     context, plus the PAIRED-difference bootstrap (dense-hybrid, lexical-hybrid, dense-lexical) —
     the correct "within noise" test is whether the paired-difference CI includes 0.

Per-query output stays raw so the tails are visible at n=29: Recall@10 first-gold rank (sets RR),
Recall@{pool}, the union recall, each relevant doc's arm tag, and a pool-independent diagnostic
full-corpus rank of the best gold per single arm (depth = |corpus|) — the last reveals
single-arm-strong tails (q12's ERO at dense #338 / lexical #4) without touching the scored config.

Usage (from repo root, with the pilot DB up and indexed):
    uv run python packages/evaluation/src/ablation_retrieval.py \
        --queries packages/evaluation/queries/pilot_exoplanet_atmospheres.yaml \
        --json-out docs/research/pilot-ablation-results.json
"""

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import typer
import yaml
from rich.console import Console

# Import the shared retrieval module from the rag package (pre-packaging monorepo).
_RAG_SRC = Path(__file__).resolve().parents[3] / "packages" / "rag" / "src"
sys.path.insert(0, str(_RAG_SRC))
from pilot_retrieval import (  # noqa: E402
    ARMS,
    RRF_K,
    _rrf_fuse,
    db_connect,
    dense_search,
    lexical_search,
)

err = Console(stderr=True)
app = typer.Typer(add_completion=False, help=__doc__)

# The gold set's own header documents the split: q01-q18 are NAMED-TARGET, q19-q30 BROAD/TOPICAL.
NAMED_MAX_ID = 18
SLICES = ("named", "broad", "all")
# Bootstrap defaults — stated in the writeup for reproducibility.
DEFAULT_BOOTSTRAP_B = 10000
DEFAULT_SEED = 20260531


def _kind(qid: str) -> str:
    """named for q01-q18, broad for q19+ (the split documented in the gold-set header)."""
    n = int("".join(c for c in qid if c.isdigit()))
    return "named" if n <= NAMED_MAX_ID else "broad"


def _recall_at(expected: list[str], rank_map: dict[str, int], cut: int) -> float:
    """Fraction of gold whose rank in ``rank_map`` is within ``cut`` (absent ⇒ not retrieved)."""
    found = sum(1 for g in expected if rank_map.get(g, 1 << 30) <= cut)
    return found / len(expected)


def _first_hit_rank(expected: list[str], rank_map: dict[str, int], cut: int) -> int | None:
    """Smallest rank at which any gold appears within ``cut`` (the rank that sets RR), or None."""
    ranks = [rank_map[g] for g in expected if g in rank_map and rank_map[g] <= cut]
    return min(ranks) if ranks else None


def _best_gold_rank(expected: list[str], rank_map: dict[str, int]) -> int | None:
    """Best (smallest) rank at which any gold appears in a full arm ranking, or None."""
    ranks = [rank_map[g] for g in expected if g in rank_map]
    return min(ranks) if ranks else None


def _corpus_size() -> int:
    with db_connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM papers")
        return int(cur.fetchone()[0])


def run(queries_path: Path, k: int, pool: int, bootstrap_b: int, seed: int) -> dict[str, Any]:
    spec = yaml.safe_load(queries_path.read_text())
    items: list[dict[str, Any]] = spec["queries"] if isinstance(spec, dict) else spec
    corpus_n = _corpus_size()
    # Match retrieve(): each arm must yield at least k candidates, else the hybrid arm (fused only
    # from the top-`pool`) is scored shallower than the single arms when k > pool. No-op for the
    # pilot config (k=10, pool=50). Recorded as the effective pool in the config block below.
    pool = max(pool, k)

    per_query: list[dict[str, Any]] = []
    skipped: list[str] = []
    for item in items:
        expected = item.get("expected_bibcodes") or []
        if not expected:
            skipped.append(item["id"])
            continue
        query = item["query"]

        # One read of each arm to full corpus depth — every readout derives from these.
        dense_full = dense_search(query, corpus_n)
        lexical_full = lexical_search(query, corpus_n)
        dense_rank = {b: i + 1 for i, (b, _) in enumerate(dense_full)}
        lexical_rank = {b: i + 1 for i, (b, _) in enumerate(lexical_full)}
        # Pool-truncated rank maps feed RRF (the candidate set is each arm's top-`pool`).
        dense_pool = {b: r for b, r in dense_rank.items() if r <= pool}
        lexical_pool = {b: r for b, r in lexical_rank.items() if r <= pool}
        fused = _rrf_fuse(dense_pool, lexical_pool)
        fused_rank = {
            b: i + 1
            for i, (b, _) in enumerate(sorted(fused.items(), key=lambda kv: kv[1], reverse=True))
        }
        arm_rank = {"dense": dense_rank, "lexical": lexical_rank, "hybrid": fused_rank}

        arms: dict[str, Any] = {}
        for arm in ARMS:
            rm = arm_rank[arm]
            hit = _first_hit_rank(expected, rm, k)
            arms[arm] = {
                "recall": _recall_at(expected, rm, k),  # recall@k (k=10) — headline, unchanged
                "recall_pool": _recall_at(expected, rm, pool),  # recall@pool (=50)
                "rr": (1.0 / hit) if hit else 0.0,
                "hit_rank": hit,
            }

        # Candidate-set (union) recall + per-document complementarity at pool depth.
        per_doc = []
        tag_counts = {"dense_only": 0, "lexical_only": 0, "both": 0, "neither": 0}
        union_found = 0
        for g in expected:
            in_d, in_l = g in dense_pool, g in lexical_pool
            tag = (
                "both" if in_d and in_l
                else "dense_only" if in_d
                else "lexical_only" if in_l
                else "neither"
            )
            tag_counts[tag] += 1
            union_found += int(in_d or in_l)
            per_doc.append({
                "bibcode": g,
                "dense_rank": dense_rank.get(g),
                "lexical_rank": lexical_rank.get(g),
                "tag": tag,
            })

        per_query.append({
            "id": item["id"],
            "target": item.get("target", ""),
            "query": query,
            "kind": _kind(item["id"]),
            "n_gold": len(expected),
            "arms": arms,
            "union_recall_pool": union_found / len(expected),
            "tags": tag_counts,
            "per_doc": per_doc,
            "diag": {
                "dense_best_rank": _best_gold_rank(expected, dense_rank),
                "lexical_best_rank": _best_gold_rank(expected, lexical_rank),
                "corpus_n": corpus_n,
            },
        })

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
            "bootstrap_B": bootstrap_b,
            "seed": seed,
        },
        "aggregates": _aggregate(per_query),
        "bootstrap": _bootstrap(per_query, bootstrap_b, seed),
        "per_query": per_query,
    }


def _slice_rows(per_query: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    return [r for r in per_query if split == "all" or r["kind"] == split]


def _aggregate(per_query: list[dict[str, Any]]) -> dict[str, Any]:
    arms: dict[str, Any] = {arm: {} for arm in ARMS}
    candidate: dict[str, Any] = {}
    for split in SLICES:
        rows = _slice_rows(per_query, split)
        n = len(rows)
        for arm in ARMS:
            arms[arm][split] = {
                "recall": sum(r["arms"][arm]["recall"] for r in rows) / n if n else 0.0,
                "recall_pool": sum(r["arms"][arm]["recall_pool"] for r in rows) / n if n else 0.0,
                "mrr": sum(r["arms"][arm]["rr"] for r in rows) / n if n else 0.0,
                "n": n,
            }
        docs = {k: sum(r["tags"][k] for r in rows) for k in
                ("dense_only", "lexical_only", "both", "neither")}
        docs["total_relevant"] = sum(r["n_gold"] for r in rows)
        candidate[split] = {
            "union_recall_pool": sum(r["union_recall_pool"] for r in rows) / n if n else 0.0,
            "dense_recall_pool": arms["dense"][split]["recall_pool"],
            "lexical_recall_pool": arms["lexical"][split]["recall_pool"],
            "n_queries": n,
            "docs": docs,
        }
    return {"arms": arms, "candidate": candidate}


def _ci(values: list[float], rng: np.random.Generator, b: int) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    if n == 0:
        return {"mean": 0.0, "lo": 0.0, "hi": 0.0}
    samples = arr[rng.integers(0, n, size=(b, n))].mean(axis=1)
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return {"mean": float(arr.mean()), "lo": float(lo), "hi": float(hi)}


# Paired difference resamples the per-query differences (the paired unit) — not the arms apart.
def _paired_ci(
    a: list[float], b_vals: list[float], rng: np.random.Generator, b: int
) -> dict[str, Any]:
    diff = np.asarray(a, dtype=float) - np.asarray(b_vals, dtype=float)
    out = _ci(list(diff), rng, b)
    out["includes_zero"] = bool(out["lo"] <= 0.0 <= out["hi"])
    return out


def _bootstrap(per_query: list[dict[str, Any]], b: int, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)  # one generator, fixed call order ⇒ fully reproducible
    metrics = {"recall10": "recall", "mrr": "rr"}
    marginal: dict[str, Any] = {arm: {} for arm in ARMS}
    for arm in ARMS:
        for split in SLICES:
            rows = _slice_rows(per_query, split)
            marginal[arm][split] = {
                m: _ci([r["arms"][arm][key] for r in rows], rng, b) for m, key in metrics.items()
            }
    pairs = {
        "dense_minus_hybrid": ("dense", "hybrid"),
        "lexical_minus_hybrid": ("lexical", "hybrid"),
        "dense_minus_lexical": ("dense", "lexical"),
    }
    paired: dict[str, Any] = {name: {} for name in pairs}
    for name, (x, y) in pairs.items():
        for split in SLICES:
            rows = _slice_rows(per_query, split)
            paired[name][split] = {
                m: _paired_ci(
                    [r["arms"][x][key] for r in rows], [r["arms"][y][key] for r in rows], rng, b
                )
                for m, key in metrics.items()
            }
    return {"B": b, "seed": seed, "marginal": marginal, "paired": paired}


# ── Markdown emitters (ready to paste into docs/research/pilot-ablation.md) ──


def _md_aggregate(agg: dict[str, Any]) -> str:
    a = agg["arms"]
    lines = [
        "| arm | named R@10 | named MRR | broad R@10 | broad MRR | all R@10 | all MRR |",
        "|---|---|---|---|---|---|---|",
    ]
    for arm in ARMS:
        x = a[arm]
        lines.append(
            f"| {arm} | {x['named']['recall']:.3f} | {x['named']['mrr']:.3f} "
            f"| {x['broad']['recall']:.3f} | {x['broad']['mrr']:.3f} "
            f"| {x['all']['recall']:.3f} | {x['all']['mrr']:.3f} |"
        )
    return "\n".join(lines)


def _md_depth(agg: dict[str, Any], pool: int) -> str:
    a = agg["arms"]
    lines = [
        f"| arm | named @10 | named @{pool} | broad @10 | broad @{pool} "
        f"| all @10 | all @{pool} |",
        "|---|---|---|---|---|---|---|",
    ]
    for arm in ARMS:
        x = a[arm]
        lines.append(
            f"| {arm} | {x['named']['recall']:.3f} | {x['named']['recall_pool']:.3f} "
            f"| {x['broad']['recall']:.3f} | {x['broad']['recall_pool']:.3f} "
            f"| {x['all']['recall']:.3f} | {x['all']['recall_pool']:.3f} |"
        )
    return "\n".join(lines)


def _md_complementarity(agg: dict[str, Any], pool: int) -> str:
    c = agg["candidate"]
    lines = [
        f"| slice | union R@{pool} | dense R@{pool} | lexical R@{pool} | Δ(union−dense) "
        "| D-only | L-only | both | neither | relevant |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for split in SLICES:
        s = c[split]
        d = s["docs"]
        lines.append(
            f"| {split} | {s['union_recall_pool']:.3f} | {s['dense_recall_pool']:.3f} "
            f"| {s['lexical_recall_pool']:.3f} "
            f"| +{s['union_recall_pool'] - s['dense_recall_pool']:.3f} "
            f"| {d['dense_only']} | {d['lexical_only']} | {d['both']} | {d['neither']} "
            f"| {d['total_relevant']} |"
        )
    return "\n".join(lines)


def _fmt_ci(ci: dict[str, float]) -> str:
    return f"{ci['mean']:+.3f} [{ci['lo']:+.3f}, {ci['hi']:+.3f}]"


def _md_bootstrap(boot: dict[str, Any]) -> str:
    out = [f"**Marginal 95% CIs** (B={boot['B']}, seed={boot['seed']}), all-scored slice:\n"]
    out.append("| arm | R@10 mean [95% CI] | MRR mean [95% CI] |")
    out.append("|---|---|---|")
    for arm in ARMS:
        r = boot["marginal"][arm]["all"]
        out.append(f"| {arm} | {_fmt_ci(r['recall10'])} | {_fmt_ci(r['mrr'])} |")

    out.append("\n**Paired-difference 95% CIs** (decision rule — does the CI include 0?):\n")
    out.append("| comparison | slice | metric | mean Δ [95% CI] | includes 0? |")
    out.append("|---|---|---|---|---|")
    plan = [
        ("dense_minus_hybrid", "all", "recall10"),
        ("dense_minus_hybrid", "broad", "recall10"),
        ("lexical_minus_hybrid", "all", "recall10"),
        ("dense_minus_hybrid", "all", "mrr"),
        ("dense_minus_lexical", "broad", "recall10"),
        ("dense_minus_lexical", "broad", "mrr"),
    ]
    for name, split, metric in plan:
        ci = boot["paired"][name][split][metric]
        yes = "yes" if ci["includes_zero"] else "**no**"
        out.append(
            f"| {name.replace('_minus_', '−')} | {split} | {metric} "
            f"| {_fmt_ci(ci)} | {yes} |"
        )
    return "\n".join(out)


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


def _tag_summary(tags: dict[str, int]) -> str:
    labels = [("both", "both"), ("dense_only", "D-only"),
              ("lexical_only", "L-only"), ("neither", "neither")]
    parts = [f"{tags[k]} {lab}" for k, lab in labels if tags[k]]
    return " · ".join(parts)


def _md_per_query_depth(per_query: list[dict[str, Any]], pool: int) -> str:
    lines = [
        f"| id | target | gold | dense R@10/@{pool} | lexical R@10/@{pool} "
        f"| hybrid R@10/@{pool} | union R@{pool} | found-by (docs) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in per_query:
        def pair(arm: str) -> str:
            a = r["arms"][arm]
            return f"{a['recall']:.2f}/{a['recall_pool']:.2f}"

        lines.append(
            f"| {r['id']} | {r['target']} | {r['n_gold']} "
            f"| {pair('dense')} | {pair('lexical')} | {pair('hybrid')} "
            f"| {r['union_recall_pool']:.2f} | {_tag_summary(r['tags'])} |"
        )
    return "\n".join(lines)


def _md_diagnostic(per_query: list[dict[str, Any]]) -> str:
    """Decisive queries — where the three arms do not all agree on hit/miss in the top 10."""
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
    bootstrap_b: int = typer.Option(DEFAULT_BOOTSTRAP_B, "--bootstrap", help="Bootstrap resample count."),  # noqa: E501
    seed: int = typer.Option(DEFAULT_SEED, help="Bootstrap RNG seed (for reproducibility)."),
    json_out: Path = typer.Option(None, "--json-out", help="Write the raw results JSON here."),
) -> None:
    results = run(queries, k=k, pool=pool, bootstrap_b=bootstrap_b, seed=seed)
    cfg = results["config"]
    err.log(
        f"corpus={cfg['corpus_n']} | scored={cfg['n_scored']} "
        f"(named={cfg['n_named']}, broad={cfg['n_broad']}) | skipped={cfg['skipped']} "
        f"| k={k} pool={pool} rrf_k={cfg['rrf_k']} | B={bootstrap_b} seed={seed}"
    )

    print("### Aggregate — Recall@10 / MRR by arm and split\n")
    print(_md_aggregate(results["aggregates"]))
    print(f"\n### Recall at depth — Recall@10 vs Recall@{pool}\n")
    print(_md_depth(results["aggregates"], pool))
    print(f"\n### Candidate-set recall & complementarity at pool depth (k={pool})\n")
    print(_md_complementarity(results["aggregates"], pool))
    print("\n### Bootstrap 95% CIs\n")
    print(_md_bootstrap(results["bootstrap"]))
    print("\n### Per-query (all scored, raw)\n")
    print(_md_per_query(results["per_query"]))
    print(f"\n### Per-query depth & complementarity (k={pool})\n")
    print(_md_per_query_depth(results["per_query"], pool))
    print("\n### Decisive queries — arms disagree on top-10 hit/miss\n")
    print(_md_diagnostic(results["per_query"]))

    if json_out:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(results, indent=2))
        err.log(f"[green]wrote raw results[/green] → {json_out}")


if __name__ == "__main__":
    app()
