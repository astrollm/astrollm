"""Pool-depth sweep on the FROZEN 2,500-abstract corpus (PR #7).

Frozen-index ablation — the OPPOSITE of the corpus-widening PR: no re-ingest, no re-embed, no
re-index. The 2,500-abstract index built in PR #6 (corpus frozen at main d84f3ff) is queried as-is.

SINGLE VARIABLE = per-arm candidate POOL depth, swept over {50, 100, 200, 500}. Held fixed and
identical across depths: the corpus, the three arms, the BGE model, chunking, the FTS config,
``k=10``, ``RRF_K=60``, and the fusion/selection code. The fusion primitives (``dense_search``,
``lexical_search``, ``_rrf_fuse``, ``RRF_K``) are IMPORTED from ``pilot_retrieval`` so the
controlled variable cannot drift; the metric/bootstrap helpers come from ``ablation_retrieval`` so
readouts are identical to the frozen pilot/widening harness.

Why a separate harness (not ``ablation_retrieval --pool``): the H8 test needs FUSED recall at FIXED
cuts {10, 50} while the candidate pool VARIES, but ``ablation_retrieval`` ties its deep cut to
``pool`` (recall@pool). This harness decouples the scored cut from the candidate depth so the
candidate-union-vs-fused-top-k GAP (the fusion leak) is measurable at each pool.

Readouts per pool depth:
  - candidate-union recall@pool — the ceiling the fused candidate set offers a stage-2 reranker;
  - FUSED top-10 and top-50 recall (fixed cuts) and the GAP = union - fused (the fusion leak);
  - dense/lexical recall@10/@50 — POOL-INDEPENDENT (a single arm's ranking doesn't depend on the
    pool); reported flat for reference;
  - complementarity (dense-only / lexical-only / both / neither) at pool depth;
  - dense-hybrid and lexical-hybrid paired-bootstrap R@10 CIs WITH leave-one-out at each depth;
  - cost proxy: mean fused candidate-set size (|union of top-pool|) — the input size a stage-2
    reranker would face; see the stage-2 boundary note in pool-sweep.md.

Dual report: (a) all 29 method-fixed; (b) stratified KNOWN-ITEM (named q01-18, scored) vs
BROAD-ENTITY (topical q19-30, coverage measurement) per the a priori partition committed in
``pool_sweep_known_item.yaml``.

Usage (frozen 2,500 index up):
    uv run python packages/evaluation/src/pool_sweep.py \
        --queries packages/evaluation/queries/pilot_exoplanet_atmospheres.yaml \
        --known-item packages/evaluation/queries/pool_sweep_known_item.yaml \
        --json-out docs/research/pool-sweep-results.json
"""

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import typer
import yaml
from rich.console import Console

# Fusion/selection code — the held-fixed controlled variable — imported, never reimplemented.
_RAG_SRC = Path(__file__).resolve().parents[3] / "packages" / "rag" / "src"
sys.path.insert(0, str(_RAG_SRC))
# Metric + bootstrap helpers — imported from the frozen harness so readouts are provably identical.
from ablation_retrieval import (  # noqa: E402
    DEFAULT_BOOTSTRAP_B,
    DEFAULT_SEED,
    _ci,
    _corpus_size,
    _kind,
    _recall_at,
)
from pilot_retrieval import (  # noqa: E402
    ARMS,
    _rrf_fuse,
    dense_search,
    lexical_search,
)

err = Console(stderr=True)
app = typer.Typer(add_completion=False, help=__doc__)

POOLS = (50, 100, 200, 500)  # the swept variable
K = 10  # fixed top-k cut (scored), decoupled from pool
DEEP = 50  # fixed deep cut (scored), decoupled from pool
PARTITIONS = ("known_item", "broad_entity")
TAGS = ("dense_only", "lexical_only", "both", "neither")
# pool=50 all-slice values published in PR #6 (corpus-widening-results.json) — validation target.
PILOT50 = {"dense_r10": 0.529, "lexical_r10": 0.437, "hybrid_r10": 0.592,
           "dense_r50": 0.776, "lexical_r50": 0.810, "hybrid_r50": 0.793, "union": 0.948}


def _read_full_rankings(items: list[dict[str, Any]], corpus_n: int) -> tuple[list, list]:
    """Read each arm's FULL ranking ONCE per scored query (pool-independent); reuse across pools."""
    base, skipped = [], []
    for item in items:
        expected = item.get("expected_bibcodes") or []
        if not expected:
            skipped.append(item["id"])
            continue
        q = item["query"]
        d_rank = {b: i + 1 for i, (b, _) in enumerate(dense_search(q, corpus_n))}
        l_rank = {b: i + 1 for i, (b, _) in enumerate(lexical_search(q, corpus_n))}
        base.append({
            "id": item["id"], "target": item.get("target", ""), "query": q,
            "kind": _kind(item["id"]), "expected": expected,
            "d_rank": d_rank, "l_rank": l_rank,
        })
    return base, skipped


def _score_at_pool(base: list[dict[str, Any]], pool: int, partition: dict[str, str]) -> list[dict]:
    """Score one pool depth — fuse top-`pool` per arm; recall at fixed cuts K/DEEP."""
    rows = []
    for r in base:
        d_pool = {b: rk for b, rk in r["d_rank"].items() if rk <= pool}
        l_pool = {b: rk for b, rk in r["l_rank"].items() if rk <= pool}
        fused = _rrf_fuse(d_pool, l_pool)
        fused_rank = {
            b: i + 1
            for i, (b, _) in enumerate(sorted(fused.items(), key=lambda kv: kv[1], reverse=True))
        }
        arm_rank = {"dense": r["d_rank"], "lexical": r["l_rank"], "hybrid": fused_rank}
        exp = r["expected"]
        arms = {
            a: {"r10": _recall_at(exp, arm_rank[a], K), "r50": _recall_at(exp, arm_rank[a], DEEP)}
            for a in ARMS
        }
        union = set(d_pool) | set(l_pool)
        tags = dict.fromkeys(TAGS, 0)
        for g in exp:
            in_d, in_l = g in d_pool, g in l_pool
            tags["both" if in_d and in_l else "dense_only" if in_d
                 else "lexical_only" if in_l else "neither"] += 1
        rows.append({
            "id": r["id"], "target": r["target"], "kind": r["kind"],
            "partition": partition[r["id"]], "n_gold": len(exp),
            "arms": arms,
            "union_recall": sum(1 for g in exp if g in union) / len(exp),
            "tags": tags, "cand_size": len(union),
        })
    return rows


def _agg_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    if not n:
        return {"n": 0}
    out: dict[str, Any] = {"n": n, "relevant": sum(r["n_gold"] for r in rows)}
    for a in ARMS:
        out[a] = {"r10": sum(r["arms"][a]["r10"] for r in rows) / n,
                  "r50": sum(r["arms"][a]["r50"] for r in rows) / n}
    out["union_recall"] = sum(r["union_recall"] for r in rows) / n
    out["gap10"] = out["union_recall"] - out["hybrid"]["r10"]
    out["gap50"] = out["union_recall"] - out["hybrid"]["r50"]
    out["docs"] = {t: sum(r["tags"][t] for r in rows) for t in TAGS}
    out["mean_cand_size"] = sum(r["cand_size"] for r in rows) / n
    return out


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    agg = {"all": _agg_rows(rows)}
    for p in PARTITIONS:
        agg[p] = _agg_rows([r for r in rows if r["partition"] == p])
    return agg


def _paired(diffs: np.ndarray, seed: int, b: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    ci = _ci(list(diffs), rng, b)
    ci["excludes_zero"] = not (ci["lo"] <= 0.0 <= ci["hi"])
    return ci


def _paired_loo(diffs: list[float], ids: list[str], seed: int, b: int) -> dict[str, Any]:
    """Paired-difference CI plus leave-one-out: never report a bare CI (n=29 is fragile, per #6)."""
    arr = np.asarray(diffs, dtype=float)
    full = _paired(arr, seed, b)
    surviving, hinge = 0, []
    for i in range(len(arr)):
        sub = _paired(np.delete(arr, i), seed, b)  # same seed => comparable LOO subsets
        if sub["excludes_zero"]:
            surviving += 1
        else:
            hinge.append(ids[i])
    full.update({"loo_surviving": surviving, "loo_total": len(arr), "hinge_queries": hinge})
    return full


def _bootstrap(per_pool: dict[int, Any], b: int, seed: int) -> dict[str, Any]:
    """dense-hybrid and lexical-hybrid paired R@10 CIs WITH LOO, at each pool (all-scored slice)."""
    out: dict[str, Any] = {}
    for pool, data in per_pool.items():
        rows = data["rows"]
        ids = [r["id"] for r in rows]
        d_h = [r["arms"]["dense"]["r10"] - r["arms"]["hybrid"]["r10"] for r in rows]
        l_h = [r["arms"]["lexical"]["r10"] - r["arms"]["hybrid"]["r10"] for r in rows]
        out[str(pool)] = {
            "dense_minus_hybrid": _paired_loo(d_h, ids, seed, b),
            "lexical_minus_hybrid": _paired_loo(l_h, ids, seed, b),
        }
    return out


def _validate_pool50(agg_all: dict[str, Any]) -> list[str]:
    """Frozen-harness cross-check: pool=50 must reproduce the published PR #6 all-slice numbers."""
    got = {"dense_r10": agg_all["dense"]["r10"], "lexical_r10": agg_all["lexical"]["r10"],
           "hybrid_r10": agg_all["hybrid"]["r10"], "dense_r50": agg_all["dense"]["r50"],
           "lexical_r50": agg_all["lexical"]["r50"], "hybrid_r50": agg_all["hybrid"]["r50"],
           "union": agg_all["union_recall"]}
    return [
        f"{k}: got {got[k]:.3f} != {v:.3f}"
        for k, v in PILOT50.items()
        if abs(got[k] - v) > 5e-4
    ]


def run(queries_path: Path, known_item_path: Path, pools: tuple[int, ...],
        bootstrap_b: int, seed: int) -> dict[str, Any]:
    spec = yaml.safe_load(queries_path.read_text())
    items = spec["queries"] if isinstance(spec, dict) else spec
    gold = {it["id"]: (it.get("expected_bibcodes") or []) for it in items}

    ki = yaml.safe_load(known_item_path.read_text())
    partition: dict[str, str] = {}
    for qid in ki["known_item"]:
        partition[qid] = "known_item"
    for qid in ki["broad_entity"]:
        partition[qid] = "broad_entity"
    # Enforce the committed list == frozen gold (so it is a record, not a relabel).
    mismatched = [qid for qid, codes in ki["known_item"].items() if codes != gold.get(qid)]
    if mismatched:
        raise ValueError(f"committed known-item list diverges from frozen gold: {mismatched}")

    corpus_n = _corpus_size()
    base, skipped = _read_full_rankings(items, corpus_n)
    for r in base:  # every scored query must be partitioned a priori
        if r["id"] not in partition:
            raise ValueError(f"query {r['id']} missing from the committed partition")

    per_pool: dict[int, Any] = {}
    for pool in pools:
        rows = _score_at_pool(base, pool, partition)
        per_pool[pool] = {"rows": rows, "agg": _aggregate(rows)}

    # Hard guard: the pool=50 reproduction is what certifies that ONLY pool depth varied. If it
    # fails (wrong DB/corpus, or retrieval drift) or pool 50 was not swept, abort BEFORE emitting
    # any results, so a stale/invalid run can never overwrite the committed frozen-index artifact.
    val_errors = (
        _validate_pool50(per_pool[50]["agg"]["all"]) if 50 in per_pool else ["pool=50 not run"]
    )
    if val_errors:
        raise RuntimeError(
            "pool=50 frozen-index validation FAILED — refusing to emit results: "
            f"{val_errors}. Re-point DATABASE_URL/FTS at the frozen 2,500 corpus "
            "(PR #6, main d84f3ff) and include pool 50 in the sweep."
        )
    return {
        "config": {
            "corpus_n": corpus_n, "pools": list(pools), "k": K, "deep": DEEP, "rrf_k": 60,
            "queries_file": str(queries_path), "known_item_file": str(known_item_path),
            "n_scored": len(base), "skipped": skipped,
            "n_known_item": sum(1 for r in base if partition[r["id"]] == "known_item"),
            "n_broad_entity": sum(1 for r in base if partition[r["id"]] == "broad_entity"),
            "bootstrap_B": bootstrap_b, "seed": seed,
            "pool50_validation": "ok (reproduces PR #6 all-slice numbers)",
        },
        "per_pool": {
            str(p): {"agg": d["agg"], "per_query": d["rows"]} for p, d in per_pool.items()
        },
        "bootstrap": _bootstrap(per_pool, bootstrap_b, seed),
    }


# ── Markdown emitters (ready to paste into docs/research/pool-sweep.md) ──


def _md_gap(results: dict[str, Any], slice_key: str) -> str:
    """H8 centerpiece: candidate-union recall vs FUSED top-k recall, and the gap, across depths."""
    lines = [
        "| pool | union R@pool | fused R@10 | fused R@50 | gap@10 (union−fused10) "
        "| gap@50 (union−fused50) |",
        "|---|---|---|---|---|---|",
    ]
    for p in results["config"]["pools"]:
        a = results["per_pool"][str(p)]["agg"][slice_key]
        lines.append(
            f"| {p} | {a['union_recall']:.3f} | {a['hybrid']['r10']:.3f} "
            f"| {a['hybrid']['r50']:.3f} | {a['gap10']:+.3f} | {a['gap50']:+.3f} |"
        )
    return "\n".join(lines)


def _md_arms(results: dict[str, Any], slice_key: str) -> str:
    lines = [
        "| pool | dense R@10 | lexical R@10 | hybrid R@10 "
        "| dense R@50 | lexical R@50 | hybrid R@50 |",
        "|---|---|---|---|---|---|---|",
    ]
    for p in results["config"]["pools"]:
        a = results["per_pool"][str(p)]["agg"][slice_key]
        lines.append(
            f"| {p} | {a['dense']['r10']:.3f} | {a['lexical']['r10']:.3f} "
            f"| {a['hybrid']['r10']:.3f} | {a['dense']['r50']:.3f} "
            f"| {a['lexical']['r50']:.3f} | {a['hybrid']['r50']:.3f} |"
        )
    return "\n".join(lines)


def _md_complementarity(results: dict[str, Any], slice_key: str) -> str:
    lines = [
        "| pool | union R@pool | D-only | L-only | both | neither | mean cand-set size |",
        "|---|---|---|---|---|---|---|",
    ]
    for p in results["config"]["pools"]:
        a = results["per_pool"][str(p)]["agg"][slice_key]
        d = a["docs"]
        lines.append(
            f"| {p} | {a['union_recall']:.3f} | {d['dense_only']} | {d['lexical_only']} "
            f"| {d['both']} | {d['neither']} | {a['mean_cand_size']:.0f} |"
        )
    return "\n".join(lines)


def _md_loo(results: dict[str, Any]) -> str:
    lines = [
        "| pool | comparison | mean Δ [95% CI] | excludes 0? | LOO surviving | hinges on |",
        "|---|---|---|---|---|---|",
    ]
    names = {"dense_minus_hybrid": "dense−hybrid", "lexical_minus_hybrid": "lexical−hybrid"}
    for p in results["config"]["pools"]:
        boot = results["bootstrap"][str(p)]
        for key, label in names.items():
            c = boot[key]
            excl = "**yes**" if c["excludes_zero"] else "no"
            hinge = ",".join(c["hinge_queries"]) if c["hinge_queries"] else "—"
            lines.append(
                f"| {p} | {label} | {c['mean']:+.3f} [{c['lo']:+.3f}, {c['hi']:+.3f}] | {excl} "
                f"| {c['loo_surviving']}/{c['loo_total']} | {hinge} |"
            )
    return "\n".join(lines)


@app.command()
def main(
    queries: Path = typer.Option(..., exists=True, help="YAML query set (frozen 29-label gold)."),
    known_item: Path = typer.Option(..., exists=True, help="Committed a-priori partition YAML."),
    pools: str = typer.Option("50,100,200,500", help="Comma-separated pool depths to sweep."),
    bootstrap_b: int = typer.Option(DEFAULT_BOOTSTRAP_B, "--bootstrap", help="Bootstrap N."),
    seed: int = typer.Option(DEFAULT_SEED, help="Bootstrap RNG seed."),
    json_out: Path = typer.Option(None, "--json-out", help="Write raw results JSON here."),
) -> None:
    pool_tuple = tuple(int(p) for p in pools.split(","))
    try:
        results = run(queries, known_item, pool_tuple, bootstrap_b, seed)
    except (RuntimeError, ValueError) as exc:  # frozen-index / partition guards: fail, never emit
        err.log(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    cfg = results["config"]
    err.log(
        f"corpus={cfg['corpus_n']} | scored={cfg['n_scored']} "
        f"(known_item={cfg['n_known_item']}, broad_entity={cfg['n_broad_entity']}) "
        f"| skipped={cfg['skipped']} | pools={cfg['pools']} | k={cfg['k']} deep={cfg['deep']}"
    )
    err.log(f"pool=50 validation: {cfg['pool50_validation']}")

    print("### (a) All 29 — candidate-union vs fused-top-k gap across depths (H8)\n")
    print(_md_gap(results, "all"))
    print("\n### (a) All 29 — R@10 / R@50 by arm across depths\n")
    print(_md_arms(results, "all"))
    print("\n### (a) All 29 — complementarity across depths (depth-dependent by construction)\n")
    print(_md_complementarity(results, "all"))
    print("\n### (b) KNOWN-ITEM (named, scored) — gap across depths\n")
    print(_md_gap(results, "known_item"))
    print("\n### (b) KNOWN-ITEM (named, scored) — R@10 / R@50 by arm across depths\n")
    print(_md_arms(results, "known_item"))
    print("\n### (b) BROAD-ENTITY (topical, measurement) — gap across depths\n")
    print(_md_gap(results, "broad_entity"))
    print("\n### Per-depth paired CIs WITH leave-one-out (H9, all-scored)\n")
    print(_md_loo(results))

    if json_out:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(results, indent=2))
        err.log(f"[green]wrote raw results[/green] → {json_out}")


if __name__ == "__main__":
    app()
