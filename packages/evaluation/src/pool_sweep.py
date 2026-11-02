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

D1 mode (``--d1-cut``; pre-registered decision D1 in docs/V1_PLAN_R2.md): score the fused hybrid
pool-100 ranking at cuts {10, 20, 30, 50}, rendered exactly as the model sees it
(``sft/context.render_context``), token-counted with the Qwen3.5 tokenizer under a 16,384-token
context budget, overflow handled by whole-abstract bottom-drop, recall scored on the DELIVERED
context (decision logic in ``context_cut.py``). Gated: pool=50 must reproduce PR #6 and pool=100's
hybrid R@10/R@50 must reproduce ``docs/research/pool-sweep-results.json`` exactly, or nothing is
emitted:
    uv run python packages/evaluation/src/pool_sweep.py --d1-cut \
        --queries packages/evaluation/queries/pilot_exoplanet_atmospheres.yaml \
        --known-item packages/evaluation/queries/pool_sweep_known_item.yaml \
        --cuts 10,20,30,50 --budget 16384 --tokenizer Qwen/Qwen3.5-4B \
        --json-out docs/research/d1-context-cut-results.json \
        --md-out docs/research/d1-context-cut.md

Serving parity (``--verify-serving``): ``retrieve(arm="hybrid", k=100, pool=100)`` must equal the
harness's fused top-100 on every scored gold query; exits 1 on any mismatch.
"""

import json
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import typer
import yaml
from rich.console import Console

# Fusion/selection code — the held-fixed controlled variable — imported, never reimplemented.
_RAG_SRC = Path(__file__).resolve().parents[3] / "packages" / "rag" / "src"
sys.path.insert(0, str(_RAG_SRC))
import context_cut  # noqa: E402  (pure D1 decision logic; same directory as this script)

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
# The committed PR #7 artifact — its pool=100 hybrid R@10/R@50 is the D1 validation target.
COMMITTED_SWEEP = Path("docs/research/pool-sweep-results.json")
D1_POOL = 100  # the beta stage-1 default (== sft/context.POOL, asserted at run time)
_SFT_SRC = Path(__file__).resolve().parents[3] / "packages" / "data-pipeline" / "src" / "sft"


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


def _fused_ranking(
    d_rank: dict[str, int], l_rank: dict[str, int], pool: int
) -> tuple[dict[str, int], dict[str, int], list[tuple[str, float]]]:
    """Fuse each arm's top-`pool` with RRF → (dense pool, lexical pool, ranked [(bibcode, rrf)]).

    Same ordering as ``pilot_retrieval.retrieve(arm="hybrid")`` given the same per-arm pools
    (stable sort over the dense-then-lexical insertion order of ``_rrf_fuse``).
    """
    d_pool = {b: rk for b, rk in d_rank.items() if rk <= pool}
    l_pool = {b: rk for b, rk in l_rank.items() if rk <= pool}
    fused = _rrf_fuse(d_pool, l_pool)
    return d_pool, l_pool, sorted(fused.items(), key=lambda kv: kv[1], reverse=True)


def _score_at_pool(base: list[dict[str, Any]], pool: int, partition: dict[str, str]) -> list[dict]:
    """Score one pool depth — fuse top-`pool` per arm; recall at fixed cuts K/DEEP."""
    rows = []
    for r in base:
        d_pool, l_pool, ordered = _fused_ranking(r["d_rank"], r["l_rank"], pool)
        fused_rank = {b: i + 1 for i, (b, _) in enumerate(ordered)}
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


def _validate_pool100(agg: dict[str, Any], committed_path: Path) -> list[str]:
    """D1 cross-check: pool=100 hybrid R@10/R@50 must reproduce the committed PR #7 artifact
    EXACTLY (same arithmetic, same order), on the all / known-item / broad-entity slices."""
    if not committed_path.exists():
        return [f"committed artifact missing: {committed_path}"]
    committed = json.loads(committed_path.read_text())["per_pool"].get(str(D1_POOL))
    if committed is None:
        return [f"committed artifact has no pool={D1_POOL} block"]
    errors = []
    for slice_key in ("all", *PARTITIONS):
        for cut in ("r10", "r50"):
            want = committed["agg"][slice_key]["hybrid"][cut]
            got = agg[slice_key]["hybrid"][cut]
            if got != want:
                errors.append(f"{slice_key} hybrid {cut}: got {got!r} != committed {want!r}")
    return errors


def _load_gold(
    queries_path: Path, known_item_path: Path
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Frozen gold items + the committed a-priori partition (asserted equal to the gold)."""
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
    return items, partition


def run(queries_path: Path, known_item_path: Path, pools: tuple[int, ...],
        bootstrap_b: int, seed: int) -> dict[str, Any]:
    items, partition = _load_gold(queries_path, known_item_path)
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


# ── D1: serving context cut (k) under the context token budget ──


def _load_sft() -> Any:
    """Lazy import of the frozen SFT context module (render_context is reused, never rewritten)."""
    if str(_SFT_SRC) not in sys.path:
        sys.path.insert(0, str(_SFT_SRC))
    import context as sft_context  # deferred: only the D1 mode needs it

    return sft_context


def _load_token_counter(name: str) -> tuple[Callable[[str], int], dict[str, Any]]:
    """Qwen3.5 tokenizer (tokenizer files only, no weights) → (counter, provenance)."""
    from huggingface_hub import try_to_load_from_cache
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name)
    cached = try_to_load_from_cache(name, "tokenizer_config.json")
    revision = Path(cached).parent.name if isinstance(cached, str) else None
    info = {"name": name, "revision": revision, "class": type(tok).__name__,
            "vocab_size": len(tok), "add_special_tokens": False}
    return (lambda text: len(tok.encode(text, add_special_tokens=False))), info


def _assembled_for(sft_context: Any, query: str, ranking: list[tuple[str, float]],
                   snapshot: Any, cfg: dict[str, Any]) -> Any:
    """An AssembledContext identical to what ``assemble_context`` builds for this ranking
    (``retrieve`` rounds the fused score to 6 dp; ``render_context`` prints it with ``:.6f``)."""
    retrieved = [
        sft_context.Retrieved(bibcode=b, chunk_id=snapshot.chunk_id(b),
                              retrieval_score=float(round(score, 6)))
        for b, score in ranking
    ]
    return sft_context.AssembledContext(
        query=query, retrieved=retrieved,
        retriever_run_id=sft_context.make_run_id(cfg, snapshot.snapshot_hash, "d1-cut"),
        corpus_snapshot_hash=snapshot.snapshot_hash, retriever_config=cfg,
        assembled_at="d1-cut",
    )


def _cut_rows(qid: str, expected: list[str], assembled: Any, snapshot: Any, sft_context: Any,
              count: Callable[[str], int], cuts: tuple[int, ...], budget: int,
              log_drops: bool) -> dict[str, dict[str, Any]]:
    """Apply the D1 overflow policy at every cut for one query's ranking."""
    bibs = [r.bibcode for r in assembled.retrieved]
    cache: dict[int, str] = {}

    def render_top(n: int) -> str:
        if n not in cache:
            cache[n] = sft_context.render_context(assembled, snapshot, limit=n)
        return cache[n]

    out: dict[str, dict[str, Any]] = {}
    for k in cuts:
        d = context_cut.deliver(bibs, render_top, count, k, budget)
        out[str(k)] = {
            "n_rendered": d.n_rendered, "n_delivered": d.n_delivered,
            "rendered_tokens": d.rendered_tokens, "delivered_tokens": d.delivered_tokens,
            "overflow": d.overflow, "n_dropped": len(d.dropped), "dropped": list(d.dropped),
            "rendered_recall": context_cut.recall(expected, bibs[: d.n_rendered]),
            "delivered_recall": context_cut.recall(expected, d.delivered),
        }
        if d.overflow and log_drops:
            err.log(
                f"[yellow]overflow[/yellow] {qid} k={k}: {d.rendered_tokens} > {budget} tokens "
                f"→ dropped {len(d.dropped)} whole abstract(s) from the bottom "
                f"(delivered {d.n_delivered}, {d.delivered_tokens} tokens): {', '.join(d.dropped)}"
            )
    return out


def _d1_aggregate(per_query: list[dict[str, Any]], cuts: tuple[int, ...],
                  key: str = "cuts") -> dict[str, Any]:
    agg: dict[str, Any] = {}
    for k in cuts:
        slices = {"all": [q[key][str(k)] for q in per_query]}
        for p in PARTITIONS:
            slices[p] = [q[key][str(k)] for q in per_query if q["partition"] == p]
        agg[str(k)] = {name: context_cut.summarize(rows) for name, rows in slices.items()}
    return agg


def _select(agg: dict[str, Any], cuts: tuple[int, ...]) -> dict[str, Any]:
    sel = context_cut.select_k(
        {k: agg[str(k)]["all"]["delivered_recall"] for k in cuts},
        {k: agg[str(k)]["all"]["fit_rate"] for k in cuts},
    )
    return {"k": sel.k, "branch": sel.branch, "reference_k": context_cut.REFERENCE_K,
            "reference_recall": sel.reference_recall, "recall_gap": sel.recall_gap,
            "literal_reading_k": sel.literal_k, "tolerance": context_cut.TOLERANCE,
            "min_fit_rate": context_cut.MIN_FIT_RATE, "slice": "all", "reason": sel.reason}


def run_d1(queries_path: Path, known_item_path: Path, cuts: tuple[int, ...], budget: int,
           tokenizer_name: str, committed_path: Path, corpus_path: Path | None) -> dict[str, Any]:
    """D1 measurement. Gates first (pool=50 vs PR #6, pool=100 vs the committed artifact,
    live index == corpus snapshot); nothing is computed against the tokenizer until all pass."""
    if context_cut.REFERENCE_K not in cuts:
        raise ValueError(f"cuts must include the reference cut {context_cut.REFERENCE_K}")
    import pilot_retrieval  # already loaded above; module handle

    sft_context = _load_sft()
    if (sft_context.ARM, sft_context.POOL) != ("hybrid", D1_POOL):
        raise RuntimeError(f"sft/context.py frozen config drifted: {sft_context.ARM} "
                           f"pool={sft_context.POOL} (D1 measures hybrid pool={D1_POOL})")
    from corpus import DEFAULT_CORPUS_PATH, CorpusSnapshot

    items, partition = _load_gold(queries_path, known_item_path)
    corpus_n = _corpus_size()
    base, skipped = _read_full_rankings(items, corpus_n)
    for r in base:
        if r["id"] not in partition:
            raise ValueError(f"query {r['id']} missing from the committed partition")

    # ── Hard gates: certify the (possibly rebuilt) index equals the frozen one ──
    gate_aggs = {p: _aggregate(_score_at_pool(base, p, partition)) for p in (50, D1_POOL)}
    errors = [f"pool=50 {e}" for e in _validate_pool50(gate_aggs[50]["all"])]
    errors += [f"pool={D1_POOL} {e}" for e in _validate_pool100(gate_aggs[D1_POOL],
                                                                 committed_path)]
    snapshot = CorpusSnapshot.load(corpus_path or DEFAULT_CORPUS_PATH)
    for store, idx in sft_context.indexed_bibcodes(pilot_retrieval).items():
        if idx != snapshot.bibcodes:
            errors.append(f"live {store} index ({len(idx)}) != corpus snapshot "
                          f"({len(snapshot.bibcodes)}) at {snapshot.path}")
    if errors:
        raise RuntimeError("D1 frozen-index validation FAILED — refusing to emit results: "
                           f"{errors}")
    gates = {
        "pool50": "ok (reproduces PR #6 all-slice numbers)",
        "pool100": "ok (hybrid R@10/R@50 equal the committed artifact on all/known/broad)",
        "pool50_all": {a: gate_aggs[50]["all"][a] for a in ARMS}
        | {"union_recall": gate_aggs[50]["all"]["union_recall"]},
        "pool100_hybrid": {s: gate_aggs[D1_POOL][s]["hybrid"] for s in ("all", *PARTITIONS)},
        "index_equals_snapshot": True,
    }

    count, tok_info = _load_token_counter(tokenizer_name)
    cfg = sft_context.retriever_config()
    per_query: list[dict[str, Any]] = []
    serving: list[dict[str, Any]] = []
    gate_rows = {r["id"]: r for r in _score_at_pool(base, D1_POOL, partition)}
    for r in base:
        _, _, ordered = _fused_ranking(r["d_rank"], r["l_rank"], D1_POOL)
        assembled = _assembled_for(sft_context, r["query"], ordered[:D1_POOL], snapshot, cfg)
        rows = _cut_rows(r["id"], r["expected"], assembled, snapshot, sft_context, count,
                         cuts, budget, log_drops=True)
        # The D1 ranking IS the gated ranking: pre-drop recall at 10/50 == the pool-100 rows.
        for cut, key in ((K, "r10"), (DEEP, "r50")):
            if str(cut) in rows and rows[str(cut)]["rendered_recall"] != (
                    gate_rows[r["id"]]["arms"]["hybrid"][key]):
                raise RuntimeError(f"{r['id']}: D1 ranking diverges from the gated pool-100 "
                                   f"ranking at cut {cut}")
        per_query.append({"id": r["id"], "target": r["target"], "partition": partition[r["id"]],
                          "n_gold": len(r["expected"]), "cuts": rows})

        # Diagnostic (not the pre-registered measurement): what the SERVING call returns.
        live = pilot_retrieval.retrieve(r["query"], arm="hybrid", k=D1_POOL, pool=D1_POOL)
        live_bibs = [x["bibcode"] for x in live]
        harness_bibs = [b for b, _ in ordered[:D1_POOL]]
        first_div = next((i + 1 for i, (a, b) in enumerate(zip(live_bibs, harness_bibs))
                          if a != b), None)
        live_ctx = sft_context.AssembledContext(
            query=r["query"],
            retrieved=[sft_context.Retrieved(bibcode=x["bibcode"],
                                             chunk_id=snapshot.chunk_id(x["bibcode"]),
                                             retrieval_score=float(x["score"])) for x in live],
            retriever_run_id="d1-serving-diagnostic", corpus_snapshot_hash=snapshot.snapshot_hash,
            retriever_config=cfg,
        )
        serving.append({
            "id": r["id"], "partition": partition[r["id"]],
            "dense_rows_returned": len(dense_search(r["query"], D1_POOL)),
            "identical_top100": live_bibs == harness_bibs, "first_divergent_rank": first_div,
            "top100_overlap": len(set(live_bibs) & set(harness_bibs)),
            "cuts": _cut_rows(r["id"], r["expected"], live_ctx, snapshot, sft_context, count,
                              cuts, budget, log_drops=False),
        })

    agg = _d1_aggregate(per_query, cuts)
    serving_agg = _d1_aggregate(serving, cuts)
    with pilot_retrieval.db_connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT '[1]'::vector")  # loads the vector library so its GUCs exist
        cur.execute("SHOW hnsw.ef_search")
        ef_search = cur.fetchone()[0]
        cur.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
        pgvector_version = cur.fetchone()[0]
    return {
        "config": {
            "decision": "D1 (docs/V1_PLAN_R2.md, pre-registered constants)",
            "generated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "cuts": list(cuts), "budget": budget,
            "preregistered": cuts == context_cut.CUTS and budget == context_cut.BUDGET,
            "prompt_reserve": context_cut.PROMPT_RESERVE,
            "answer_allowance": context_cut.ANSWER_ALLOWANCE,
            "max_seq_length_stated": context_cut.MAX_SEQ_LENGTH,
            "ranking": f"fused hybrid RRF, pool={D1_POOL} per arm, full-depth exact arm rankings "
                       "truncated to pool (the gated pool-sweep ranking)",
            "rendering": "sft/context.render_context(assembled, snapshot, limit=n) — includes the "
                         "'query:' and 'retrieved … pool=100' header lines",
            "token_count": "len(tokenizer.encode(rendering, add_special_tokens=False)); context "
                           "rendering only (system prompt + question reserved separately)",
            "overflow_policy": "drop whole abstracts from the bottom until the rendering fits; "
                               "never truncate inside an abstract; drops logged per query/cut",
            "recall": "fraction of gold bibcodes in the delivered context (== pool_sweep "
                      "_recall_at for a top-n prefix)",
            "tokenizer": tok_info, "retriever_config": cfg, "corpus_n": corpus_n,
            "corpus_snapshot": str(snapshot.path), "corpus_snapshot_hash": snapshot.snapshot_hash,
            "queries_file": str(queries_path), "known_item_file": str(known_item_path),
            "committed_sweep": str(committed_path),
            "n_scored": len(base), "skipped": skipped,
            "n_known_item": sum(1 for q in per_query if q["partition"] == "known_item"),
            "n_broad_entity": sum(1 for q in per_query if q["partition"] == "broad_entity"),
        },
        "gates": gates,
        "aggregate": agg,
        "selection": _select(agg, cuts),
        "per_query": per_query,
        "serving_path_diagnostic": {
            "note": "NOT the pre-registered measurement. Same rendering/budget/policy applied to "
                    "what pilot_retrieval.retrieve(arm='hybrid', k=100, pool=100) returns live — "
                    "the call sft/context.assemble_context makes.",
            "pgvector_version": pgvector_version, "hnsw_ef_search": ef_search,
            "n_identical_top100": sum(1 for s in serving if s["identical_top100"]),
            "n_queries": len(serving),
            "aggregate": serving_agg,
            "selection_if_applied": _select(serving_agg, cuts),
            "per_query": serving,
        },
    }


def _md_d1_table(agg: dict[str, Any], cuts: tuple[int, ...]) -> str:
    lines = [
        "| k | delivered recall (all) | known-item | broad-entity | median tokens | p95 tokens "
        "| % queries overflowing | mean abstracts dropped |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for k in cuts:
        a = agg[str(k)]
        lines.append(
            f"| {k} | {a['all']['delivered_recall']:.3f} "
            f"| {a['known_item']['delivered_recall']:.3f} "
            f"| {a['broad_entity']['delivered_recall']:.3f} "
            f"| {a['all']['median_rendered_tokens']:,.0f} | {a['all']['p95_rendered_tokens']:,.0f} "
            f"| {100 * a['all']['overflow_rate']:.1f}% | {a['all']['mean_dropped']:.2f} |"
        )
    return "\n".join(lines)


def _md_d1(results: dict[str, Any]) -> str:
    cfg, gates, sel = results["config"], results["gates"], results["selection"]
    cuts = tuple(cfg["cuts"])
    tok = cfg["tokenizer"]
    g100 = gates["pool100_hybrid"]
    g50 = gates["pool50_all"]
    diag = results["serving_path_diagnostic"]
    k_line = (f"**Candidate per pre-registered rule: k = {sel['k']}** — branch "
              f"`{sel['branch']}`: {sel['reason']}.")
    if sel["literal_reading_k"] != sel["k"]:
        k_line += (f" (Reading in which k = {sel['reference_k']} may win the first branch "
                   f"against itself: k = {sel['literal_reading_k']}.)")
    drops = [
        f"| {q['id']} | {k} | {c['rendered_tokens']:,} | {c['n_dropped']} "
        f"| {c['delivered_tokens']:,} |"
        for k in cuts for q in results["per_query"]
        for c in [q["cuts"][str(k)]] if c["overflow"]
    ]
    out = [
        "# D1 — Serving Context Cut Under the Context Budget",
        "",
        f"Generated {cfg['generated_at'][:10]} by `packages/evaluation/src/pool_sweep.py --d1-cut` "
        "(raw: [`d1-context-cut-results.json`](d1-context-cut-results.json)). Decision D1 of "
        "[Plan Revision 2](../V1_PLAN_R2.md): score the fused hybrid pool-100 ranking at cuts "
        f"{{{', '.join(map(str, cuts))}}}, rendered exactly as the model sees it, under a "
        f"{cfg['budget']:,}-token context budget; overflow drops whole abstracts from the bottom; "
        "recall is scored on the delivered context.",
        "",
        "## Setup",
        "",
        f"- Ranking: {cfg['ranking']}; RRF_K = {cfg['retriever_config']['rrf_k']}; corpus "
        f"{cfg['corpus_n']:,} abstracts (`{cfg['corpus_snapshot_hash'][:19]}…`).",
        f"- Rendering: {cfg['rendering']}.",
        f"- Tokens: `{tok['name']}` tokenizer (revision `{tok['revision']}`), "
        "`add_special_tokens=False`, context rendering only; system prompt + question reserved "
        f"separately ({cfg['prompt_reserve']:,}), answer allowance {cfg['answer_allowance']:,}.",
        f"- Gold: the frozen {cfg['n_scored']}-query set ({cfg['n_known_item']} known-item, "
        f"{cfg['n_broad_entity']} broad-entity; unscored: {', '.join(cfg['skipped']) or '—'}).",
        "",
        "## Validation gates (all passed before any D1 number was computed)",
        "",
        f"- pool=50 reproduces PR #6: dense R@10 {g50['dense']['r10']:.3f}, lexical R@10 "
        f"{g50['lexical']['r10']:.3f}, hybrid R@10 {g50['hybrid']['r10']:.3f}, hybrid R@50 "
        f"{g50['hybrid']['r50']:.3f}, union {g50['union_recall']:.3f}.",
        f"- pool=100 hybrid R@10 / R@50 equal `{cfg['committed_sweep']}` exactly: all "
        f"{g100['all']['r10']:.3f} / {g100['all']['r50']:.3f}, known-item "
        f"{g100['known_item']['r10']:.3f} / {g100['known_item']['r50']:.3f}, broad-entity "
        f"{g100['broad_entity']['r10']:.3f} / {g100['broad_entity']['r50']:.3f}.",
        "- Live pgvector and FTS bibcode sets equal the corpus snapshot.",
        "",
        "## Result",
        "",
        _md_d1_table(results["aggregate"], cuts),
        "",
        "Token columns are the **rendered top-k (pre-drop)** context tokens over all scored "
        "queries (p95: linear interpolation). Known-item / broad-entity columns are delivered "
        "recall on those partitions (broad-entity is a coverage measurement, not true recall). "
        "Max rendered tokens: "
        + ", ".join(f"k={k} {results['aggregate'][str(k)]['all']['max_rendered_tokens']:,}"
                    for k in cuts)
        + f" (budget {cfg['budget']:,}).",
        "",
        k_line,
        "",
        "### Overflow log (whole abstracts dropped from the bottom)",
        "",
    ]
    if drops:
        out += ["| query | k | rendered tokens | dropped | delivered tokens |",
                "|---|---|---|---|---|", *drops]
    else:
        out.append("No query overflowed the budget at any cut.")
    sd = diag["aggregate"]
    out += [
        "",
        "## Serving-path diagnostic (not the pre-registered measurement)",
        "",
        "The same rendering, budget and overflow policy applied to what "
        "`pilot_retrieval.retrieve(arm=\"hybrid\", k=100, pool=100)` returns live (the call "
        f"`sft/context.assemble_context` makes). pgvector {diag['pgvector_version']}; the dense "
        "arm runs exact kNN (index scan disabled per query, so `hnsw.ef_search` = "
        f"{diag['hnsw_ef_search']} does not bound it): top-100 identical to the harness ranking "
        f"on {diag['n_identical_top100']}/{diag['n_queries']} queries.",
        "",
        "| k | serving delivered recall (all) | harness delivered recall (all) "
        "| serving % overflowing |",
        "|---|---|---|---|",
        *[
            f"| {k} | {sd[str(k)]['all']['delivered_recall']:.3f} "
            f"| {results['aggregate'][str(k)]['all']['delivered_recall']:.3f} "
            f"| {100 * sd[str(k)]['all']['overflow_rate']:.1f}% |"
            for k in cuts
        ],
        "",
    ]
    return "\n".join(out)



def verify_serving(queries_path: Path, known_item_path: Path,
                   pool: int = D1_POOL) -> dict[str, Any]:
    """Train/serve parity: the live serving call ``retrieve(arm="hybrid", k=pool, pool=pool)``
    (what ``sft/context.assemble_context`` calls) must return exactly the harness's fused top-pool
    (bibcode order and 6-dp score) on every scored gold query."""
    import pilot_retrieval  # already loaded above; module handle

    items, _ = _load_gold(queries_path, known_item_path)
    base, _ = _read_full_rankings(items, _corpus_size())
    mismatches = []
    for r in base:
        _, _, ordered = _fused_ranking(r["d_rank"], r["l_rank"], pool)
        want = [(b, round(s, 6)) for b, s in ordered[:pool]]
        live = pilot_retrieval.retrieve(r["query"], arm="hybrid", k=pool, pool=pool)
        got = [(x["bibcode"], x["score"]) for x in live]
        if got != want:
            first = next((i + 1 for i, (a, b) in enumerate(zip(got, want)) if a != b),
                         min(len(got), len(want)) + 1)
            mismatches.append({"id": r["id"], "first_divergent_rank": first,
                               "n_live": len(got), "n_harness": len(want)})
    return {"pool": pool, "k": pool, "n_queries": len(base),
            "n_identical": len(base) - len(mismatches), "mismatches": mismatches}


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
    d1_cut: bool = typer.Option(
        False, "--d1-cut", help="Run the D1 serving-context-cut measurement instead of the sweep."
    ),
    cuts: str = typer.Option("10,20,30,50", help="[D1] Comma-separated context cuts k."),
    budget: int = typer.Option(context_cut.BUDGET, help="[D1] Context token budget per query."),
    tokenizer: str = typer.Option("Qwen/Qwen3.5-4B", help="[D1] HF tokenizer for token counts."),
    committed: Path = typer.Option(
        COMMITTED_SWEEP, help="[D1] Committed pool-sweep artifact (pool=100 validation target)."
    ),
    corpus: Path = typer.Option(None, help="[D1] Corpus snapshot JSONL (default: sft corpus)."),
    md_out: Path = typer.Option(None, "--md-out", help="[D1] Write the markdown report here."),
    verify_serving_flag: bool = typer.Option(
        False, "--verify-serving",
        help="Check retrieve(hybrid, k=100, pool=100) == the harness fused top-100 on every "
        "scored gold query (train/serve parity); exit 1 on any mismatch.",
    ),
) -> None:
    if verify_serving_flag:
        res = verify_serving(queries, known_item)
        err.log(f"serving parity @ pool=k={res['pool']}: {res['n_identical']}/{res['n_queries']} "
                "queries identical (bibcode order + 6-dp fused score)")
        for m in res["mismatches"]:
            err.log(f"[red]mismatch[/red] {m['id']}: first divergent rank "
                    f"{m['first_divergent_rank']} (live {m['n_live']} rows, harness "
                    f"{m['n_harness']})")
        if res["mismatches"]:
            raise typer.Exit(1)
        return
    if d1_cut:
        _main_d1(queries, known_item, tuple(int(c) for c in cuts.split(",")), budget, tokenizer,
                 committed, corpus, json_out, md_out)
        return
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


def _main_d1(queries: Path, known_item: Path, cuts: tuple[int, ...], budget: int, tokenizer: str,
             committed: Path, corpus: Path | None, json_out: Path | None,
             md_out: Path | None) -> None:
    try:
        results = run_d1(queries, known_item, cuts, budget, tokenizer, committed, corpus)
    except (RuntimeError, ValueError) as exc:  # frozen-index / partition guards: fail, never emit
        err.log(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    cfg, sel = results["config"], results["selection"]
    err.log(f"gates: {results['gates']['pool50']}; {results['gates']['pool100']}")
    err.log(f"tokenizer={cfg['tokenizer']['name']}@{cfg['tokenizer']['revision']} "
            f"budget={cfg['budget']} cuts={cfg['cuts']} scored={cfg['n_scored']}")
    print(_md_d1_table(results["aggregate"], cuts))
    print(f"\ncandidate per pre-registered rule: k = {sel['k']} [{sel['branch']}] — {sel['reason']}"
          + (f" (literal reading: k = {sel['literal_reading_k']})"
             if sel["literal_reading_k"] != sel["k"] else ""))
    diag = results["serving_path_diagnostic"]
    if diag["n_identical_top100"] != diag["n_queries"]:
        err.log(f"[yellow]serving-path diagnostic:[/yellow] retrieve(k=100, pool=100) matches "
                f"the harness top-100 on {diag['n_identical_top100']}/{diag['n_queries']} "
                f"queries (pgvector {diag['pgvector_version']}, hnsw.ef_search="
                f"{diag['hnsw_ef_search']})")
    if json_out:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(results, indent=2))
        err.log(f"[green]wrote raw D1 results[/green] → {json_out}")
    if md_out:
        md_out.parent.mkdir(parents=True, exist_ok=True)
        md_out.write_text(_md_d1(results))
        err.log(f"[green]wrote D1 report[/green] → {md_out}")


if __name__ == "__main__":
    app()
