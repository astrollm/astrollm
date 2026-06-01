"""Frozen context assembly — call the real retriever, render the real abstracts.

The labeler must ground answers on *exactly* what the inference-time retriever returns, so this
module calls the frozen `retrieve()` (hybrid RRF) at the beta stage-1 default **pool=100** and emits
the result as schema `Retrieved` records plus a reproducible `retriever_run_id` /
`corpus_snapshot_hash`. Single-variable discipline: the retrieval config is frozen here as
constants — the harness never lets the labeler change a retrieval knob.

This is the only module in the harness that touches the database / embedding stack, and it does so
behind a lazy import (mirroring the evaluation harness) so the pure modules stay importable offline.
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from corpus import CorpusSnapshot
from schema import Retrieved

# ── Frozen retrieval config (the beta stage-1 default; never varied by the harness) ──
ARM = "hybrid"
POOL = 100  # candidate depth fed to RRF — the beta stage-1 default
K = 100  # context size returned == the full fused pool-100 stage-1 output


def _rag_src() -> Path:
    # packages/data-pipeline/src/sft/context.py → repo_root/packages/rag/src
    return Path(__file__).resolve().parents[4] / "packages" / "rag" / "src"


def _load_retriever():
    """Lazy import of the shared rag module (psycopg2 + sentence-transformers; needs the DB up)."""
    src = str(_rag_src())
    if src not in sys.path:
        sys.path.insert(0, src)
    import pilot_retrieval  # deferred on purpose — heavy import, needs the DB up

    return pilot_retrieval


def retriever_config() -> dict[str, Any]:
    """The frozen retrieval config, with embedding details read from the rag module."""
    pr = _load_retriever()
    return {
        "arm": ARM,
        "k": K,
        "pool": POOL,
        "rrf_k": pr.RRF_K,
        "embed_model": pr.EMBED_MODEL,
        "query_instruction": pr.QUERY_INSTRUCTION,
        "pipeline": "pilot-retrieval-0.1.0",
    }


def config_fingerprint(config: dict[str, Any]) -> str:
    """Short stable fingerprint of a retrieval config (for the run id)."""
    blob = json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:8]


def make_run_id(config: dict[str, Any], corpus_hash: str, timestamp: str) -> str:
    """Provenance handle: ``rrf-pool100-<corpus8>-<config8>-<UTC stamp>``.

    The retrieved context itself is fully reproducible from (query, corpus_hash, config) because
    retrieval is deterministic; the run id just labels *when* a labeling session assembled it.
    """
    corpus8 = corpus_hash.split(":")[-1][:8]
    return f"rrf-pool{config['pool']}-{corpus8}-{config_fingerprint(config)}-{timestamp}"


def utc_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


@dataclass
class AssembledContext:
    """The frozen retriever's output for one query, ready to drop into a worksheet."""

    query: str
    retrieved: list[Retrieved]
    retriever_run_id: str
    corpus_snapshot_hash: str
    retriever_config: dict[str, Any]
    assembled_at: str = field(default="")


def assemble_context(
    query: str, snapshot: CorpusSnapshot, *, timestamp: str | None = None
) -> AssembledContext:
    """Run the frozen retriever for ``query`` and return schema `Retrieved` records + provenance.

    `retrieval_score` is the retriever's fused RRF score; `chunk_id` is the snapshot's deterministic
    one-chunk-per-abstract id. Every returned bibcode is asserted present in the snapshot — a miss
    means the live index and the corpus file have drifted, which would silently corrupt provenance.
    """
    pr = _load_retriever()
    config = retriever_config()
    raw = pr.retrieve(query, arm=ARM, k=K, pool=POOL)

    drift = sorted(r["bibcode"] for r in raw if r["bibcode"] not in snapshot)
    if drift:
        raise RuntimeError(
            f"retriever returned {len(drift)} bibcode(s) absent from the corpus snapshot "
            f"({snapshot.path}); the live index has drifted from the frozen corpus: "
            f"{', '.join(drift[:5])}{' …' if len(drift) > 5 else ''}"
        )

    retrieved = [
        Retrieved(
            bibcode=r["bibcode"],
            chunk_id=snapshot.chunk_id(r["bibcode"]),
            retrieval_score=float(r["score"]),
        )
        for r in raw
    ]
    stamp = timestamp or utc_stamp()
    return AssembledContext(
        query=query,
        retrieved=retrieved,
        retriever_run_id=make_run_id(config, snapshot.snapshot_hash, stamp),
        corpus_snapshot_hash=snapshot.snapshot_hash,
        retriever_config=config,
        assembled_at=stamp,
    )


def render_context(
    assembled: AssembledContext, snapshot: CorpusSnapshot, *, limit: int | None = None
) -> str:
    """Render the retrieved abstracts as plain text for the labeler to read and cite from."""
    rows = assembled.retrieved if limit is None else assembled.retrieved[:limit]
    lines = [
        f"query: {assembled.query}",
        f"retrieved {len(assembled.retrieved)} chunks @ {assembled.retriever_config['arm']} "
        f"pool={assembled.retriever_config['pool']}"
        + (f" (showing top {limit})" if limit and limit < len(assembled.retrieved) else ""),
        "",
    ]
    for rank, r in enumerate(rows, 1):
        lines.append(f"[{rank:>3}] {r.bibcode}  (score {r.retrieval_score:.6f})")
        lines.append(f"      {snapshot.title(r.bibcode)}")
        lines.append(f"      {snapshot.abstract(r.bibcode)}")
        lines.append("")
    return "\n".join(lines)
