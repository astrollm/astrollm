"""Hybrid retrieval for the pilot prototype: dense (pgvector) + lexical (SQLite FTS5),
merged with reciprocal-rank fusion (RRF).

Also the shared module for the pilot: DB/embedder helpers live here and are imported by
``index_corpus.py`` and the evaluation harness.

CLI (query in, ranked bibcodes + scores out):
    uv run python packages/rag/src/pilot_retrieval.py "JWST transmission spectrum of WASP-39b"
"""

import os
import re
import sqlite3
from functools import lru_cache
from pathlib import Path
from typing import Any

import psycopg2
import typer
from dotenv import load_dotenv
from pgvector.psycopg2 import register_vector
from rich.console import Console
from rich.table import Table

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
# BGE retrieval: queries get an instruction prefix; passages do not.
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
FTS_DB_PATH = Path("data/processed/pilot_fts.db")
RRF_K = 60  # standard reciprocal-rank-fusion constant

err = Console(stderr=True)


def get_dsn() -> str:
    load_dotenv()
    dsn = os.environ.get("DATABASE_URL")
    if not dsn:
        raise RuntimeError("DATABASE_URL is not set (see .env)")
    return dsn


def db_connect() -> psycopg2.extensions.connection:
    conn = psycopg2.connect(get_dsn())
    register_vector(conn)
    return conn


def get_target_label() -> str:
    """Human-readable host:port/dbname of DATABASE_URL — for destructive-op safety messages."""
    from urllib.parse import urlparse

    u = urlparse(get_dsn())
    return f"{u.hostname or '?'}:{u.port or '?'}/{(u.path or '').lstrip('/') or '?'}"


@lru_cache(maxsize=1)
def _embedder():  # lazy import — torch/sentence-transformers is heavy
    from sentence_transformers import SentenceTransformer

    err.log(f"loading embedder {EMBED_MODEL} …")
    return SentenceTransformer(EMBED_MODEL)


def embed_passages(texts: list[str]):
    return _embedder().encode(texts, normalize_embeddings=True, show_progress_bar=False)


def embed_query(text: str):
    return _embedder().encode(
        QUERY_INSTRUCTION + text, normalize_embeddings=True, show_progress_bar=False
    )


def _fts_match(query: str) -> str:
    """Sanitize a free-text query into an FTS5 MATCH string (OR of quoted tokens).

    Quoting each token avoids FTS5 syntax errors on hyphens/digits (e.g. WASP-39b,
    TRAPPIST-1) and OR maximizes lexical recall — fusion handles precision.
    """
    tokens = re.findall(r"[A-Za-z0-9]+", query.lower())
    return " OR ".join(f'"{t}"' for t in tokens) if tokens else '""'


def dense_search(query: str, k: int) -> list[tuple[str, float]]:
    """Top-k by cosine similarity in pgvector. Returns [(bibcode, similarity)]."""
    vec = embed_query(query)
    sql = (
        "SELECT p.bibcode, 1 - (c.embedding <=> %s::vector) AS sim "
        "FROM chunks c JOIN papers p ON p.id = c.paper_id "
        "ORDER BY c.embedding <=> %s::vector LIMIT %s"
    )
    with db_connect() as conn, conn.cursor() as cur:
        cur.execute(sql, (vec, vec, k))
        return [(bibcode, float(sim)) for bibcode, sim in cur.fetchall()]


def lexical_search(query: str, k: int) -> list[tuple[str, float]]:
    """Top-k by BM25 in SQLite FTS5. Returns [(bibcode, bm25_score)] (lower bm25 = better)."""
    if not FTS_DB_PATH.exists():
        raise FileNotFoundError(f"FTS index missing: {FTS_DB_PATH}. Run index_corpus.py first.")
    conn = sqlite3.connect(FTS_DB_PATH)
    try:
        rows = conn.execute(
            "SELECT bibcode, bm25(docs) AS score FROM docs "
            "WHERE docs MATCH ? ORDER BY score LIMIT ?",
            (_fts_match(query), k),
        ).fetchall()
    finally:
        conn.close()
    return [(bibcode, float(score)) for bibcode, score in rows]


def _ranks(results: list[tuple[str, float]]) -> dict[str, int]:
    return {bibcode: i + 1 for i, (bibcode, _) in enumerate(results)}


ARMS = ("dense", "lexical", "hybrid")


def _rrf_fuse(dense_rank: dict[str, int], lexical_rank: dict[str, int]) -> dict[str, float]:
    """Reciprocal-rank fusion of two rank maps into a {bibcode: rrf_score} dict."""
    fused: dict[str, float] = {}
    for ranks in (dense_rank, lexical_rank):
        for bibcode, rank in ranks.items():
            fused[bibcode] = fused.get(bibcode, 0.0) + 1.0 / (RRF_K + rank)
    return fused


def retrieve(query: str, arm: str = "hybrid", k: int = 10, pool: int = 50) -> list[dict[str, Any]]:
    """Single entry point for the three retrieval arms — the only knob that varies.

    Single-variable discipline: every arm draws candidates from the SAME ``dense_search`` /
    ``lexical_search`` over the SAME corpus snapshot, embeddings, ``pool`` and ``k``. The only
    thing that changes is which arm(s) feed the final ranking:
      - ``dense``   — pgvector / BGE cosine ranking, top-k.
      - ``lexical`` — SQLite FTS5 / BM25 ranking, top-k.
      - ``hybrid``  — dense+lexical fused with RRF over the pool, top-k (the pilot config).

    ``pool`` is the per-arm candidate depth fed to RRF; it is inert for the single arms (their
    top-k is the first k of the pool either way) and is kept identical across arms so the only
    moving part is the arm. Returns top-k records, each with the bibcode, the arm's native
    ``score``, and whichever per-arm ranks apply.
    """
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; expected one of {ARMS}")
    # Each backend must return at least k candidates, or Recall@k/MRR@k under-report
    # (e.g. eval --k 100 with a pool of 50 would never see ranks 51-100).
    pool = max(pool, k)

    if arm == "dense":
        ordered = dense_search(query, pool)[:k]
        return [
            {"bibcode": b, "score": round(s, 6), "dense_rank": i + 1, "lexical_rank": None}
            for i, (b, s) in enumerate(ordered)
        ]
    if arm == "lexical":
        ordered = lexical_search(query, pool)[:k]
        return [
            {"bibcode": b, "score": round(s, 6), "dense_rank": None, "lexical_rank": i + 1}
            for i, (b, s) in enumerate(ordered)
        ]

    # hybrid
    dense_rank = _ranks(dense_search(query, pool))
    lexical_rank = _ranks(lexical_search(query, pool))
    fused = _rrf_fuse(dense_rank, lexical_rank)
    ordered = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:k]
    return [
        {
            "bibcode": bibcode,
            "score": round(score, 6),
            "dense_rank": dense_rank.get(bibcode),
            "lexical_rank": lexical_rank.get(bibcode),
        }
        for bibcode, score in ordered
    ]


def hybrid_search(query: str, k: int = 10, pool: int = 50) -> list[dict[str, Any]]:
    """Back-compat shim for the hybrid arm of :func:`retrieve` (key ``rrf_score``)."""
    return [
        {
            "bibcode": r["bibcode"],
            "rrf_score": r["score"],
            "dense_rank": r["dense_rank"],
            "lexical_rank": r["lexical_rank"],
        }
        for r in retrieve(query, arm="hybrid", k=k, pool=pool)
    ]


app = typer.Typer(add_completion=False, help=__doc__)


@app.command()
def main(
    query: str = typer.Argument(..., help="Free-text query."),
    k: int = typer.Option(10, help="Number of results to return."),
    arm: str = typer.Option("hybrid", help="Retrieval arm: dense | lexical | hybrid."),
) -> None:
    results = retrieve(query, arm=arm, k=k)
    table = Table(title=f"{arm.capitalize()} retrieval — top {k}")
    table.add_column("#", justify="right")
    table.add_column("bibcode")
    table.add_column("score", justify="right")
    table.add_column("dense", justify="right")
    table.add_column("lexical", justify="right")
    for i, r in enumerate(results, 1):
        table.add_row(
            str(i), r["bibcode"], f"{r['score']:.4f}",
            str(r["dense_rank"] or "—"), str(r["lexical_rank"] or "—"),
        )
    err.print(table)
    for r in results:
        print(f"{r['bibcode']}\t{r['score']:.6f}")


if __name__ == "__main__":
    app()
