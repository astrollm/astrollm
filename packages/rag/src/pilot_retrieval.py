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


def hybrid_search(query: str, k: int = 10, pool: int = 50) -> list[dict[str, Any]]:
    """Dense + lexical retrieval fused with RRF. Returns top-k ranked records."""
    dense = dense_search(query, pool)
    lexical = lexical_search(query, pool)
    dense_rank, lexical_rank = _ranks(dense), _ranks(lexical)

    fused: dict[str, float] = {}
    for ranks in (dense_rank, lexical_rank):
        for bibcode, rank in ranks.items():
            fused[bibcode] = fused.get(bibcode, 0.0) + 1.0 / (RRF_K + rank)

    ordered = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:k]
    return [
        {
            "bibcode": bibcode,
            "rrf_score": round(score, 6),
            "dense_rank": dense_rank.get(bibcode),
            "lexical_rank": lexical_rank.get(bibcode),
        }
        for bibcode, score in ordered
    ]


app = typer.Typer(add_completion=False, help=__doc__)


@app.command()
def main(
    query: str = typer.Argument(..., help="Free-text query."),
    k: int = typer.Option(10, help="Number of results to return."),
) -> None:
    results = hybrid_search(query, k=k)
    table = Table(title=f"Hybrid retrieval — top {k}")
    table.add_column("#", justify="right")
    table.add_column("bibcode")
    table.add_column("RRF", justify="right")
    table.add_column("dense", justify="right")
    table.add_column("lexical", justify="right")
    for i, r in enumerate(results, 1):
        table.add_row(
            str(i), r["bibcode"], f"{r['rrf_score']:.4f}",
            str(r["dense_rank"] or "—"), str(r["lexical_rank"] or "—"),
        )
    err.print(table)
    for r in results:
        print(f"{r['bibcode']}\t{r['rrf_score']:.6f}")


if __name__ == "__main__":
    app()
