"""Index a JSONL abstract corpus for hybrid retrieval.

Embeds each abstract with BGE-small-en-v1.5 and writes:
  - dense vectors into pgvector (papers + chunks tables, one chunk per abstract), and
  - a SQLite FTS5 index for BM25 lexical search.

Naive chunking only (one chunk == one abstract); section-aware chunking is weeks 5-10.

Usage (from repo root):
    # real corpus
    uv run python packages/rag/src/index_corpus.py \
        --input data/raw/ads_exoplanet_atmospheres/abstracts.jsonl
    # offline smoke test
    uv run python packages/rag/src/index_corpus.py --input tests/fixtures/pilot/corpus.jsonl
    # validate parsing only
    uv run python packages/rag/src/index_corpus.py --input <file> --dry-run
"""

import json
import sqlite3
from pathlib import Path
from typing import Any

import typer

# Same-directory import (script dir is on sys.path when run via `python <path>`).
from pilot_retrieval import FTS_DB_PATH, db_connect, embed_passages
from rich.console import Console

err = Console(stderr=True)
app = typer.Typer(add_completion=False, help=__doc__)


def _load_corpus(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _embed_text(rec: dict[str, Any]) -> str:
    """Text used for both dense embedding and lexical indexing: title + abstract."""
    title = (rec.get("title") or "").strip()
    abstract = (rec.get("abstract") or "").strip()
    return f"{title}. {abstract}".strip(". ").strip()


@app.command()
def main(
    input: Path = typer.Option(..., exists=True, help="JSONL abstract corpus to index."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Parse + report; do not embed or write."),
) -> None:
    records = _load_corpus(input)
    texts = [_embed_text(r) for r in records]
    err.log(f"loaded {len(records)} records from {input}")

    if dry_run:
        err.log("[bold]DRY RUN[/bold] — not embedding or writing")
        err.log(f"  example bibcode: {records[0]['bibcode'] if records else '(empty)'}")
        err.log(f"  would index into pgvector + {FTS_DB_PATH}")
        raise typer.Exit(0)

    # ── Dense: embed + load into pgvector (full rebuild for an idempotent pilot run) ──
    embeddings = embed_passages(texts)
    err.log(f"embedded {len(texts)} passages (dim {embeddings.shape[1]})")

    with db_connect() as conn, conn.cursor() as cur:
        cur.execute("TRUNCATE papers RESTART IDENTITY CASCADE")
        for rec, text, emb in zip(records, texts, embeddings, strict=True):
            cur.execute(
                "INSERT INTO papers (bibcode, title, authors, abstract, year) "
                "VALUES (%s, %s, %s, %s, %s) RETURNING id",
                (rec["bibcode"], rec.get("title") or "", rec.get("authors") or [],
                 rec.get("abstract"), rec.get("year")),
            )
            paper_id = cur.fetchone()[0]
            cur.execute(
                "INSERT INTO chunks "
                "(paper_id, section, content, token_count, chunk_index, embedding) "
                "VALUES (%s, 'abstract', %s, %s, 0, %s)",
                (paper_id, text, len(text.split()), emb),
            )
        conn.commit()
    err.log("[green]pgvector index built[/green] (papers + chunks)")

    # ── Lexical: SQLite FTS5 (BM25) ──
    FTS_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    FTS_DB_PATH.unlink(missing_ok=True)
    fts = sqlite3.connect(FTS_DB_PATH)
    try:
        fts.execute("CREATE VIRTUAL TABLE docs USING fts5(bibcode UNINDEXED, content)")
        fts.executemany(
            "INSERT INTO docs (bibcode, content) VALUES (?, ?)",
            [(r["bibcode"], t) for r, t in zip(records, texts, strict=True)],
        )
        fts.commit()
    finally:
        fts.close()
    err.log(f"[green]FTS5 index built[/green] ({FTS_DB_PATH})")
    print(f"indexed {len(records)} abstracts")


if __name__ == "__main__":
    app()
