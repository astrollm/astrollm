"""ADS ingestion for the pilot retrieval prototype (weeks 1-2 of V1_FINAL_PLAN).

Fetches abstracts (NOT full text) for exoplanet-atmosphere papers from the NASA ADS
search API and writes them to ``data/raw/`` as JSONL plus a provenance ``manifest.json``.

Scope is deliberately minimal: abstracts only, naive one-chunk-per-abstract downstream,
no fielded/object-aware querying, no SIMBAD expansion (those are weeks 5-10).

Usage:
    uv run python packages/data-pipeline/src/ingest_ads.py --limit 500
    uv run python packages/data-pipeline/src/ingest_ads.py --dry-run

Requires ADS_API_KEY in the environment (or .env). Logs go to stderr; the output
path is printed to stdout.
"""

import json
import os
from datetime import date
from pathlib import Path
from typing import Any

import httpx
import typer
from dotenv import load_dotenv
from rich.console import Console

ADS_SEARCH_URL = "https://api.adsabs.harvard.edu/v1/search/query"
DEFAULT_QUERY = 'abs:"exoplanet atmosphere"'
DEFAULT_FIELDS = "bibcode,title,year,author,abstract"
PAGE_ROWS = 200  # ADS recommends <= 200 rows per request
PIPELINE_VERSION = "pilot-retrieval-0.1.0"

OUTPUT_DIR = Path("data/raw/ads_exoplanet_atmospheres")
ABSTRACTS_FILE = OUTPUT_DIR / "abstracts.jsonl"
MANIFEST_FILE = OUTPUT_DIR / "manifest.json"

err = Console(stderr=True)
app = typer.Typer(add_completion=False, help=__doc__)


def _normalize(doc: dict[str, Any]) -> dict[str, Any] | None:
    """Coerce a raw ADS doc into our flat record. Returns None if unusable."""
    abstract = doc.get("abstract")
    bibcode = doc.get("bibcode")
    if not abstract or not bibcode:
        return None  # abstracts are required for a retrieval corpus
    title = doc.get("title") or []
    year = doc.get("year")
    return {
        "bibcode": bibcode,
        "title": title[0] if isinstance(title, list) and title else (title or ""),
        "year": int(year) if year else None,
        "authors": doc.get("author") or [],
        "abstract": abstract,
    }


def fetch_ads(
    query: str, year_start: int, limit: int, token: str
) -> list[dict[str, Any]]:
    """Page through the ADS search API and return up to ``limit`` normalized records."""
    headers = {"Authorization": f"Bearer {token}"}
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    start = 0
    with httpx.Client(timeout=30.0) as client:
        while len(records) < limit:
            params = {
                "q": query,
                "fq": f"year:[{year_start} TO *]",
                "fl": DEFAULT_FIELDS,
                "rows": min(PAGE_ROWS, limit - len(records)),
                "start": start,
                "sort": "citation_count desc",
            }
            resp = client.get(ADS_SEARCH_URL, headers=headers, params=params)
            resp.raise_for_status()
            body = resp.json()["response"]
            docs = body.get("docs", [])
            total = body.get("numFound", 0)
            if not docs:
                break
            for doc in docs:
                rec = _normalize(doc)
                if rec and rec["bibcode"] not in seen:
                    seen.add(rec["bibcode"])
                    records.append(rec)
            err.log(f"fetched {len(records)}/{limit} (ADS numFound={total})")
            start += len(docs)
            if start >= total:
                break
    return records[:limit]


@app.command()
def main(
    query: str = typer.Option(DEFAULT_QUERY, help="ADS query string."),
    year_start: int = typer.Option(2018, help="Earliest publication year to include."),
    limit: int = typer.Option(500, help="Max number of abstracts to ingest."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print plan; do not call ADS or write."),
) -> None:
    load_dotenv()
    token = os.environ.get("ADS_API_KEY", "").strip()

    if dry_run:
        err.log("[bold]DRY RUN[/bold] — no network calls, no files written")
        err.log(f"  query      : {query}")
        err.log(f"  year_start : {year_start}")
        err.log(f"  limit      : {limit}")
        err.log(f"  output     : {ABSTRACTS_FILE}")
        err.log(f"  ADS_API_KEY: {'set' if token else 'MISSING'}")
        raise typer.Exit(0)

    if not token:
        err.log(
            "[red]ADS_API_KEY is not set.[/red] Add it to .env "
            "(https://ui.adsabs.harvard.edu/user/settings/token), then re-run. "
            "For an offline machinery check, use the fixture corpus instead "
            "(see packages/rag/src/index_corpus.py --input tests/fixtures/pilot/corpus.jsonl)."
        )
        raise typer.Exit(1)

    err.log(f"Querying ADS: {query!r} (year >= {year_start}, limit {limit})")
    records = fetch_ads(query, year_start, limit, token)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with ABSTRACTS_FILE.open("w") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    manifest = {
        "source": "NASA ADS search API",
        "endpoint": ADS_SEARCH_URL,
        "query": query,
        "year_start": year_start,
        "requested_limit": limit,
        "returned_count": len(records),
        "fields": DEFAULT_FIELDS.split(","),
        "chunking": "naive: one chunk per abstract",
        "retrieved_at": date.today().isoformat(),
        "pipeline_version": PIPELINE_VERSION,
    }
    MANIFEST_FILE.write_text(json.dumps(manifest, indent=2) + "\n")

    err.log(f"[green]Wrote {len(records)} abstracts[/green] + manifest")
    print(ABSTRACTS_FILE)


if __name__ == "__main__":
    app()
