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

DEFAULT_OUTPUT_DIR = Path("data/raw/ads_exoplanet_atmospheres")
ABSTRACTS_NAME = "abstracts.jsonl"
MANIFEST_NAME = "manifest.json"

err = Console(stderr=True)
app = typer.Typer(add_completion=False, help=__doc__)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load a JSONL abstract corpus into a list of records (order preserved)."""
    records: list[dict[str, Any]] = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _merge_corpora(
    base: list[dict[str, Any]], fetched: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], int]:
    """Union ``base`` (kept verbatim, as a prefix) with new records from ``fetched``.

    Records already present by ``bibcode`` keep their ORIGINAL ``base`` content — so the base
    corpus is a byte-identical, strictly-contained prefix of the result. Returns the merged
    records and the count of ``fetched`` bibcodes that overlapped the base (i.e. were skipped).
    """
    seen = {r["bibcode"] for r in base}
    merged = list(base)
    overlap = 0
    for rec in fetched:
        if rec["bibcode"] in seen:
            overlap += 1
            continue
        seen.add(rec["bibcode"])
        merged.append(rec)
    return merged, overlap


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
    output_dir: Path = typer.Option(
        DEFAULT_OUTPUT_DIR, help="Directory for abstracts.jsonl + manifest.json (NOT the input)."
    ),
    merge_with: Path = typer.Option(
        None,
        "--merge-with",
        help="Existing abstracts.jsonl to union with (kept verbatim as a strict-subset prefix). "
        "Use to widen a corpus without dropping the frozen original.",
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print plan; do not call ADS or write."),
) -> None:
    load_dotenv()
    token = os.environ.get("ADS_API_KEY", "").strip()
    abstracts_file = output_dir / ABSTRACTS_NAME
    manifest_file = output_dir / MANIFEST_NAME

    if dry_run:
        err.log("[bold]DRY RUN[/bold] — no network calls, no files written")
        err.log(f"  query      : {query}")
        err.log(f"  year_start : {year_start}")
        err.log(f"  limit      : {limit}")
        err.log(f"  output     : {abstracts_file}")
        err.log(f"  merge_with : {merge_with or '(none)'}")
        err.log(f"  ADS_API_KEY: {'set' if token else 'MISSING'}")
        raise typer.Exit(0)

    if merge_with:
        if not merge_with.exists():
            err.log(f"[red]--merge-with corpus not found: {merge_with}[/red]")
            raise typer.Exit(1)
        # A merge widens a corpus; refuse to write it back over the frozen pilot slice.
        if output_dir.resolve() == DEFAULT_OUTPUT_DIR.resolve():
            err.log(
                "[red]Refusing to write a merged (widened) corpus into the default pilot dir "
                f"({DEFAULT_OUTPUT_DIR}).[/red] Pass --output-dir to a new location."
            )
            raise typer.Exit(1)

    if not token:
        err.log(
            "[red]ADS_API_KEY is not set.[/red] Add it to .env "
            "(https://ui.adsabs.harvard.edu/user/settings/token), then re-run. "
            "For an offline machinery check, use the fixture corpus instead "
            "(see packages/rag/src/index_corpus.py --input tests/fixtures/pilot/corpus.jsonl)."
        )
        raise typer.Exit(1)

    err.log(f"Querying ADS: {query!r} (year >= {year_start}, limit {limit})")
    fetched = fetch_ads(query, year_start, limit, token)

    # Merge keeps the base corpus a byte-identical, strictly-contained prefix (the widening
    # invariant): re-running with the same query same-day will re-fetch the originals, but we
    # keep the base versions and skip the duplicates so nothing in the original slice is dropped.
    base: list[dict[str, Any]] = []
    overlap = 0
    if merge_with:
        base = _load_jsonl(merge_with)
        records, overlap = _merge_corpora(base, fetched)
        err.log(
            f"merged: base={len(base)} + fetched={len(fetched)} "
            f"(overlap={overlap}, new={len(records) - len(base)}) → {len(records)} total"
        )
    else:
        records = fetched

    output_dir.mkdir(parents=True, exist_ok=True)
    with abstracts_file.open("w") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    manifest = {
        "source": "NASA ADS search API",
        "endpoint": ADS_SEARCH_URL,
        "query": query,
        "year_start": year_start,
        "requested_limit": limit,
        "fetched_count": len(fetched),
        "returned_count": len(records),
        "fields": DEFAULT_FIELDS.split(","),
        "chunking": "naive: one chunk per abstract",
        "retrieved_at": date.today().isoformat(),
        "pipeline_version": PIPELINE_VERSION,
    }
    if merge_with:
        manifest["merged_with"] = {
            "base_corpus": str(merge_with),
            "base_count": len(base),
            "fetched_overlap": overlap,
            "added_from_fetch": len(records) - len(base),
            "strict_superset_of_base": True,
            "note": (
                "Widening confound: size and recency move together here — a deeper "
                "citation-ranked slice (same query/year_start) admits more recent, "
                "lower-citation papers. This does NOT isolate size vs recency; the controlled "
                "variable is the retrieval METHOD, not the corpus axis."
            ),
        }
    manifest_file.write_text(json.dumps(manifest, indent=2) + "\n")

    err.log(f"[green]Wrote {len(records)} abstracts[/green] + manifest → {output_dir}")
    print(abstracts_file)


if __name__ == "__main__":
    app()
