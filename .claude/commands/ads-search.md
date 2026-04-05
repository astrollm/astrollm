# /ads-search — Search NASA ADS for Papers

Search the NASA ADS database for astronomy papers. Useful during SFT curation, research log writing, and verifying citations.

## Usage
```
/ads-search [query]
```

Examples:
- `/ads-search TRAPPIST-1 atmosphere JWST`
- `/ads-search "dark energy" survey year:2024-2025`
- `/ads-search author:"de Haan" AstroSage`
- `/ads-search bibcode:2411.09012` (look up specific paper)

## Workflow

1. Check that `ADS_API_KEY` is set in the environment (from `.env`)
2. Query the NASA ADS API at `https://api.adsabs.harvard.edu/v1/search/query`
3. Return results in a formatted table:

```
NASA ADS Search: "TRAPPIST-1 atmosphere JWST"
=============================================

 #  Year  Citations  Bibcode              Title
 1  2025       142   2025Natur.123..456L  "No thick satisfies the..."
 2  2024       298   2024ApJ...967L..12G  "Broadband transmission..."
 3  2024       187   2024Natur.618..753Z  "Secondary atmospheres..."
 ...

Showing top 10 of 234 results. Add --all for full list.
```

4. For each result, fetch: bibcode, title, authors (first 3 + et al.), year, citation count, abstract (truncated)

## API Fields
Request these fields from ADS:
```
fl=bibcode,title,author,year,citation_count,abstract,keyword,pubdate,doctype,arxiv_class
```

## Search Syntax Tips
ADS supports rich query syntax:
- `author:"Einstein, A"` — author search
- `year:2024-2025` — date range
- `bibstem:ApJ` — specific journal (ApJ, MNRAS, A&A, AJ)
- `citations(bibcode:XXXX)` — papers citing a specific paper
- `references(bibcode:XXXX)` — references of a specific paper
- `object:"M87"` — papers about a specific astronomical object
- `full:"machine learning" AND full:"exoplanet"` — full-text search

## Implementation
```python
import httpx
import os

ADS_TOKEN = os.environ.get("ADS_API_KEY")

def search_ads(query: str, rows: int = 10) -> dict:
    response = httpx.get(
        "https://api.adsabs.harvard.edu/v1/search/query",
        params={
            "q": query,
            "fl": "bibcode,title,author,year,citation_count,abstract",
            "sort": "citation_count desc",
            "rows": rows,
        },
        headers={"Authorization": f"Bearer {ADS_TOKEN}"},
    )
    return response.json()
```

## Notes
- ADS API is free with registration (5,000 requests/day)
- Get your token at: https://ui.adsabs.harvard.edu/user/settings/token
- Store in `.env` as `ADS_API_KEY`
- Results are sorted by citation count by default (most influential first)
- For bulk harvesting (data pipeline), use `packages/data-pipeline/src/download_arxiv.py` instead
