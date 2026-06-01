"""The frozen corpus snapshot — the single artifact every gold example is grounded on.

Phase A holds the corpus fixed (see ``docs/research/sft-pilot.md``): the 2,500-abstract ADS
exoplanet-atmosphere slice that the frozen retriever indexes. This module loads that JSONL and
exposes everything the harness and validator need *without* touching the database:

- a reproducible ``snapshot_hash`` over exactly the text that gets indexed (so an example's
  provenance can be checked against the corpus it was authored on);
- the ``bibcodes`` set, for validator rule 6 (every retrieved bibcode must exist here);
- ``abstract`` / ``title`` / ``record`` lookups, to render real abstracts for the labeler; and
- ``chunk_id``, the deterministic id for the one-chunk-per-abstract indexing.

The corpus file's bibcodes are asserted (separately, by the index build) to equal the indexed DB,
so the file is a faithful stand-in for the index when assembling provenance and rendering context.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

# Phase-A corpus: the widened 2,500-abstract slice that the frozen pool=100 index was built from
# (pilot 500 ⊂ widened 2,500). Held fixed for the whole pilot.
DEFAULT_CORPUS_PATH = Path("data/raw/ads_exoplanet_atmospheres_widened/abstracts.jsonl")


def indexed_text(record: dict[str, Any]) -> str:
    """The text that is embedded *and* lexically indexed for a record: ``title. abstract``.

    Mirrors ``index_corpus._embed_text`` exactly so the snapshot hash reflects what the retriever
    actually sees. Kept as a local 3-line copy to avoid importing the heavy rag/torch stack.
    """
    title = (record.get("title") or "").strip()
    abstract = (record.get("abstract") or "").strip()
    return f"{title}. {abstract}".strip(". ").strip()


def chunk_id_for(bibcode: str) -> str:
    """Deterministic chunk id for the one-chunk-per-abstract indexing (section ``abstract``, idx 0).

    The DB's ``chunks.id`` is a non-reproducible SERIAL; this snapshot-derivable id is stable across
    re-index and is what the gold examples record.
    """
    return f"{bibcode}:abstract:0"


class CorpusSnapshot:
    """An immutable view over the frozen corpus file, keyed by bibcode."""

    def __init__(self, records: list[dict[str, Any]], path: Path) -> None:
        by_bibcode: dict[str, dict[str, Any]] = {}
        for rec in records:
            bibcode = rec.get("bibcode")
            if not bibcode:
                raise ValueError(f"corpus record missing bibcode: {rec!r:.120}")
            if bibcode in by_bibcode:
                raise ValueError(f"duplicate bibcode in corpus: {bibcode}")
            by_bibcode[bibcode] = rec
        self.path = path
        self._by_bibcode = by_bibcode
        self._hash = _hash_corpus(by_bibcode)

    @classmethod
    def load(cls, path: Path | str = DEFAULT_CORPUS_PATH) -> CorpusSnapshot:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"corpus snapshot not found: {path}")
        records = [
            json.loads(line) for line in path.read_text().splitlines() if line.strip()
        ]
        if not records:
            raise ValueError(f"corpus snapshot is empty: {path}")
        return cls(records, path)

    @property
    def snapshot_hash(self) -> str:
        """Order-independent ``sha256:<hex>`` over the indexed text of every record."""
        return self._hash

    @property
    def bibcodes(self) -> set[str]:
        return set(self._by_bibcode)

    def __len__(self) -> int:
        return len(self._by_bibcode)

    def __contains__(self, bibcode: object) -> bool:
        return bibcode in self._by_bibcode

    def record(self, bibcode: str) -> dict[str, Any]:
        if bibcode not in self._by_bibcode:
            raise KeyError(f"bibcode not in corpus snapshot: {bibcode}")
        return self._by_bibcode[bibcode]

    def title(self, bibcode: str) -> str:
        return (self.record(bibcode).get("title") or "").strip()

    def abstract(self, bibcode: str) -> str:
        return (self.record(bibcode).get("abstract") or "").strip()

    def chunk_id(self, bibcode: str) -> str:
        return chunk_id_for(bibcode)


def _hash_corpus(by_bibcode: dict[str, dict[str, Any]]) -> str:
    """SHA-256 over ``bibcode\\tindexed_text`` lines sorted by bibcode (order-independent)."""
    h = hashlib.sha256()
    for bibcode in sorted(by_bibcode):
        h.update(bibcode.encode("utf-8"))
        h.update(b"\t")
        h.update(indexed_text(by_bibcode[bibcode]).encode("utf-8"))
        h.update(b"\n")
    return f"sha256:{h.hexdigest()}"
