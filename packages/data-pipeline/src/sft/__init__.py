"""Gold-seed authoring harness for the RAG-SFT pilot (Phase A).

Turns a curated query into a schema-valid gold example so the human labeler supplies only
judgment (answers + per-claim labels) and never plumbing (retrieval, provenance, partitioning).

The gold seed NEVER trains — it is split disjointly into a verifier-calibration partition and an
eval-seed partition (see ``docs/research/sft-pilot.md``). The scripts run script-style (sibling
imports, matching the rag/evaluation packages); the modules are:

- ``schema``    — Pydantic contract + validator (the only source of truth for example shape).
- ``corpus``    — the frozen corpus snapshot: hash, bibcode set, abstract lookup, chunk ids.
- ``context``   — frozen retrieval (hybrid RRF @ pool=100) + abstract rendering for the labeler.
- ``author``    — ``prepare`` / ``commit`` CLI: query -> worksheet -> schema-valid JSONL.
- ``partition`` — composition targets, query-level disjointness, manifest construction.
- ``manifest``  — ``build`` / ``status`` CLI: validate the gold JSONL and emit ``manifest.json``.
- ``paths``     — default artifact locations under ``data/sft/``.
"""
