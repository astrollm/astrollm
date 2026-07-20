"""Authoring harness CLI — turn a curated query into a schema-valid gold example.

Two steps, so the human labeler supplies only judgment and never plumbing:

    # 1. machine assembles the frozen context and writes a worksheet to fill
    uv run python packages/data-pipeline/src/sft/author.py prepare \
        --query "What did JWST measure for the C/O ratio of WASP-39b?" \
        --family lit_qa --partition eval

    # ... the labeler edits only the answer / claims / abstention_reason in the worksheet ...

    # 2. machine validates the filled worksheet and appends it to the gold seed
    uv run python packages/data-pipeline/src/sft/author.py commit \
        --worksheet data/sft/worksheets/lit_qa-eval-1a2b3c4d.yaml

`prepare` calls the frozen retriever (hybrid RRF @ pool=100) and locks the real `retrieved_context`,
`retriever_run_id`, and `corpus_snapshot_hash` into the worksheet. `commit` rejects anything that
fails the schema contract or whose locked corpus hash no longer matches the live corpus. Negatives
(deliberate wrong-citation calibration examples) are seeded with `--negative --negative-type ...`.

Per repo Python rules: progress logs go to stderr; the resulting path/id is printed to stdout.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import typer
import yaml
from context import assemble_context, render_context
from corpus import DEFAULT_CORPUS_PATH, CorpusSnapshot
from paths import DEFAULT_GOLD_PATH, DEFAULT_WORKSHEET_DIR
from rich.console import Console
from schema import (
    AbstentionReason,
    NegativeType,
    Partition,
    TaskFamily,
    validate_record,
)

err = Console(stderr=True)
app = typer.Typer(add_completion=False, help=__doc__)

# Keys the labeler fills; everything else lives under the locked block.
_FILL_KEYS = ("answer", "claims", "abstention_reason", "is_negative", "negative_type")


def _query_slug(query: str) -> str:
    return hashlib.sha1(query.strip().encode("utf-8")).hexdigest()[:8]


def _default_example_id(
    family: TaskFamily, partition: Partition, query: str, negative: bool
) -> str:
    suffix = "-neg" if negative else ""
    return f"{family.value}-{partition.value}-{_query_slug(query)}{suffix}"


def _claim_template(*, abstention: bool, negative: bool) -> dict:
    if abstention:
        # Abstention answers may cite nothing; the labeler explains why context was insufficient.
        return {"text": "", "cited_bibcode": None, "support_span": None, "supported": True}
    return {
        "text": "",
        "cited_bibcode": None,  # REQUIRED for lit_qa/summarization — set to a bibcode from the list
        "support_span": "",
        # A negative deliberately mislabels: at least one claim must be supported=False.
        "supported": not negative,
    }


def _compose_worksheet(data: dict, rendered: str, *, negative: bool) -> str:
    banner = [
        "# " + "=" * 74,
        "# GOLD-SEED WORKSHEET — fill `answer`, `claims`"
        + (", `abstention_reason`" if data["_locked"]["task_family"] == "abstention" else "")
        + " below.",
        "# Do NOT edit anything under `_locked` (retrieved context + provenance are frozen).",
        "# Cite bibcodes only from the retrieved list shown here. supported=False marks a claim",
        "# the cited paper does not actually support"
        + (" (this is a NEGATIVE)." if negative else "."),
        "# " + "=" * 74,
        "#",
        "# RETRIEVED CONTEXT (read-only reference):",
        "#",
    ]
    banner += ["# " + line if line else "#" for line in rendered.splitlines()]
    banner += ["# " + "=" * 74, ""]
    body = yaml.safe_dump(data, sort_keys=False, allow_unicode=True, width=100)
    return "\n".join(banner) + "\n" + body


@app.command()
def prepare(
    query: str = typer.Option(..., help="The curated query to ground an example on."),
    family: TaskFamily = typer.Option(..., help="lit_qa | summarization | abstention."),
    partition: Partition = typer.Option(..., help="Partition: calibration | eval."),
    negative: bool = typer.Option(
        False, "--negative", help="Seed a deliberate wrong-citation negative (calibration only)."
    ),
    negative_type: NegativeType | None = typer.Option(
        None, help="Failure mode for a negative (required with --negative)."
    ),
    abstention_provenance: str | None = typer.Option(
        None, help="Note on why context is insufficient (e.g. 'q21=thin', 'q15 LHS 475b=absent')."
    ),
    example_id: str | None = typer.Option(None, help="Override the auto-generated example_id."),
    corpus: Path = typer.Option(DEFAULT_CORPUS_PATH, help="Frozen corpus snapshot JSONL."),
    out: Path | None = typer.Option(None, help="Worksheet path (default: data/sft/worksheets/)."),
    render_limit: int | None = typer.Option(
        None, help="Render only the top-N abstracts (default: all retrieved)."
    ),
    force: bool = typer.Option(False, "--force", help="Overwrite an existing worksheet."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Assemble + render; do not write."),
) -> None:
    """Assemble the frozen context for a query and write a worksheet for the labeler to fill."""
    if negative:
        if partition is not Partition.CALIBRATION:
            err.log("[red]negatives must target the calibration partition[/red]")
            raise typer.Exit(1)
        if negative_type is None:
            err.log("[red]--negative requires --negative-type[/red]")
            raise typer.Exit(1)
    if family is TaskFamily.ABSTENTION and negative:
        err.log("[red]abstention examples are not negatives; drop --negative[/red]")
        raise typer.Exit(1)

    snapshot = CorpusSnapshot.load(corpus)
    err.log(f"corpus snapshot: {len(snapshot)} abstracts, {snapshot.snapshot_hash}")
    assembled = assemble_context(query, snapshot)
    err.log(f"retrieved {len(assembled.retrieved)} chunks — run {assembled.retriever_run_id}")

    rendered = render_context(assembled, snapshot, limit=render_limit)
    eid = example_id or _default_example_id(family, partition, query, negative)
    is_abstention = family is TaskFamily.ABSTENTION

    provenance = {
        "author": "human",
        "date": datetime.now(UTC).date().isoformat(),
        "retriever_run_id": assembled.retriever_run_id,
        "corpus_snapshot_hash": assembled.corpus_snapshot_hash,
    }
    if abstention_provenance:
        provenance["abstention_provenance"] = abstention_provenance

    data = {
        "_locked": {
            "example_id": eid,
            "task_family": family.value,
            "partition": partition.value,
            "query": query,
            "retrieved_context": [r.model_dump() for r in assembled.retrieved],
            "provenance": provenance,
            "retriever_config": assembled.retriever_config,
        },
        "answer": "",
        "claims": [_claim_template(abstention=is_abstention, negative=negative)],
        "abstention_reason": (AbstentionReason.THIN.value if is_abstention else None),
        "is_negative": negative,
        "negative_type": (negative_type.value if negative_type else None),
    }

    if dry_run:
        err.log("[bold]DRY RUN[/bold] — worksheet not written")
        err.print(rendered)
        print(eid)
        return

    out = out or DEFAULT_WORKSHEET_DIR / f"{eid}.yaml"
    if out.exists() and not force:
        err.log(f"[red]worksheet exists: {out} (use --force to overwrite)[/red]")
        raise typer.Exit(1)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(_compose_worksheet(data, rendered, negative=negative))
    err.log(f"[green]worksheet written[/green] → {out}  (fill answer/claims, then `commit`)")
    print(str(out))


@app.command()
def commit(
    worksheet: Path = typer.Option(..., exists=True, help="A filled worksheet YAML."),
    gold: Path = typer.Option(DEFAULT_GOLD_PATH, help="Gold-seed JSONL to append to."),
    corpus: Path = typer.Option(DEFAULT_CORPUS_PATH, help="Frozen corpus snapshot JSONL."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Validate only; do not append."),
) -> None:
    """Validate a filled worksheet against the schema + corpus and append it to the gold seed."""
    sheet = yaml.safe_load(worksheet.read_text())
    if not isinstance(sheet, dict) or "_locked" not in sheet:
        err.log("[red]worksheet missing the _locked block — was it produced by `prepare`?[/red]")
        raise typer.Exit(1)

    locked = dict(sheet["_locked"])
    locked.pop("retriever_config", None)  # not part of the GoldExample contract; lives in manifest
    record = {**locked, **{k: sheet.get(k) for k in _FILL_KEYS}}

    snapshot = CorpusSnapshot.load(corpus)
    # Stale-corpus guard: the example was authored against the hash locked at prepare time.
    locked_hash = record.get("provenance", {}).get("corpus_snapshot_hash")
    if locked_hash != snapshot.snapshot_hash:
        err.log(
            f"[red]corpus hash drift[/red]: worksheet locked {locked_hash}, "
            f"current corpus {snapshot.snapshot_hash}. Re-prepare against the current corpus."
        )
        raise typer.Exit(1)

    example, errors, warnings = validate_record(record, snapshot.bibcodes)
    for w in warnings:
        err.log(f"[yellow]warning[/yellow]: {w}")
    if errors:
        for e in errors:
            err.log(f"[red]error[/red]: {e}")
        err.log(f"[red]rejected {worksheet.name} — {len(errors)} error(s)[/red]")
        raise typer.Exit(1)
    assert example is not None

    if _id_already_present(gold, example.example_id):
        err.log(
            f"[red]example_id {example.example_id!r} already in {gold} — "
            "set a distinct --example-id at prepare time[/red]"
        )
        raise typer.Exit(1)

    err.log(
        f"[green]valid[/green]: {example.example_id} "
        f"({example.task_family.value}/{example.partition.value}, "
        f"{len(example.claims)} claim(s), negative={example.is_negative})"
    )
    if dry_run:
        err.log("[bold]DRY RUN[/bold] — not appended")
        print(example.example_id)
        return

    gold.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(example.model_dump(mode="json"), ensure_ascii=False)
    with gold.open("a") as fh:
        fh.write(line + "\n")
    err.log(f"[green]appended[/green] → {gold}")
    print(example.example_id)


def _id_already_present(gold: Path, example_id: str) -> bool:
    if not gold.exists():
        return False
    for raw in gold.read_text().splitlines():
        raw = raw.strip()
        if raw and json.loads(raw).get("example_id") == example_id:
            return True
    return False


if __name__ == "__main__":
    app()
