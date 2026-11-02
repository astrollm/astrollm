"""Astronomy tools for the M1 copilot: a SIMBAD object card and an opt-in live ADS search.

This module implements the *tool side* of the evidence contract in ``docs/V1_PLAN_R2.md``:

* **SIMBAD is display-only.** :func:`simbad_card` returns an :class:`ObjectCard` meant to be
  shown to the user. Nothing here turns a card into citable evidence, and the card carries no
  bibcode.
* **Live ADS is opt-in.** :func:`ads_live_search` raises
  :class:`LiveModeRequired` unless it is called with ``live=True`` -- before it touches the
  token, the client, or the network. Results come back as :class:`LiveDoc`, tagged
  ``source="ads_live"``, and :func:`render_live_docs` keeps a visible ``(live ADS)`` tag in the
  prompt text so grounded and live evidence can never be confused downstream.

The chat CLI imports exactly: ``ObjectCard``, ``simbad_card``, ``render_card``, ``LiveDoc``,
``LiveModeRequired``, ``ads_live_search``, ``render_live_docs``.

Usage:
    uv run python packages/inference/src/tools.py simbad "Vega"
    uv run python packages/inference/src/tools.py ads "JWST transmission spectrum WASP-39b" --live
    uv run python packages/inference/src/tools.py ads "WASP-39b" --live --dry-run

Requires ADS_API_KEY in the environment (or .env) for ``ads``. Logs go to stderr; data (the
rendered card / documents) goes to stdout.
"""

from __future__ import annotations

import math
import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, TypeVar

import httpx
import typer
from rich.console import Console

err = Console(stderr=True)
app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="AstroLLM copilot tools: SIMBAD object card + opt-in live ADS search.",
)

MAX_ALIASES = 8
ADS_FIELDS = ["bibcode", "title", "abstract", "year"]
ADS_MAX_ROWS = 100
LIVE_SOURCE = "ads_live"
NO_ABSTRACT = "(no abstract in ADS)"  # e.g. Springer/Nature records carry none
# astroquery's default SIMBAD mirror order. The first mirror that answers a cheap probe is
# remembered for the process, so an unreachable mirror costs one probe timeout, not one per call.
SIMBAD_MIRRORS = ("simbad.cds.unistra.fr", "simbad.harvard.edu")

T = TypeVar("T")


class LiveModeRequired(RuntimeError):
    """Raised when live ADS results are requested without an explicit ``live=True``.

    This is the evidence-contract guard: grounded mode may cite only the frozen retriever's
    top-k, so live results must never enter a prompt by default or by accident.
    """


# ── SIMBAD object card (display only -- NEVER evidence) ──────────────────────


@dataclass(frozen=True)
class ObjectCard:
    """A SIMBAD object card. Shown to the user; never cited as evidence."""

    queried_name: str
    main_id: str
    ra_deg: float | None
    dec_deg: float | None
    object_type: str | None
    aliases: tuple[str, ...]


_simbad_mirror_lock = threading.Lock()
_simbad_good_mirror: str | None = None


def _adql_literal(text: str) -> str:
    """Escape a value for an ADQL string literal (double any single quote)."""
    return text.replace("'", "''")


def _simbad_query(name: str) -> str:
    """One TAP query returning the object's basics and every identifier (one row per alias)."""
    return (
        "SELECT basic.main_id, basic.ra, basic.dec, basic.otype, ident.id AS alias "
        "FROM ident AS q "
        "JOIN basic ON basic.oid = q.oidref "
        "JOIN ident ON ident.oidref = q.oidref "
        f"WHERE q.id = '{_adql_literal(name)}'"
    )


def _clean(value: Any) -> str | None:
    """Table cell -> stripped str, or None for masked / empty cells."""
    if value is None or bool(getattr(value, "mask", False)):
        return None
    text = " ".join(str(value).split())
    return text or None


def _to_float(value: Any) -> float | None:
    """Table cell -> float, or None for masked / non-finite cells."""
    text = _clean(value)
    if text is None:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def _alias_sort_key(alias: tuple[bool, str]) -> tuple[bool, int, str]:
    # Common names first, then short catalogue ids (HD 172167) before long ones (2MASS J...).
    is_common_name, text = alias
    return (not is_common_name, len(text), text)


def _card_from_table(name: str, table: Any) -> ObjectCard | None:
    """Build a card from a SIMBAD result table; None when the name did not resolve."""
    if table is None or len(table) == 0:
        return None
    first = table[0]
    main_id = _clean(first["main_id"])
    if main_id is None:
        return None
    seen = {main_id.casefold(), name.casefold()}
    aliases: list[tuple[bool, str]] = []  # (is SIMBAD common name, display text)
    for row in table:
        raw = _clean(row["alias"])
        if raw is None:
            continue
        text = raw.removeprefix("NAME ")
        if text.casefold() not in seen:
            seen.add(text.casefold())
            aliases.append((raw.startswith("NAME "), text))
    aliases.sort(key=_alias_sort_key)
    shown = tuple(text for _, text in aliases[:MAX_ALIASES])
    return ObjectCard(
        queried_name=name,
        main_id=main_id,
        ra_deg=_to_float(first["ra"]),
        dec_deg=_to_float(first["dec"]),
        object_type=_clean(first["otype"]),
        aliases=shown,
    )


def _run_with_deadline(fn: Callable[[], T], timeout_s: float) -> T:
    """Run ``fn`` in a daemon thread; raise TimeoutError if it outlives ``timeout_s``.

    astroquery's SIMBAD client has no per-request network timeout, so the deadline is enforced
    here. A timed-out thread is abandoned (daemon), never joined, so it cannot block exit.
    """
    box: dict[str, Any] = {}

    def target() -> None:
        try:
            box["value"] = fn()
        except Exception as exc:  # re-raised in the caller's thread
            box["error"] = exc

    worker = threading.Thread(target=target, daemon=True)
    worker.start()
    worker.join(timeout_s)
    if worker.is_alive():
        raise TimeoutError(f"no answer within {timeout_s:.1f}s")
    if "error" in box:
        raise box["error"]
    return box["value"]


@lru_cache(maxsize=len(SIMBAD_MIRRORS))
def _simbad_client(server: str) -> Any:
    """One astroquery client per mirror, so TAP capabilities are fetched once per process."""
    from astroquery.simbad import SimbadClass  # lazy: keeps ``import tools`` cheap

    client = SimbadClass()
    client.server = server
    return client


def _query_simbad_mirror(server: str, adql: str) -> Any:
    """Run ``adql`` against one SIMBAD mirror."""
    return _simbad_client(server).query_tap(adql)


def _probe_simbad_mirror(server: str, timeout_s: float) -> bool:
    """Cheap reachability check with a real network timeout (astroquery has none)."""
    url = f"https://{server}/simbad/sim-tap/capabilities"
    try:
        return httpx.get(url, timeout=timeout_s, follow_redirects=True).status_code < 500
    except httpx.HTTPError:
        return False


def _pick_simbad_mirror(probe_s: float) -> str | None:
    """The remembered mirror, else the first one that answers a probe (None if all are down).

    Probing first matters: astroquery's TAP client holds a process-wide lock while it fetches
    capabilities, so a query abandoned on a dead mirror would otherwise stall the next one.
    """
    global _simbad_good_mirror
    with _simbad_mirror_lock:
        if _simbad_good_mirror is not None:
            return _simbad_good_mirror
    for server in SIMBAD_MIRRORS:
        if _probe_simbad_mirror(server, probe_s):
            with _simbad_mirror_lock:
                _simbad_good_mirror = server
            return server
        err.print(f"[yellow]SIMBAD {server}: unreachable[/yellow]")
    return None


def simbad_card(name: str, timeout_s: float = 10.0) -> ObjectCard | None:
    """Resolve ``name`` in SIMBAD and return a display-only card, or None if unresolved.

    ``timeout_s`` bounds the whole call (mirror probing on the first call, then the query).
    An unreachable SIMBAD returns None (logged to stderr) instead of crashing the chat.
    """
    global _simbad_good_mirror

    name = " ".join(name.split())
    if not name:
        return None
    started = time.monotonic()
    server = _pick_simbad_mirror(probe_s=max(1.0, timeout_s / 4))
    if server is None:
        err.print(f"[yellow]SIMBAD lookup for {name!r} skipped: no mirror reachable.[/yellow]")
        return None
    remaining = max(1.0, timeout_s - (time.monotonic() - started))
    try:
        table = _run_with_deadline(
            lambda: _query_simbad_mirror(server, _simbad_query(name)), remaining
        )
    except Exception as exc:  # network/TAP errors must not crash the chat
        err.print(f"[yellow]SIMBAD {server}: {type(exc).__name__}: {exc}[/yellow]")
        with _simbad_mirror_lock:
            _simbad_good_mirror = None  # re-probe on the next call
        return None
    return _card_from_table(name, table)


def render_card(card: ObjectCard) -> str:
    """Multi-line plain-text card. Display only -- carries no bibcode, so it cannot be cited."""

    def fmt(value: float | None) -> str:
        return "n/a" if value is None else f"{value:.5f}"

    lines = [
        f"SIMBAD: {card.main_id}",
        f"  queried as : {card.queried_name}",
        f"  type       : {card.object_type or 'n/a'}",
        f"  RA, Dec    : {fmt(card.ra_deg)}, {fmt(card.dec_deg)} deg (ICRS)",
    ]
    if card.aliases:
        lines.append(f"  also known : {'; '.join(card.aliases)}")
    lines.append("  (object card for display; not a citable source)")
    return "\n".join(lines)


# ── Live ADS search (opt-in; contract-guarded) ───────────────────────────────


@dataclass(frozen=True)
class LiveDoc:
    """A live ADS result. Citable only in live mode, and always tagged with its source."""

    bibcode: str
    title: str
    abstract: str
    year: int | None
    source: str = LIVE_SOURCE


def _ads_token() -> str | None:
    """ADS token from ``ADS_API_KEY`` (loading .env if present); never logged."""
    from dotenv import load_dotenv

    load_dotenv()
    return os.environ.get("ADS_API_KEY") or None


def _make_ads_client(token: str, rows: int, timeout_s: float) -> Any:
    """A fresh astroquery ADS client (not the module singleton, so settings never leak)."""
    from astroquery.nasa_ads import ADSClass

    client = ADSClass()
    client.TOKEN = token
    client.ADS_FIELDS = list(ADS_FIELDS)
    client.NROWS = rows
    client.NSTART = 0
    client.SORT = "score desc"
    client.TIMEOUT = timeout_s
    return client


def _first_text(value: Any) -> str:
    """ADS returns title as a list of strings and abstract as a string; normalise to text."""
    if isinstance(value, list | tuple):
        value = value[0] if value else ""
    return " ".join(str(value or "").split())


def _to_year(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _parse_ads_docs(payload: dict[str, Any]) -> list[LiveDoc]:
    docs: list[LiveDoc] = []
    for raw in payload.get("response", {}).get("docs", []):
        bibcode = _first_text(raw.get("bibcode"))
        if not bibcode:
            continue
        docs.append(
            LiveDoc(
                bibcode=bibcode,
                title=_first_text(raw.get("title")),
                abstract=_first_text(raw.get("abstract")),
                year=_to_year(raw.get("year")),
            )
        )
    return docs


def ads_live_search(
    query: str, *, live: bool, rows: int = 10, timeout_s: float = 15.0
) -> list[LiveDoc]:
    """Search NASA ADS live. Raises :class:`LiveModeRequired` unless ``live is True``.

    The guard runs first: with ``live`` falsy there is no token lookup, no client, no network.
    On network/auth/parse errors this logs to stderr and returns ``[]`` so a flaky API never
    crashes the chat -- but the guard error is never swallowed.
    """
    if live is not True:
        raise LiveModeRequired(
            "ads_live_search needs live=True: live ADS results are outside the grounded "
            "evidence contract (V1_PLAN_R2, 'Evidence contract')."
        )
    query = " ".join(query.split())
    rows = max(0, min(rows, ADS_MAX_ROWS))
    if not query or rows == 0:
        return []
    token = _ads_token()
    if token is None:
        err.print("[red]ADS live search skipped: ADS_API_KEY is not set.[/red]")
        return []
    try:
        client = _make_ads_client(token, rows, timeout_s)
        response = client.query_simple(query, get_raw_response=True, cache=False)
        response.raise_for_status()
        return _parse_ads_docs(response.json())
    except Exception as exc:  # network/auth/parse errors must not crash the chat
        status = getattr(getattr(exc, "response", None), "status_code", None)
        suffix = f" (HTTP {status})" if status else ""
        err.print(f"[yellow]ADS live search failed: {type(exc).__name__}{suffix}: {exc}[/yellow]")
        return []


def render_live_docs(docs: list[LiveDoc]) -> str:
    """Prompt text for live results: ``[<bibcode>] (live ADS)`` + title + abstract, per doc.

    Same bibcode-first shape as the grounded context so the citation guard can read both, but
    with the ``(live ADS)`` tag so the source is always visible in the prompt.
    """
    blocks = [f"[{d.bibcode}] (live ADS)\n{d.title}\n{d.abstract or NO_ABSTRACT}\n" for d in docs]
    return "\n".join(blocks)


# ── CLI ──────────────────────────────────────────────────────────────────────


@app.command("simbad")
def simbad_cmd(
    name: str = typer.Argument(..., help='Object name, e.g. "Vega" or "WASP-39".'),
    timeout: float = typer.Option(10.0, help="Total timeout in seconds."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print the query; do not call SIMBAD."),
) -> None:
    """Print a SIMBAD object card (display only; never evidence)."""
    if dry_run:
        err.print(f"[dim]would query SIMBAD mirrors {', '.join(SIMBAD_MIRRORS)} with:[/dim]")
        typer.echo(_simbad_query(" ".join(name.split())))
        return
    card = simbad_card(name, timeout_s=timeout)
    if card is None:
        err.print(f"[yellow]No SIMBAD card for {name!r} (unresolved or unreachable).[/yellow]")
        raise typer.Exit(1)
    typer.echo(render_card(card))


@app.command("ads")
def ads_cmd(
    query: str = typer.Argument(..., help="ADS query string."),
    live: bool = typer.Option(False, "--live", help="Required: opt in to live ADS results."),
    rows: int = typer.Option(10, help="Maximum documents to fetch."),
    timeout: float = typer.Option(15.0, help="Timeout in seconds."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print the request; do not call ADS."),
) -> None:
    """Run a live ADS search (requires --live) and print the prompt rendering."""
    if dry_run:
        err.print(
            f"[dim]would call ADS (live={live}) fields={','.join(ADS_FIELDS)} sort='score desc' "
            f"rows={rows} timeout={timeout}s[/dim]"
        )
        typer.echo(f"q={query}")
        if not live:
            err.print("[yellow]note: without --live this would raise LiveModeRequired.[/yellow]")
        return
    try:
        docs = ads_live_search(query, live=live, rows=rows, timeout_s=timeout)
    except LiveModeRequired as exc:
        err.print(f"[red]{exc}[/red]")
        raise typer.Exit(2) from exc
    err.print(f"[dim]{len(docs)} live ADS doc(s)[/dim]")
    typer.echo(render_live_docs(docs))


if __name__ == "__main__":
    app()
