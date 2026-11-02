"""Copilot v0 (M1) — a grounded chat CLI over base Qwen3.5-9B with a bibcode citation guard.

Grounded mode (the default and the only evaluated path; evidence contract in docs/V1_PLAN_R2.md):

    verbatim query ─► frozen hybrid RRF @ pool=100 (pilot_retrieval.retrieve)
                   ─► top-k (D1) rendered by sft/context.render_context, exactly as D1 measured it
                   ─► D1 overflow policy (context_cut.deliver: bottom-drop whole abstracts)
                   ─► [system, user] ─► llama-server /v1/chat/completions (no-think)
                   ─► D4 parse + citation guard (citation.py); one regeneration on a violation
                   ─► answer on stdout; one JSONL transcript line per turn

Turns are independent (no conversation memory): each answer's evidence is exactly its own top-k.
SIMBAD object cards (``/simbad`` in chat, ``--simbad`` on ask) are display-only — never added to
the context or the allowlist. ``--live`` (flagged experiment) appends live ADS results after the
grounded context, unions their bibcodes into the allowlist, and logs to a separate directory.

Usage (pilot Postgres up, llama-server on LLM_BASE_URL):
    uv run python packages/inference/src/copilot.py ask "JWST CO2 detection in WASP-39b"
    uv run python packages/inference/src/copilot.py ask "..." --dry-run    # payload only, no LLM
    uv run python packages/inference/src/copilot.py chat --tag citation

Env (.env): LLM_BASE_URL (default http://127.0.0.1:8080/v1), LLM_MODEL (default qwen3.5-9b),
LLM_TEMPERATURE (default 0.2), DATABASE_URL (retrieval), ADS_API_KEY (live mode only).
Logs go to stderr; the answer (or, with --dry-run, the request payload) goes to stdout.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from types import ModuleType
from typing import Any

import httpx
import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.markup import escape
from rich.prompt import Prompt

_SRC_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[3]
_SFT_SRC = REPO_ROOT / "packages" / "data-pipeline" / "src" / "sft"
_EVAL_SRC = REPO_ROOT / "packages" / "evaluation" / "src"
for _p in (_SRC_DIR, _SFT_SRC, _EVAL_SRC):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# The pure D1 overflow policy, the frozen renderer and the D4 guard are imported, never rewritten.
from citation import (  # noqa: E402
    GuardResult,
    ParsedAnswer,
    check_citations,
    parse_answer,
    strip_think,
)
from context import (  # noqa: E402
    ARM,
    POOL,
    AssembledContext,
    K,
    _load_retriever,
    indexed_bibcodes,
    make_run_id,
    render_context,
    retriever_config,
    utc_stamp,
)
from context_cut import BUDGET, deliver  # noqa: E402
from corpus import DEFAULT_CORPUS_PATH, CorpusSnapshot  # noqa: E402
from schema import Retrieved  # noqa: E402

err = Console(stderr=True)
app = typer.Typer(add_completion=False, no_args_is_help=True, help=__doc__)

DEFAULT_LLM_BASE_URL = "http://127.0.0.1:8080/v1"
DEFAULT_LLM_MODEL = "qwen3.5-9b"
DEFAULT_TEMPERATURE = 0.2
DEFAULT_MAX_TOKENS = 2048
LLM_TIMEOUT_S = 300.0
TOKENIZE_TIMEOUT_S = 30.0
TRANSCRIPT_DIR = REPO_ROOT / "data" / "eval" / "transcripts"
LIVE_TRANSCRIPT_DIR = TRANSCRIPT_DIR / "live"
LIVE_ROWS = 10
LIVE_SEPARATOR = "=== Live ADS results (unfrozen) ==="
CONTEXT_INTRO = "Retrieved abstracts (the only evidence you may use):"

SYSTEM_PROMPT = """\
You are AstroLLM, an astronomy research copilot. Answer the user's question using ONLY the \
retrieved paper abstracts in their message. Do not use outside knowledge, even when you are \
confident.

Each abstract starts with a line such as "[  3] 2021ApJ...123L..45X  (score 0.032787)": a rank \
number in brackets, the paper's 19-character ADS bibcode, and a retrieval score. The title and \
abstract follow.

Citations:
- Cite by bibcode, never by rank number: write the bibcode in square brackets exactly as it \
appears in the context, e.g. [2021ApJ...123L..45X].
- Every factual claim needs a citation to an abstract that supports it, at the end of the sentence \
that makes the claim.
- Cite only bibcodes that appear in the context. Never invent, alter or complete a bibcode.
- You may end with one line "Sources: " followed by the cited bibcodes, comma-separated, and \
nothing else on that line.

If the abstracts are insufficient to answer, reply with "ABSTAIN: " and one reason code, \
optionally followed by at most one sentence (40 words or fewer), and nothing else:
- thin: the abstracts touch the topic but lack the specific information asked for.
- contradictory: the abstracts disagree and give no basis for choosing between them.
- off_topic: the question is outside what these abstracts cover.
- absent: the specific object, paper or result asked about does not appear in the abstracts.
Example: "ABSTAIN: absent. None of the abstracts discuss the planet named in the question."

Either answer with citations or abstain, never both. Start directly with the answer or the \
ABSTAIN line: no preamble, no markdown headers, and do not think aloud or describe your \
reasoning."""
SYSTEM_PROMPT_SHA = hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest()[:12]


class Tag(StrEnum):
    """Astronomy error taxonomy (docs/V1_FINAL_PLAN.md, 'Astronomy-Specific Error Taxonomy')."""

    citation = "citation"
    object_identity = "object_identity"
    unit_system = "unit_system"
    coordinate_epoch = "coordinate_epoch"
    catalog_semantic = "catalog_semantic"
    literature_timeline = "literature_timeline"
    database_boundary = "database_boundary"
    tool = "tool"


# ── Pure helpers (no I/O) ────────────────────────────────────────────────────


def build_user_message(context: str, query: str, live_context: str | None = None) -> str:
    """The user turn: grounded context, optional live block after a separator, then the question."""
    parts = [CONTEXT_INTRO, context]
    if live_context:
        parts += [LIVE_SEPARATOR, live_context]
    parts.append(f"Question: {query}")
    return "\n\n".join(parts)


def build_messages(
    context: str, query: str, live_context: str | None = None
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_message(context, query, live_context)},
    ]


def chat_request_body(
    model: str, messages: list[dict[str, str]], temperature: float, max_tokens: int
) -> dict[str, Any]:
    """OpenAI-compatible body. ``enable_thinking=False`` is honoured by llama-server --jinja."""
    return {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }


def corrective_message(guard: GuardResult) -> str:
    """The single regeneration prompt, naming what the guard rejected."""
    parts: list[str] = []
    if guard.violations:
        parts.append(
            "Your answer cited bibcodes that are not in the provided context: "
            + ", ".join(guard.violations)
            + "."
        )
    if not guard.uncited_ok:
        parts.append(
            "Your answer made claims without citing any bibcode from the provided context."
        )
    parts.append(
        "Rewrite your answer: use only the provided sources or abstain. Cite every factual claim "
        "with a bibcode from the context, in square brackets, exactly as written there. If the "
        "context is insufficient, reply with ABSTAIN: followed by one reason code (thin, "
        "contradictory, off_topic or absent) and at most one sentence."
    )
    return " ".join(parts)


def final_status(first: GuardResult, second: GuardResult | None) -> str:
    """pass | pass_after_regen | flagged."""
    if first.status == "pass":
        return "pass"
    if second is not None and second.status == "pass":
        return "pass_after_regen"
    return "flagged"


def server_root(base_url: str) -> str:
    """``http://host:port/v1`` → ``http://host:port`` (llama-server's non-OpenAI endpoints)."""
    root = base_url.rstrip("/")
    return root[: -len("/v1")] if root.endswith("/v1") else root


def transcript_path(directory: Path, session_id: str, day: str) -> Path:
    return directory / f"{day}-{session_id[:8]}.jsonl"


def resolve_temperature(cli_value: float | None) -> float:
    if cli_value is not None:
        return cli_value
    env = os.environ.get("LLM_TEMPERATURE")
    return float(env) if env else DEFAULT_TEMPERATURE


def retag_last_turn(path: Path, tag: str) -> bool:
    """Set ``tag`` on the last transcript line (chat ``/tag``). False if there is no turn yet."""
    if not path.exists():
        return False
    lines = path.read_text().splitlines()
    if not lines:
        return False
    record = json.loads(lines[-1])
    record["tag"] = tag
    lines[-1] = json.dumps(record, ensure_ascii=False)
    path.write_text("\n".join(lines) + "\n")
    return True


# ── Token counting (llama-server /tokenize; chars/4 fallback) ────────────────


class TokenCounter:
    """Counts tokens with the served model's tokenizer; falls back to ``len(text)//4``."""

    def __init__(self, base_url: str, client: httpx.Client) -> None:
        self.url = f"{server_root(base_url)}/tokenize"
        self.client = client
        self.fallback = False
        self._cache: dict[str, int] = {}

    @property
    def source(self) -> str:
        return "chars/4 (fallback)" if self.fallback else f"llama-server {self.url}"

    def __call__(self, text: str) -> int:
        if text in self._cache:
            return self._cache[text]
        n: int | None = None
        if not self.fallback:
            try:
                r = self.client.post(self.url, json={"content": text}, timeout=TOKENIZE_TIMEOUT_S)
                r.raise_for_status()
                n = len(r.json()["tokens"])
            except (httpx.HTTPError, KeyError, ValueError) as exc:
                self.fallback = True
                err.print(
                    f"[yellow]WARNING: {self.url} unavailable ({type(exc).__name__}); "
                    "counting tokens as len(text)//4 — the budget is approximate.[/yellow]"
                )
        if n is None:
            n = len(text) // 4
        self._cache[text] = n
        return n


# ── Retrieval + context ──────────────────────────────────────────────────────


@dataclass
class Grounding:
    """One query's frozen retrieval: the fused pool ranking as an AssembledContext."""

    assembled: AssembledContext  # all `pool` ranked docs (render_context limits to the top-n)
    k: int

    @property
    def pool_bibcodes(self) -> list[str]:
        return [r.bibcode for r in self.assembled.retrieved]


@dataclass
class BuiltContext:
    text: str
    delivered: list[str]  # bibcodes delivered, rank order (the guard allowlist in grounded mode)
    dropped: list[str]  # top-k bibcodes dropped by the budget, rank order
    tokens: int


class GroundedRetriever:
    """The frozen retriever, loaded once per session with the index==snapshot integrity check."""

    def __init__(self, corpus_path: Path) -> None:
        self.snapshot = CorpusSnapshot.load(corpus_path)
        self._pr: ModuleType | None = None

    def _retriever(self) -> ModuleType:
        if self._pr is None:
            pr = _load_retriever()
            if not pr.FTS_DB_PATH.is_absolute():  # rag uses a cwd-relative path
                pr.FTS_DB_PATH = REPO_ROOT / pr.FTS_DB_PATH
            snap = self.snapshot.bibcodes
            for store, idx in indexed_bibcodes(pr).items():
                if idx != snap:
                    raise RuntimeError(
                        f"live {store} index ({len(idx)} docs) does not match the corpus snapshot "
                        f"({len(snap)} docs, {self.snapshot.path}); re-index or fix DATABASE_URL."
                    )
            self._pr = pr
        return self._pr

    def retrieve(self, query: str, k: int, pool: int) -> Grounding:
        """Verbatim query → fused hybrid ranking of depth ``pool`` (≥ k). No rewriting."""
        pr = self._retriever()
        raw = pr.retrieve(query, arm=ARM, k=max(k, pool), pool=pool)
        if not raw:
            raise RuntimeError("retriever returned no results (index being rebuilt?) — retry")
        config = {**retriever_config(), "k": k, "pool": pool}
        stamp = utc_stamp()
        assembled = AssembledContext(
            query=query,
            retrieved=[
                Retrieved(
                    bibcode=r["bibcode"],
                    chunk_id=self.snapshot.chunk_id(r["bibcode"]),
                    retrieval_score=float(r["score"]),
                )
                for r in raw
            ],
            retriever_run_id=make_run_id(config, self.snapshot.snapshot_hash, stamp),
            corpus_snapshot_hash=self.snapshot.snapshot_hash,
            retriever_config=config,
            assembled_at=stamp,
        )
        return Grounding(assembled=assembled, k=k)


def build_grounded_context(
    grounding: Grounding,
    snapshot: CorpusSnapshot,
    count_tokens: Callable[[str], int],
    budget: int,
) -> BuiltContext:
    """Top-k rendered by ``render_context(assembled, snapshot, limit=n)`` under the D1 policy."""
    cache: dict[int, str] = {}

    def render_top(n: int) -> str:
        if n not in cache:
            cache[n] = render_context(grounding.assembled, snapshot, limit=n)
        return cache[n]

    d = deliver(grounding.pool_bibcodes, render_top, count_tokens, grounding.k, budget)
    return BuiltContext(
        text=d.text, delivered=list(d.delivered), dropped=list(d.dropped), tokens=d.delivered_tokens
    )


@dataclass
class LiveContext:
    text: str | None
    bibcodes: list[str]
    dropped_budget: list[str]
    dropped_no_abstract: list[str]
    dropped_duplicate: list[str]
    tokens: int


def build_live_context(
    tools: ModuleType,
    query: str,
    grounded: BuiltContext,
    count_tokens: Callable[[str], int],
    budget: int,
) -> LiveContext:
    """Live ADS docs (abstract-less and already-grounded ones excluded), fit into the budget left
    after the grounded context — grounded evidence always has priority."""
    docs = tools.ads_live_search(query, live=True, rows=LIVE_ROWS)
    seen = set(grounded.delivered)
    no_abstract = [d.bibcode for d in docs if not (d.abstract or "").strip()]
    duplicate = [d.bibcode for d in docs if d.bibcode in seen and d.bibcode not in no_abstract]
    usable = [d for d in docs if (d.abstract or "").strip() and d.bibcode not in seen]
    room = budget - grounded.tokens - count_tokens(LIVE_SEPARATOR)
    if not usable or room < 1:
        return LiveContext(None, [], [d.bibcode for d in usable], no_abstract, duplicate, 0)
    d = deliver(
        [x.bibcode for x in usable],
        lambda n: tools.render_live_docs(usable[:n]),
        count_tokens,
        len(usable),
        room,
    )
    text = d.text if d.n_delivered else None
    return LiveContext(
        text, list(d.delivered), list(d.dropped), no_abstract, duplicate,
        d.delivered_tokens if text else 0,
    )


# ── LLM call ─────────────────────────────────────────────────────────────────


@dataclass
class Completion:
    text: str  # think-stripped content
    had_think: bool
    latency_ms: int
    usage: dict[str, Any] = field(default_factory=dict)
    finish_reason: str | None = None


def chat_completion(client: httpx.Client, base_url: str, body: dict[str, Any]) -> Completion:
    t0 = time.perf_counter()
    r = client.post(f"{base_url.rstrip('/')}/chat/completions", json=body, timeout=LLM_TIMEOUT_S)
    r.raise_for_status()
    data = r.json()
    choice = data["choices"][0]
    message = choice.get("message") or {}
    clean, had_think = strip_think(message.get("content") or "")
    had_think = had_think or bool((message.get("reasoning_content") or "").strip())
    return Completion(
        text=clean.strip(),
        had_think=had_think,
        latency_ms=round((time.perf_counter() - t0) * 1000),
        usage=data.get("usage") or {},
        finish_reason=choice.get("finish_reason"),
    )


# ── Session ──────────────────────────────────────────────────────────────────


@dataclass
class Settings:
    k: int
    pool: int
    budget: int
    max_tokens: int
    temperature: float
    base_url: str
    model: str
    live: bool
    dry_run: bool
    tag: str | None
    corpus: Path


def _load_tools() -> ModuleType | None:
    try:
        import tools  # lazy: the SIMBAD/ADS layer is optional for grounded mode
    except ImportError:
        err.print("[yellow]tools layer not available yet[/yellow]")
        return None
    return tools


class Session:
    def __init__(self, settings: Settings) -> None:
        self.s = settings
        self.id = uuid.uuid4().hex
        self.turn = 0
        self.client = httpx.Client()
        self.count = TokenCounter(settings.base_url, self.client)
        self.retriever = GroundedRetriever(settings.corpus)
        self.tools: ModuleType | None = None
        self.live = False
        if settings.live:
            self.tools = _load_tools()
            if self.tools is None or not hasattr(self.tools, "ads_live_search"):
                err.print("[yellow]--live: live ADS unavailable; running grounded.[/yellow]")
            else:
                self.live = True
        directory = LIVE_TRANSCRIPT_DIR if self.live else TRANSCRIPT_DIR
        day = datetime.now(UTC).date().isoformat()
        self.path = transcript_path(directory, self.id, day)

    @property
    def mode(self) -> str:
        return "live" if self.live else "grounded"

    def show_simbad(self, name: str) -> None:
        """Display-only SIMBAD card: never enters the context or the allowlist."""
        tools = self.tools or _load_tools()
        if tools is None:
            return
        self.tools = tools
        card = tools.simbad_card(name)
        if card is None:
            err.print(f"[yellow]No SIMBAD match for {escape(name)!r}.[/yellow]")
            return
        err.print(tools.render_card(card), markup=False, highlight=False)

    def _ask_llm(self, messages: list[dict[str, str]]) -> Completion:
        body = chat_request_body(self.s.model, messages, self.s.temperature, self.s.max_tokens)
        return chat_completion(self.client, self.s.base_url, body)

    def run_turn(self, query: str, tag: str | None = None) -> dict[str, Any] | None:
        """One grounded (or live) turn. Returns the transcript record (None on --dry-run)."""
        t0 = time.perf_counter()
        s = self.s
        grounding = self.retriever.retrieve(query, s.k, s.pool)
        ctx = build_grounded_context(grounding, self.retriever.snapshot, self.count, s.budget)
        live: LiveContext | None = None
        if self.live and self.tools is not None:
            live = build_live_context(self.tools, query, ctx, self.count, s.budget)
        allowlist = list(ctx.delivered) + (live.bibcodes if live else [])
        messages = build_messages(ctx.text, query, live.text if live else None)

        summary = (
            f"mode={self.mode} k={s.k} delivered={len(ctx.delivered)} dropped={len(ctx.dropped)} "
            f"context_tokens={ctx.tokens}/{s.budget} counter={self.count.source}"
        )
        if live is not None:
            summary += (
                f" live={len(live.bibcodes)} live_dropped_budget={len(live.dropped_budget)} "
                f"live_dropped_no_abstract={len(live.dropped_no_abstract)}"
            )
        if ctx.dropped:
            err.print(
                f"[yellow]budget: dropped {len(ctx.dropped)} abstract(s) from the bottom of the "
                f"top-{s.k}: {', '.join(ctx.dropped)}[/yellow]",
                highlight=False,
            )
        if s.dry_run:
            err.print(f"[bold]DRY RUN[/bold] — no LLM call. {summary}", highlight=False)
            body = chat_request_body(s.model, messages, s.temperature, s.max_tokens)
            print(json.dumps(body, ensure_ascii=False, indent=2), flush=True)
            return None

        first = self._ask_llm(messages)
        parsed1 = parse_answer(first.text)
        guard1 = check_citations(parsed1, set(allowlist))
        second: Completion | None = None
        parsed2: ParsedAnswer | None = None
        guard2: GuardResult | None = None
        if guard1.status == "violation":
            err.print(
                f"[yellow]citation guard: violation (not in context: "
                f"{escape(', '.join(guard1.violations)) or '-'}; cites nothing: "
                f"{not guard1.uncited_ok}) — regenerating once[/yellow]",
                highlight=False,
            )
            retry = [
                *messages,
                {"role": "assistant", "content": first.text},
                {"role": "user", "content": corrective_message(guard1)},
            ]
            second = self._ask_llm(retry)
            parsed2 = parse_answer(second.text)
            guard2 = check_citations(parsed2, set(allowlist))
        status = final_status(guard1, guard2)
        final, parsed, guard = (
            (second, parsed2, guard2) if second is not None else (first, parsed1, guard1)
        )
        assert final is not None and parsed is not None and guard is not None
        self.turn += 1
        a = grounding.assembled
        record: dict[str, Any] = {
            "ts": datetime.now(UTC).isoformat(timespec="seconds"),
            "session_id": self.id,
            "turn": self.turn,
            "mode": self.mode,
            "model": s.model,
            "base_url": s.base_url,
            "k": s.k,
            "pool": s.pool,
            "budget": s.budget,
            "query": query,
            "allowlist": allowlist,
            "pool_bibcodes": grounding.pool_bibcodes,
            "dropped_bibcodes": ctx.dropped,
            "delivered_n": len(ctx.delivered),
            "dropped_n": len(ctx.dropped),
            "context_tokens": ctx.tokens,
            "token_counter": self.count.source,
            "temperature": s.temperature,
            "max_tokens": s.max_tokens,
            "answer": final.text,
            "first_answer": first.text if second is not None else None,
            "parsed": parsed.to_dict(),
            "first_parsed": parsed1.to_dict() if second is not None else None,
            "guard": {
                "status": status,
                "violations": guard.violations,
                "first_violations": guard1.violations if second is not None else None,
                "uncited_ok": guard.uncited_ok,
                "regenerated": second is not None,
                "abstain_with_citation": guard.abstain_with_citation,
            },
            "had_think": first.had_think or (second.had_think if second else False),
            "tag": tag,
            "latency_ms": round((time.perf_counter() - t0) * 1000),
            "llm_latency_ms": [first.latency_ms] + ([second.latency_ms] if second else []),
            "usage": [first.usage] + ([second.usage] if second else []),
            "finish_reason": final.finish_reason,
            "system_prompt_sha": SYSTEM_PROMPT_SHA,
            "retriever_run_id": a.retriever_run_id,
            "corpus_snapshot_hash": a.corpus_snapshot_hash,
        }
        if live is not None:
            record["live_bibcodes"] = live.bibcodes
            record["live_dropped_no_abstract"] = live.dropped_no_abstract
            record["live_dropped_budget"] = live.dropped_budget
            record["live_dropped_duplicate"] = live.dropped_duplicate
            record["live_tokens"] = live.tokens
        _append_jsonl(self.path, record)

        print(final.text, flush=True)
        if status == "flagged":
            err.print(
                f"[bold red]WARNING: citation guard FLAGGED this answer (not in context: "
                f"{escape(', '.join(guard.violations)) or '-'}; cites nothing: "
                f"{not guard.uncited_ok}). Treat it as unverified.[/bold red]",
                highlight=False,
            )
        err.print(
            f"[dim]guard={status} regenerated={second is not None} "
            f"abstain={parsed.pure_abstention} {summary} latency={record['latency_ms']} ms "
            f"→ {self.path}[/dim]",
            highlight=False,
        )
        return record


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


# ── CLI ──────────────────────────────────────────────────────────────────────


def _settings(
    *, k: int, pool: int, budget: int, tag: Tag | None, live: bool, dry_run: bool,
    max_tokens: int, temperature: float | None, corpus: Path,
) -> Settings:
    load_dotenv(REPO_ROOT / ".env")
    if pool != POOL:
        err.print(f"[yellow]--pool {pool} departs from the frozen pool={POOL} config.[/yellow]")
    if k > pool:
        err.print(f"[yellow]--k {k} exceeds --pool {pool}; the pool is widened to k.[/yellow]")
    return Settings(
        k=k,
        pool=pool,
        budget=budget,
        max_tokens=max_tokens,
        temperature=resolve_temperature(temperature),
        base_url=os.environ.get("LLM_BASE_URL") or DEFAULT_LLM_BASE_URL,
        model=os.environ.get("LLM_MODEL") or DEFAULT_LLM_MODEL,
        live=live,
        dry_run=dry_run,
        tag=tag.value if tag else None,
        corpus=corpus if corpus.is_absolute() else REPO_ROOT / corpus,
    )


def _run_or_report(session: Session, query: str, tag: str | None) -> bool:
    """Run a turn, reporting (not raising) service errors. True on success."""
    try:
        session.run_turn(query, tag)
    except httpx.HTTPError as exc:
        err.print(
            f"[red]LLM call failed ({type(exc).__name__}: {escape(str(exc))}). Is llama-server "
            f"running at {session.s.base_url}?[/red]"
        )
        return False
    except RuntimeError as exc:
        err.print(f"[red]{escape(str(exc))}[/red]")
        return False
    return True


_K_HELP = "Serving context k: top-k of the fused pool ranking (default: context.K, the D1 value)."
_TEMP_HELP = f"Sampling temperature (default: $LLM_TEMPERATURE or {DEFAULT_TEMPERATURE})."
_LIVE_HELP = "Live mode (flagged experiment): add live ADS results; separate transcript log."


@app.command()
def ask(
    query: str = typer.Argument(..., help="The question, used verbatim for retrieval."),
    k: int = typer.Option(K, "--k", help=_K_HELP),
    pool: int = typer.Option(POOL, help="Fused candidate pool depth (frozen at 100)."),
    budget: int = typer.Option(BUDGET, help="Context token budget (D1)."),
    tag: Tag | None = typer.Option(None, help="Error-taxonomy tag recorded in the transcript."),
    live: bool = typer.Option(False, "--live", help=_LIVE_HELP),
    dry_run: bool = typer.Option(False, "--dry-run", help="Retrieve + render; print payload."),
    max_tokens: int = typer.Option(DEFAULT_MAX_TOKENS, help="Answer token allowance."),
    temperature: float | None = typer.Option(None, help=_TEMP_HELP),
    simbad: str | None = typer.Option(None, help="Show a SIMBAD card (display only)."),
    corpus: Path = typer.Option(DEFAULT_CORPUS_PATH, help="Frozen corpus snapshot JSONL."),
) -> None:
    """Answer one question (answer → stdout, everything else → stderr)."""
    settings = _settings(
        k=k, pool=pool, budget=budget, tag=tag, live=live, dry_run=dry_run,
        max_tokens=max_tokens, temperature=temperature, corpus=corpus,
    )
    session = Session(settings)
    if simbad:
        session.show_simbad(simbad)
    if not _run_or_report(session, query, settings.tag):
        raise typer.Exit(1)


_CHAT_HELP = """\
commands: /simbad <name> (display-only object card) · /tag <tag> (tag the last turn) ·
/help · /quit. Anything else is a question; turns are independent (no memory)."""


@app.command()
def chat(
    k: int = typer.Option(K, "--k", help=_K_HELP),
    pool: int = typer.Option(POOL, help="Fused candidate pool depth (frozen at 100)."),
    budget: int = typer.Option(BUDGET, help="Context token budget (D1)."),
    tag: Tag | None = typer.Option(None, help="Default error-taxonomy tag for every turn."),
    live: bool = typer.Option(False, "--live", help=_LIVE_HELP),
    dry_run: bool = typer.Option(False, "--dry-run", help="Retrieve + render; print payload."),
    max_tokens: int = typer.Option(DEFAULT_MAX_TOKENS, help="Answer token allowance."),
    temperature: float | None = typer.Option(None, help=_TEMP_HELP),
    corpus: Path = typer.Option(DEFAULT_CORPUS_PATH, help="Frozen corpus snapshot JSONL."),
) -> None:
    """Interactive REPL (rich prompt on stderr; answers on stdout)."""
    settings = _settings(
        k=k, pool=pool, budget=budget, tag=tag, live=live, dry_run=dry_run,
        max_tokens=max_tokens, temperature=temperature, corpus=corpus,
    )
    session = Session(settings)
    err.print(
        f"[bold]AstroLLM copilot v0[/bold] — mode={session.mode} model={settings.model} "
        f"k={settings.k} transcript={session.path}",
        highlight=False,
    )
    err.print(_CHAT_HELP, markup=False)
    while True:
        try:
            line = Prompt.ask("[bold cyan]astro[/bold cyan]", console=err).strip()
        except (EOFError, KeyboardInterrupt):
            err.print()
            break
        if not line:
            continue
        if line in {"/quit", "/exit", "/q"}:
            break
        if line == "/help":
            err.print(_CHAT_HELP, markup=False)
            continue
        if line.startswith("/simbad"):
            name = line.removeprefix("/simbad").strip()
            if name:
                session.show_simbad(name)
            else:
                err.print("usage: /simbad <object name>")
            continue
        if line.startswith("/tag"):
            value = line.removeprefix("/tag").strip()
            if value not in {t.value for t in Tag}:
                err.print(f"usage: /tag <{'|'.join(t.value for t in Tag)}>")
            elif not retag_last_turn(session.path, value):
                err.print("no turn to tag yet")
            else:
                err.print(f"tagged turn {session.turn} as {value}")
            continue
        _run_or_report(session, line, settings.tag)


if __name__ == "__main__":
    app()
