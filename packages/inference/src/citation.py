"""Citation guard + D4 answer-format parser for the M1 copilot — pure, offline, no I/O.

Implements the pre-registered answer-format contract **D4** (``docs/V1_PLAN_R2.md``; normative
reward text in ``docs/research/training-pilot.md``) once, so serving, the gold seed, teacher
synthesis, H3/T-H3 and the EXP-005 reward can all classify answers with the same parser:

* A **pure abstention** begins with ``ABSTAIN: <thin|contradictory|off_topic|absent>`` followed by
  at most one sentence of at most 40 words. Any further sentence (or > 40 words) makes it a
  **mixed** response, scored as a *non-abstaining* answer.
* A **non-abstaining** answer carries inline bibcode citations and may end with one trailing
  ``Sources:`` line containing bibcodes and nothing else. Prose on that line makes it malformed
  (and the line then counts as ordinary answer text).
* **Citation validity** applies to every answer, pure abstentions included: every bibcode anywhere
  in the answer — ``Sources:`` lines included, well-formed or not — must be in the serving
  context. A non-abstaining answer with zero *inline* citations is also a violation (grounded
  answers must cite; a ``Sources:`` line never substitutes for inline citations).

Fixed rules (deterministic, not under the model's control):

* The answer is ``strip()``-ed before parsing; the marker must then be the very first text and is
  case-sensitive (``ABSTAIN:``, lowercase reason code). ``ABSTAIN:`` anywhere later sets
  ``marker_misplaced`` (recorded only; the classification is unchanged).
* Only the **last** non-empty line may be the Sources line, spelled exactly ``Sources:``. Any
  earlier (or second) line starting ``Sources:``/``Source:``, or a trailing ``Source:`` line, sets
  ``sources_line_malformed``; its bibcodes still go through the allowlist check but never count
  as inline citations.
* **Sentence heuristic** (the shared H3/T-H3 classifier; applied to the abstention remainder):
  1. whitespace is normalised first (runs of spaces/newlines → one space), so a line break inside
     a sentence is not a boundary;
  2. bibcodes are masked, then the periods of these abbreviations are masked: ``Dr. Mr. Ms.
     Prof. Fig. Eq. Sect. e.g. i.e. et al. vs. ca. approx. No.`` and single capital initials
     (``J. Smith``) — case-sensitive;
  3. a sentence ends at ``.``/``!``/``?``, optionally followed by closing quotes/brackets
     (``" ” ’ ' ) ]``), then whitespace; the final sentence needs no terminator.
* Words are whitespace-separated tokens; the 40-word cap counts every word after the marker.

The D1 whole-abstract context budget is *not* here: it lives in
``packages/evaluation/src/context_cut.py`` (``deliver``), which the copilot reuses so serving
applies exactly the overflow policy the D1 measurement used.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from typing import Literal

# 19-char ADS bibcode: YYYY + 5-char journal + 4 volume + 1 qualifier + 4 page + 1 author initial.
# Verified 2026-11-02: matches 38/38 gold expected_bibcodes and 2,500/2,500 corpus bibcodes.
# The lookbehind rejects only an alphanumeric neighbour, so a bibcode right after ``.``/``&`` (e.g.
# ``...2019A&A...621A..68P``) is still seen by the guard.
BIBCODE_RE = re.compile(
    r"(?<![A-Za-z0-9])\d{4}[A-Za-z][A-Za-z0-9&.]{4}[A-Za-z0-9.]{9}[A-Z.](?![A-Za-z0-9])"
)

ABSTAIN_REASONS: tuple[str, ...] = ("thin", "contradictory", "off_topic", "absent")
ABSTAIN_RE = re.compile(r"^ABSTAIN:[ \t]*(thin|contradictory|off_topic|absent)(?![A-Za-z0-9_])")
_ABSTAIN_PREFIX_RE = re.compile(r"^ABSTAIN\b")
_MARKER_ANYWHERE_RE = re.compile(r"ABSTAIN\s*:")
SOURCES_RE = re.compile(r"^Sources:(.*)$")  # the only accepted spelling for the trailing line
_SOURCES_LIKE_RE = re.compile(r"^Sources?:")  # anything that looks like a Sources line
MAX_ABSTAIN_WORDS = 40
MAX_ABSTAIN_SENTENCES = 1

# Separators allowed between bibcodes on a ``Sources:`` line — anything else is prose.
_SOURCES_RESIDUE_RE = re.compile(r"^[\s,;\[\]()]*$")
# Separators between the reason code and the one-sentence explanation.
_LEADING_SEP_RE = re.compile(r"^[\s.:;,\-–—]+")
_ABBREVIATIONS = re.compile(
    r"\b(?:Dr|Mr|Ms|Prof|Fig|Eq|Sect|e\.g|i\.e|et al|vs|ca|approx|No)\.|\b[A-Z]\."
)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])[\"”’')\]]*\s+")
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

GuardStatus = Literal["pass", "violation"]


def _dedupe(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def extract_bibcodes(text: str) -> list[str]:
    """Every bibcode in ``text``, in order of first appearance, de-duplicated."""
    return _dedupe(m.group(0) for m in BIBCODE_RE.finditer(text))


def count_sentences(text: str) -> int:
    """Sentences in ``text`` under the fixed rule in the module docstring (0 for blank text)."""
    masked = BIBCODE_RE.sub("BIBCODE", " ".join(text.split()))
    masked = _ABBREVIATIONS.sub(lambda m: m.group(0)[:-1], masked)
    return sum(1 for s in _SENTENCE_SPLIT_RE.split(masked) if s.strip())


def count_words(text: str) -> int:
    return len(text.split())


@dataclass(frozen=True)
class ParsedAnswer:
    """One answer classified under D4."""

    text: str  # the stripped answer that was parsed
    is_abstain: bool  # starts with a valid ``ABSTAIN: <reason>`` marker
    reason: str | None  # the reason code when is_abstain
    mixed: bool  # marker + more than one sentence or > 40 words (scored as non-abstaining)
    body: str  # the answer minus a well-formed trailing Sources line
    sources_line: str | None  # the trailing ``Sources:`` line, verbatim, if any
    sources_bibcodes: list[str] = field(default_factory=list)
    sources_line_malformed: bool = False  # prose on it, ``Source:``, or a non-trailing Sources line
    inline_bibcodes: list[str] = field(default_factory=list)  # body bibcodes, no Sources lines
    all_bibcodes: list[str] = field(default_factory=list)  # inline then sources, de-duplicated
    abstain_marker_malformed: bool = False  # starts ``ABSTAIN`` but no valid ``: <reason>``
    marker_misplaced: bool = False  # ``ABSTAIN:`` somewhere other than the very start
    abstain_sentences: int = 0  # sentences after the marker (excluding a Sources line)
    abstain_words: int = 0  # words after the marker (excluding a Sources line)

    @property
    def pure_abstention(self) -> bool:
        """Scored as an abstention: valid marker and no substantive text beyond one sentence."""
        return self.is_abstain and not self.mixed

    def to_dict(self) -> dict[str, object]:
        """Transcript view (drops the raw text, which the transcript stores separately)."""
        d = asdict(self)
        d.pop("text")
        d.pop("body")
        d["pure_abstention"] = self.pure_abstention
        return d


def _split_sources(text: str) -> tuple[str, str | None, list[str], list[str], bool]:
    """(body, sources_line, inline_bibcodes, sources_bibcodes, malformed).

    Only the last non-empty line may be the Sources line (exact ``Sources:`` spelling, bibcodes
    only); a well-formed one is removed from the body. Every Sources-looking line contributes its
    bibcodes to ``sources_bibcodes`` (so they are allowlist-checked) and never to the inline set.
    """
    lines = text.split("\n")
    last = len(lines) - 1
    while last >= 0 and not lines[last].strip():
        last -= 1
    sources_line: str | None = None
    malformed = False
    keep_last = True
    inline: list[str] = []
    sources: list[str] = []
    for i, raw in enumerate(lines):
        line = raw.strip()
        if not _SOURCES_LIKE_RE.match(line):
            inline.extend(extract_bibcodes(raw))
            continue
        bibcodes = extract_bibcodes(line)
        sources.extend(bibcodes)
        if i != last:  # an earlier (or second) Sources line
            malformed = True
            continue
        sources_line = line
        m = SOURCES_RE.match(line)
        if m is None:  # trailing ``Source:`` (wrong spelling)
            malformed = True
            continue
        if not bibcodes or not _SOURCES_RESIDUE_RE.match(BIBCODE_RE.sub("", m.group(1))):
            malformed = True  # prose on the line → it stays in the body as ordinary (claim) text
            continue
        keep_last = False
    body = text if keep_last else "\n".join(lines[:last]).rstrip()
    return body, sources_line, _dedupe(inline), _dedupe(sources), malformed


def parse_answer(text: str) -> ParsedAnswer:
    """Classify ``text`` under the D4 grammar (see the module docstring for the fixed rules)."""
    stripped = text.strip()
    body, sources_line, inline, sources_bibcodes, malformed = _split_sources(stripped)
    all_bibcodes = _dedupe([*inline, *sources_bibcodes])

    m = ABSTAIN_RE.match(body)
    is_abstain = m is not None
    reason = m.group(1) if m else None
    n_sentences = n_words = 0
    mixed = False
    if m is not None:
        remainder = _LEADING_SEP_RE.sub("", body[m.end():])
        n_sentences = count_sentences(remainder)
        n_words = count_words(remainder)
        mixed = n_sentences > MAX_ABSTAIN_SENTENCES or n_words > MAX_ABSTAIN_WORDS
    return ParsedAnswer(
        text=stripped,
        is_abstain=is_abstain,
        reason=reason,
        mixed=mixed,
        body=body,
        sources_line=sources_line,
        sources_bibcodes=sources_bibcodes,
        sources_line_malformed=malformed,
        inline_bibcodes=inline,
        all_bibcodes=all_bibcodes,
        abstain_marker_malformed=not is_abstain and bool(_ABSTAIN_PREFIX_RE.match(body)),
        marker_misplaced=any(x.start() > 0 for x in _MARKER_ANYWHERE_RE.finditer(stripped)),
        abstain_sentences=n_sentences,
        abstain_words=n_words,
    )


@dataclass(frozen=True)
class GuardResult:
    """Outcome of the hard citation check for one answer."""

    status: GuardStatus
    violations: list[str]  # cited bibcodes not in the allowlist, in order of appearance
    uncited_ok: bool  # False iff a non-abstaining answer has no inline citation (a violation)
    abstain_with_citation: bool  # a pure abstention that cites anything (eval-time flag only)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def check_citations(parsed: ParsedAnswer, allowlist: set[str]) -> GuardResult:
    """Hard citation guard: every cited bibcode ∈ ``allowlist``; non-abstaining answers must cite.

    Citation validity is applied to every answer, pure abstentions included (training-pilot.md,
    reward component 1). A pure abstention that cites only allowed bibcodes is **not** a
    violation — R2 makes abstention-citation an eval-time classification, so it is recorded as
    ``abstain_with_citation=True`` rather than triggering a regeneration. Mixed responses are
    non-abstaining answers and must cite. Membership is checked on every bibcode (Sources lines
    included); the must-cite rule counts inline bibcodes only.
    """
    violations = [b for b in parsed.all_bibcodes if b not in allowlist]
    uncited_ok = parsed.pure_abstention or bool(parsed.inline_bibcodes)
    status: GuardStatus = "pass" if not violations and uncited_ok else "violation"
    return GuardResult(
        status=status,
        violations=violations,
        uncited_ok=uncited_ok,
        abstain_with_citation=parsed.pure_abstention and bool(parsed.all_bibcodes),
    )


def strip_think(text: str) -> tuple[str, bool]:
    """Remove ``<think>…</think>`` reasoning defensively; return (clean, had_think).

    Handles a closed block anywhere, an orphan ``</think>`` (template opened the block in the
    prompt: everything before it is reasoning), and an unclosed ``<think>`` (truncated reasoning:
    everything from it on is dropped).
    """
    had = "<think>" in text or "</think>" in text
    if not had:
        return text, False
    clean = _THINK_BLOCK_RE.sub("", text)
    if "</think>" in clean:
        clean = clean.rsplit("</think>", 1)[1]
    if "<think>" in clean:
        clean = clean.split("<think>", 1)[0]
    return clean.strip(), True
