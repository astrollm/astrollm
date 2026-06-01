"""Schema contract + validator for gold-seed SFT examples.

This module is the single source of truth for what a valid gold example is. It is intentionally
pure (no DB, no torch, no network) so it imports and lints offline and is cheap to unit-test.

Two layers of validation:

1. **Structural + cross-field rules** (`GoldExample` model validator) — enforced at construction
   time, so it is impossible to build a `GoldExample` that violates the contract's rules 1-5. These
   need only the example itself.
2. **Corpus-grounding rules** (`validate_against_corpus`) — rule 6 (every retrieved bibcode exists
   in the current corpus snapshot) plus advisory grounding warnings. These need the corpus snapshot,
   so they live in a separate function the harness calls with the loaded snapshot.

`validate_record` ties both together for the authoring CLIs: dict in, `(example, errors, warnings)`
out — errors reject the example, warnings are surfaced to the labeler but do not block.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, ValidationError, model_validator


# Enums kept as (str, Enum) verbatim from the commission's contract; UP042 (StrEnum) is suppressed
# rather than silently changing the specified base classes.
class TaskFamily(str, Enum):  # noqa: UP042
    LIT_QA = "lit_qa"
    SUMMARIZATION = "summarization"
    ABSTENTION = "abstention"


class AbstentionReason(str, Enum):  # noqa: UP042
    THIN = "thin"
    CONTRADICTORY = "contradictory"
    OFF_TOPIC = "off_topic"
    ABSENT = "absent"


class NegativeType(str, Enum):  # noqa: UP042
    CLAIM_NOT_SUPPORTED = "claim_not_supported"
    CONTRADICTS = "contradicts"
    WRONG_PAPER = "wrong_paper"
    OVERGENERALIZATION = "overgeneralization"


class Partition(str, Enum):  # noqa: UP042
    CALIBRATION = "calibration"
    EVAL = "eval"  # gold seed NEVER trains


class Retrieved(BaseModel):
    bibcode: str
    chunk_id: str
    retrieval_score: float


class Claim(BaseModel):
    text: str
    cited_bibcode: str | None  # None only for abstention answers
    support_span: str | None  # the abstract sentence/offset the claim rests on
    supported: bool  # human label; False == a calibration negative


class GoldExample(BaseModel):
    example_id: str
    task_family: TaskFamily
    partition: Partition
    query: str
    retrieved_context: list[Retrieved]  # actual frozen pool=100 output
    answer: str
    claims: list[Claim]
    abstention_reason: AbstentionReason | None = None
    is_negative: bool = False
    negative_type: NegativeType | None = None
    provenance: dict = Field(default_factory=dict)  # {author, date, retriever_run_id, corpus_*}

    @model_validator(mode="after")
    def _check_contract(self) -> GoldExample:
        """Cross-field rules 1-5 (rule 6 needs the corpus — see validate_against_corpus)."""
        is_abstention = self.task_family is TaskFamily.ABSTENTION

        # Rule 1: ABSTENTION ⇒ abstention_reason set (claims may have cited_bibcode=None).
        if is_abstention and self.abstention_reason is None:
            raise ValueError("abstention task_family requires abstention_reason")

        # Rule 2: non-ABSTENTION ⇒ abstention_reason is None AND ≥1 claim has a cited_bibcode.
        if not is_abstention:
            if self.abstention_reason is not None:
                raise ValueError("abstention_reason is only valid for the abstention task_family")
            if not any(c.cited_bibcode for c in self.claims):
                raise ValueError(
                    f"{self.task_family.value} requires at least one claim with a cited_bibcode"
                )

        # Rule 3: negative ⇒ CALIBRATION AND negative_type set AND ≥1 unsupported claim.
        if self.is_negative:
            if self.partition is not Partition.CALIBRATION:
                raise ValueError("negatives must be in the calibration partition")
            if self.negative_type is None:
                raise ValueError("is_negative requires a negative_type")
            if not any(not c.supported for c in self.claims):
                raise ValueError("a negative must have at least one claim with supported=False")

        # Rule 4: non-negative ⇒ negative_type is None.
        if not self.is_negative and self.negative_type is not None:
            raise ValueError("negative_type is only valid when is_negative=True")

        # Rule 5: EVAL ⇒ not negative (no negatives ever reach eval or training).
        if self.partition is Partition.EVAL and self.is_negative:
            raise ValueError("eval partition must not contain negatives")

        return self


def validate_against_corpus(
    example: GoldExample, corpus_bibcodes: set[str]
) -> tuple[list[str], list[str]]:
    """Rule 6 + advisory grounding checks against the frozen corpus snapshot.

    Returns ``(errors, warnings)``. Errors reject the example; warnings are surfaced to the labeler
    but never block — they flag likely mistakes (e.g. citing a paper outside the retrieved pool)
    without inventing rejection rules beyond the contract.
    """
    errors: list[str] = []
    warnings: list[str] = []

    # Rule 6 (hard): every retrieved bibcode must exist in the current corpus snapshot.
    retrieved = {r.bibcode for r in example.retrieved_context}
    missing = sorted(
        r.bibcode for r in example.retrieved_context if r.bibcode not in corpus_bibcodes
    )
    if missing:
        errors.append(
            f"{len(missing)} retrieved bibcode(s) absent from corpus snapshot: "
            f"{', '.join(missing[:5])}{' …' if len(missing) > 5 else ''}"
        )

    # Advisory: a cited bibcode should be a real corpus paper, and grounding means it should be
    # one the retriever actually surfaced for this query (a WRONG_PAPER negative deliberately cites
    # the wrong-but-real paper, so out-of-pool citations are warned, never rejected).
    for i, claim in enumerate(example.claims):
        if claim.cited_bibcode is None:
            continue
        if claim.cited_bibcode not in corpus_bibcodes:
            warnings.append(f"claim[{i}] cites {claim.cited_bibcode}, not in corpus snapshot")
        elif claim.cited_bibcode not in retrieved:
            warnings.append(
                f"claim[{i}] cites {claim.cited_bibcode}, not in this query's retrieved_context"
            )
        if claim.supported and not claim.support_span:
            warnings.append(f"claim[{i}] is supported but has no support_span")

    return errors, warnings


def validate_record(
    data: dict, corpus_bibcodes: set[str] | None = None
) -> tuple[GoldExample | None, list[str], list[str]]:
    """Validate a raw example dict.

    Returns ``(example, errors, warnings)``. On structural/contract failure ``example`` is None and
    ``errors`` carries the pydantic messages. When ``corpus_bibcodes`` is provided, rule 6 and the
    advisory grounding checks are run too; when omitted (pure offline validation) they are skipped.
    """
    try:
        example = GoldExample.model_validate(data)
    except ValidationError as exc:
        return None, [_fmt_error(e) for e in exc.errors()], []

    if corpus_bibcodes is None:
        return example, [], []
    errors, warnings = validate_against_corpus(example, corpus_bibcodes)
    return example, errors, warnings


def _fmt_error(err: dict) -> str:
    loc = ".".join(str(p) for p in err.get("loc", ())) or "(root)"
    return f"{loc}: {err.get('msg', 'invalid')}"
