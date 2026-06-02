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

from enum import StrEnum

from pydantic import BaseModel, Field, ValidationError, model_validator


# The contract specifies (str, Enum); StrEnum is its behavior-equivalent, ruff-idiomatic form
# (requires-python >= 3.11) — members are still str and every .value below is unchanged.
class TaskFamily(StrEnum):
    LIT_QA = "lit_qa"
    SUMMARIZATION = "summarization"
    ABSTENTION = "abstention"


class AbstentionReason(StrEnum):
    THIN = "thin"
    CONTRADICTORY = "contradictory"
    OFF_TOPIC = "off_topic"
    ABSENT = "absent"


class NegativeType(StrEnum):
    CLAIM_NOT_SUPPORTED = "claim_not_supported"
    CONTRADICTS = "contradicts"
    WRONG_PAPER = "wrong_paper"
    OVERGENERALIZATION = "overgeneralization"


class Partition(StrEnum):
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
        """Rules 1-5 + non-empty human fields. (Rule 6 + cited-bibcode existence need the
        corpus — see validate_against_corpus.)"""
        # Human-authored fields must be filled — catches an unedited `prepare` template
        # (answer:"" and claim text:"") slipping into the calibration/eval seed.
        if not self.answer.strip():
            raise ValueError("answer must not be empty")
        for i, claim in enumerate(self.claims):
            if not claim.text.strip():
                raise ValueError(f"claim[{i}].text must not be empty")

        is_abstention = self.task_family is TaskFamily.ABSTENTION

        # Rule 1: ABSTENTION ⇒ abstention_reason set (claims may have cited_bibcode=None).
        if is_abstention and self.abstention_reason is None:
            raise ValueError("abstention task_family requires abstention_reason")

        # Rule 2: non-ABSTENTION ⇒ abstention_reason is None AND EVERY claim cites a bibcode.
        # cited_bibcode=None is contract-restricted to abstention; each claim must be verifiable
        # on its own, so one citation can't cover an uncited sibling claim.
        if not is_abstention:
            if self.abstention_reason is not None:
                raise ValueError("abstention_reason is only valid for the abstention task_family")
            if not self.claims or not all(c.cited_bibcode for c in self.claims):
                raise ValueError(
                    f"every {self.task_family.value} claim must cite a bibcode "
                    "(≥1 claim; cited_bibcode=None is only for abstention)"
                )

        # Rule 3: negative ⇒ CALIBRATION AND negative_type set AND ≥1 unsupported claim.
        if self.is_negative:
            if self.partition is not Partition.CALIBRATION:
                raise ValueError("negatives must be in the calibration partition")
            if self.negative_type is None:
                raise ValueError("is_negative requires a negative_type")
            if not any(not c.supported for c in self.claims):
                raise ValueError("a negative must have at least one claim with supported=False")

        # Rule 4: non-negative ⇒ negative_type is None AND every claim is supported.
        # supported=False marks a calibration negative, so an unsupported claim ⇒ is_negative
        # (this also catches an unsupported claim sneaking into an eval example, which can't be
        # negative). Together with Rule 3 this makes is_negative ⟺ "has an unsupported claim".
        if not self.is_negative:
            if self.negative_type is not None:
                raise ValueError("negative_type is only valid when is_negative=True")
            if any(not c.supported for c in self.claims):
                raise ValueError(
                    "a non-negative example must have all claims supported "
                    "(supported=False marks a calibration negative)"
                )

        # Rule 5: EVAL ⇒ not negative (no negatives ever reach eval or training).
        if self.partition is Partition.EVAL and self.is_negative:
            raise ValueError("eval partition must not contain negatives")

        return self


def validate_against_corpus(
    example: GoldExample, corpus_bibcodes: set[str]
) -> tuple[list[str], list[str]]:
    """Rule 6 + cited-bibcode existence + advisory grounding checks against the frozen corpus.

    Returns ``(errors, warnings)``. Errors reject the example — a retrieved *or cited* bibcode that
    does not exist in the snapshot. Warnings never block: they flag grounding smells that are
    legitimate for some examples (e.g. a negative citing a real paper outside this query's pool).
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

    # Citation rules for a non-None cited bibcode:
    #  - absent from the corpus snapshot → error (a typo'd/invented pointer, not provenance);
    #  - in the corpus but outside *this query's* retrieved pool → error for positives, eval, and
    #    non-WRONG_PAPER negatives (it breaks the frozen-context guarantee — the answer would rest
    #    on a paper the model never saw); warning only for a deliberate WRONG_PAPER negative, which
    #    cites a real-but-un-retrieved wrong paper on purpose.
    for i, claim in enumerate(example.claims):
        if claim.cited_bibcode is None:
            continue
        if claim.cited_bibcode not in corpus_bibcodes:
            errors.append(f"claim[{i}] cites {claim.cited_bibcode}, absent from corpus snapshot")
        elif claim.cited_bibcode not in retrieved:
            is_wrong_paper = (
                example.is_negative and example.negative_type is NegativeType.WRONG_PAPER
            )
            msg = f"claim[{i}] cites {claim.cited_bibcode}, not in this query's retrieved_context"
            (warnings if is_wrong_paper else errors).append(msg)
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
