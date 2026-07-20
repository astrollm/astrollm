# RAG-SFT Pilot — Recipe Validation on the Frozen Exoplanet-Atmosphere Corpus (corpus-fixed)

Pivot from the retrieval thread (closed: [pilot ablation](pilot-ablation.md), [corpus
widening](corpus-widening.md), [pool sweep](pool-sweep.md)) to SFT data curation — the active
critical path for the week-12 beta and the "fine-tuned weights + SFT dataset on HuggingFace"
deliverable. Nothing is fine-tuned yet and no SFT data is curated; this commission builds the
**recipe** and pre-registers the eval that gates it.

This is **Phase A** of a two-phase arc that mirrors the retrieval thread's pilot→widen structure
(pilot 500 → widened 2,500). Phase A holds the **corpus frozen** and treats the **curation recipe**
as the manipulated thing: build a gold seed, a teacher-synthesis step, a citation-grounding
verification pass, and a pre-registered fine-tune eval, all on the corpus that already exists. Phase
B (a later, separate commission) holds the **recipe frozen** and widens the corpus to a broad
astro-ph ADS slice — that is the run that produces the shippable, general v1 SFT set. Single-variable
discipline is preserved across the pivot: exactly one of {recipe, corpus} moves per phase.

## Scope

**Phase A builds and validates the RAG-SFT curation recipe.** It produces a *pilot* SFT set on the
narrow corpus, a verification pass with measured precision/recall, and a pre-registered eval
comparing a QLoRA fine-tune against the base model.

Phase A explicitly does **not**:

- ship anything to HuggingFace, or describe anything as fine-tuned, until the acceptance gates below
  are met (honest-status constraint);
- publish the pilot set as "the v1 astronomy SFT dataset" — it is single-topic
  (exoplanet-atmosphere) by construction and is the pilot, not the product artifact;
- wire tool-call examples or structured-SIMBAD object-property QA — tools are not live (per
  `CLAUDE.md`, ADS/SIMBAD MCP is deferred to "set up when needed"), and synthesizing examples of a
  capability that does not exist trains a fiction.

The transferable output of Phase A is the **recipe and the verifier**, not the pilot set's
magnitudes — exactly as the retrieval pilot's transferable result was the fusion-dilution mechanism,
not the 500-abstract recall numbers. Phase B re-runs the validated recipe on a corpus that matches
the product's general-astronomy framing and the AstroMLab-1 eval distribution.

## Setup — what is fixed and what varies

### The corpus (the controlled variable — held fixed)

The only ADS corpus that exists in the repo is the retrieval corpus. Phase A grounds all synthesis
on it, unchanged, so that the SFT context distribution equals the distribution the retriever serves
at inference.

| property | value |
|---|---|
| corpus | 2,500 ADS abstracts, query `abs:"exoplanet atmosphere"`, `year:[2018 TO *]`, citation-ranked (pilot 500 ⊂ widened 2,500) |
| pipeline | `pilot-retrieval-0.1.0` (one chunk = title + abstract) |
| retrieval for context assembly | hybrid BM25+dense, RRF, **pool=100** (the beta stage-1 default) |
| corpus snapshot | recorded by hash in the manifest; identical to the [pool-sweep](pool-sweep.md) index |

Grounding on a single-topic corpus is a stated limitation, not a defect: the topic was chosen to
test retrieval *mechanisms*, and Phase A reuses it to test *curation mechanisms* on a
distribution-matched, fully-understood corpus before paying to widen. The narrowness is the reason
Phase A ships nothing public.

### The recipe (the manipulated variable)

Exactly one thing is under study: the curation recipe (gold seed + teacher synthesis + verification
+ split). The corpus, the retriever, and the base model are held fixed; the recipe is what Phase B
will later re-run unchanged against a different corpus.

| held fixed | value |
|---|---|
| base model | Qwen3.5-4B (the v1 first-experiment model) |
| fine-tune method | QLoRA (config committed at the results step under `configs/`) |
| retriever | hybrid RRF @ pool=100, frozen |
| corpus | the 2,500-abstract slice above |

## Task shape (all examples)

Every example is **retrieval-augmented**: `(query + retrieved abstracts) → grounded answer with
bibcode citations`. No example is `question → answer-from-parametric-memory` — that shape trains the
hallucination behaviour the retrieval layer exists to suppress, i.e. training against the
architecture. Three task families:

| family | weight | what it trains |
|---|---|---|
| literature-grounded QA | ~45% | answer a question using only the retrieved abstracts, citing bibcodes for specific claims |
| citation-grounded summarization | ~35% | summarize what the retrieved literature says on a topic, with per-claim bibcode attribution |
| explicit abstention / insufficient-context flag | ~20% | when retrieved context is thin, contradictory, or off-topic, say so and decline rather than fabricate |

Abstention is its own family and is deliberately over-weighted relative to the V1 plan's implicit
treatment: it is the single highest-leverage honesty behaviour for a research copilot and the
cheapest to under-train. Pedagogy is **out of Phase A** unless trivially cheap, and if included is
grounded only on explicitly open sources (OpenStax Astronomy CC-BY, arXiv reviews) — never
copyrighted textbooks. Object-property QA is reframed, where it appears at all, as RAG-over-abstracts
(object named → SIMBAD alias expansion feeds retrieval → answer cites abstracts), so the whole set
has one task shape and one verification mechanism; structured-SIMBAD object QA is deferred with the
tool slice.

## Pipeline (`packages/data-pipeline/src/sft/`)

Four stages, each a separate build commit after this pre-registration is locked.

1. **Gold seed (hand-curated).** 150–250 examples written by hand across the three families,
   grounded on real abstracts from the corpus, with correct bibcodes and several deliberately hard
   abstention cases (thin retrieval, contradictory abstracts, plausibly-relevant-but-actually-absent
   targets). The gold seed is partitioned **disjointly** into a *verifier-calibration* set and an
   *eval-seed* set — no example calibrates the verifier and also scores the model. This is the
   human-labelled reference the rest of the pipeline is measured against; teacher synthesis does not
   substitute for it.
2. **Teacher synthesis.** A teacher LLM (Claude API) generates QA/summaries grounded on context
   assembled by the frozen `retrieve()` at pool=100, so the synthetic context distribution equals the
   inference context distribution. Synthesis provenance recorded on every example.
3. **Citation-grounding verification pass (built in-repo, not an API call).** For each synthesized
   example, check that each cited claim is actually supported by the cited abstract. The method
   (NLI/entailment or claim-span overlap + a held-back judge) is committed before it runs, validated
   against the gold-seed calibration partition, and its precision/recall reported (SFT-H5). Examples
   that fail verification are dropped, not silently kept.
4. **Manifest + split.** JSONL + `manifest.json`. 95/5 split **by task family**, with eval queries
   disjoint from training queries and, where the corpus allows, drawn on held-out abstracts to limit
   contamination.

### Per-example provenance schema (non-negotiable)

```
example_id
task_family                 # lit_qa | summarization | abstention
query
retrieved_context[]         # {bibcode, chunk_id, retrieval_score}
answer
cited_bibcodes[]
synthesis                   # {teacher_model, teacher_version, prompt_template_hash, timestamp}
verification                # {method, claim_support_result, passed, verifier_version, human_reviewed}
split                       # train | eval
license_note                # generated text + bibcode pointers; no reproduced abstract text published
```

The public HuggingFace artifact (Phase B) is the generated QA plus **bibcode provenance pointers**,
never reproduced ADS abstract text — abstract text is publisher-copyrighted.

## Predictions (pre-registered)

Registered **before** any data is generated. These are the commission's hypotheses; the comparison
is always QLoRA-SFT Qwen3.5-4B vs the same base model, on the held-out eval-seed set, with
paired-bootstrap CIs reported as in the retrieval thread. The pilot eval set is small, so — as with
the retrieval pilot's n=29 — point estimates are read as suggestive and CIs as the discipline;
direction and the existence-proof behaviours are what Phase A is powered to show, not tight
magnitudes.

- **SFT-H1 (grounding lift).** The fine-tune answers more faithfully from provided context than base.
  Falsifiable form: paired-bootstrap 95% CI on (SFT − base) faithfulness excludes 0, target point
  estimate ≥ +0.10 absolute. *If grounding does not beat base, the honest shippable status is "RAG
  with the base model" and SFT is iterated, not shipped (see gates).*
- **SFT-H2 (citation accuracy).** Cited-bibcode-supports-claim rate clears the V1 plan's >80% target
  **and** beats base. Reported on the eval-seed set with the verifier *and* spot-checked against
  human labels (the verifier is itself a measured instrument — SFT-H5 — not assumed correct).
- **SFT-H3 (abstention, two-sided).** On thin/empty/off-topic-retrieval cases the correct
  refusal-or-flag rate beats base, **subject to a pre-registered cap on over-abstention**: false
  refusal on answerable queries ≤ 0.10. The cap is load-bearing — naive abstention training "wins"
  by refusing everything, so the hypothesis is only met if both move the right way.
- **SFT-H4 (no knowledge regression — disambiguated).** A drop on the AstroMLab-1 subset conflates
  two causes that this model deliberately separates: genuine knowledge loss, and benign task-format
  interference (the model was trained to attend to provided context, cite, and abstain on thin
  context — handed a closed-book MCQ with no context, that training correctly fires as hedging,
  which scores as "wrong" without being knowledge loss). The −2 pp tripwire could not tell these
  apart and would mostly catch the benign one. AstroLLM is positioned as a grounded copilot, not a
  closed-book MCQ competitor to AstroSage; gating Phase A on closed-book MCQ retention would measure
  it against precisely the model it is positioned not to be. So H4 is measured both ways and
  gated on the meaningful one:

  - **Open-book, topic-overlap slice (the gate).** On the subset of AstroMLab-1 the frozen
    exoplanet-atmosphere corpus can genuinely ground (exoplanet/atmosphere questions), evaluate
    the MCQ with relevant retrieved context supplied (retrieve() @ pool=100), base and SFT in
    the same mode. SFT open-book accuracy ≥ base open-book − 2 pp. A clear regression here means
    the SFT degraded the model's ability to use astronomy knowledge even when handed the right
    material — genuine damage, kill signal. The slice may be small (cf. the retrieval pilot at
    n = 29): report a paired-bootstrap CI and read a directional regression beyond the CI as the
    operative kill, not a knife-edge 2 pp.
  - **Closed-book, full subset (reported drift, not gated).** Report the closed-book delta vs base
    as an expected, documented Phase-A characteristic — the corpus is single-topic and the model
    is abstention-trained, so some softening is by design, and Phase B's corpus widening is its
    designed remedy. Catastrophe backstop: a closed-book drop > 10 pp, or accuracy sliding
    toward the dataset's random-choice baseline, triggers a training-stability investigation
    before Phase B — but is not on its own a Phase-A kill if the open-book gate passes.
  - **Scope honesty.** Full open-book general-astronomy retention is not measurable in Phase A —
    the single-topic corpus cannot ground general questions. The comprehensive knowledge-retention
    gate is therefore a Phase-B responsibility, on the general corpus that actually ships,
    where open-book general retention becomes measurable. Phase A keeps only the recipe-soundness
    kill (open-book overlap regression) and the catastrophe backstop.
    Dual reporting is preserved: both modes go in the same table; only the open-book overlap number
    carries kill power in Phase A.
- **SFT-H5 (verifier validity).** The verification pass's precision/recall against the gold-seed
  calibration partition is reported. The verifier is trusted as a filter only if precision ≥ 0.85 on
  the gold labels; below that, its drop decisions are advisory and every "passed" example feeding
  training is human-reviewed instead.

### Pre-registered confounds and contamination controls

- **Teacher/judge shared-bias.** The teacher (Claude) writes the training data; the eval must not be
  graded solely by the same model family, or shared bias inflates every score. SFT-H1–H3 are scored
  against human gold labels and/or a judge from a different family; this is recorded per metric.
- **Eval-train leakage.** Eval queries are disjoint from training queries; verifier-calibration and
  eval-seed gold partitions are disjoint; held-out abstracts used for eval context where the corpus
  permits. Any unavoidable overlap is reported, not hidden.
- **Single-topic ceiling.** Every Phase-A number is on exoplanet-atmosphere abstracts. The recipe is
  the transferable claim; the magnitudes are not asserted to hold on a general corpus. That is what
  Phase B tests.

## Acceptance gates / kill criteria

Fine-tune into the beta **only if**: SFT-H1 holds (grounding beats base), SFT-H2 holds (citation
accuracy clears 80% and beats base), and SFT-H4's **open-book topic-overlap** gate holds (no clear
regression beyond CI vs base, evaluated with relevant retrieved context). The closed-book number is
reported but does not gate Phase A unless it trips the catastrophe backstop (> 10 pp drop or
approaching random-choice), which triggers a training-stability check rather than an automatic kill.
Comprehensive general-astronomy retention is gated at Phase B, on the corpus that ships.

If the gates are not met, the honest outcome is **"RAG-grounded beta on the base model, SFT
iterating"** — not a fine-tune shipped to hit week 12. The differentiation is grounded, cited,
honest answers; shipping weak grounding to make a milestone forfeits exactly that. The week-12 date
does not override the gates.

## Amendments

- **2026-06-02** (pre-data — before any SFT data generated or eval run): SFT-H4 revised. Original
  closed-book −2 pp AstroMLab-1 gate replaced with an open-book topic-overlap gate (closed-book
  reported as drift + catastrophe backstop; general retention deferred to Phase B). Rationale:
  closed-book MCQ conflates knowledge loss with by-design abstention hedging. See commit a60e1c9.

- **2026-07-21** (pre-data — still true: 0 of the 150–250 gold examples authored): the 2026-06-02
  amendment operationalized after an adversarial review found it under-specified in four ways that
  each, unintentionally, weakened the gate. Recorded as EXP-004-A1 in the
  [Research Log](../RESEARCH_LOG.md); this bullet is the normative text.

    1. **Frozen slice-selection rule.** "The subset of AstroMLab-1 the frozen corpus can genuinely
       ground" was undefined — a post-hoc degree of freedom. The open-book topic-overlap slice is
       now selected by a **committed keyword/topic filter over AstroMLab-1 question text, fixed
       and recorded before any model output is seen**, and the slice size n is reported before the
       eval runs. The filter is committed alongside the eval harness, not chosen after results.
    2. **Minimum-power fallback.** "Regression beyond the CI is the operative kill" inverts the
       gate's polarity on a small slice: a wide CI passes by default under noise. If the slice has
       n < 40, or the paired-bootstrap CI half-width on the base arm exceeds 5 pp, the open-book
       gate **additionally** requires the SFT point estimate ≥ base − 5 pp. The CI is still
       reported; low power no longer converts to an automatic pass.
    3. **Backstop defined, and made a ship-block.** "Approaching random-choice" is replaced with
       thresholds: a closed-book drop > 10 pp vs base, **or** closed-book accuracy < 35% (4-option
       MCQ, 25% chance), is a **ship-block pending investigation** — not merely a
       training-stability check that shipping can outrun.
    4. **The Phase-A fine-tune never ships.** The gates section read as if a passing Phase-A model
       could "fine-tune into the beta," while the Scope section says the pilot set is not the
       product artifact. Resolved in Scope's favor: **no Phase-A fine-tune ships into the beta
       under any Phase-A result.** Phase A validates the recipe; only a Phase-B run that passes
       the Phase-B pre-registered general-retention gate (to be written into the Phase-B
       pre-registration before that run) ships. The acceptance gates decide whether the recipe is
       validated for Phase B, not whether weights ship.

  Same-review operationalizations of the other hypotheses (measurement hygiene; no gate changes):

    - **SFT-H1**: the faithfulness rubric, scale, and judge identity are committed per metric
      before the eval runs; the "human gold labels and/or a different-family judge" disjunction is
      resolved to a named grader per metric at that time.
    - **SFT-H2**: citation density / claim coverage is reported alongside the support rate (a
      citation-parsimonious model can game the rate by citing only trivial claims), and the human
      spot-check carries a pre-stated agreement floor with the verifier.
    - **SFT-H3**: the refusal/flag judgment rubric is committed before the eval runs.
    - **SFT-H5**: verifier **recall** and the distribution of dropped examples are reported next to
      precision (a high-precision/low-recall verifier silently skews training data toward
      trivially-verifiable claims). Reporting requirements, not new gates.

---

*Pre-registration commit. Results, Interpretation, Verdict on the hypotheses, and Reproduce are
appended at the results step (separate commit, no squash), after the QLoRA run and the eval.*