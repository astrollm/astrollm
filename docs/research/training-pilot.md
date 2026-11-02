# EXP-005 — Training Pilot: base vs SFT vs GRPO on the frozen corpus (pre-registration)

*Pre-registered 2026-11-02, before any training data is synthesized, any verifier is built, or any
GPU hour is spent. Executes milestone M4 of [V1 Plan R2](../V1_PLAN_R2.md). Extends
[EXP-004](sft-pilot.md) — the corpus, recipe, eval seed, hypotheses H1–H5 and the EXP-004-A1
amendments all carry over unchanged; this document adds a third arm and its guards.*

## Why a third arm

EXP-004 pre-registered one comparison: QLoRA-SFT Qwen3.5-4B on teacher-synthesized, verifier-
filtered data vs the prompted base. Since then (verified 2026-10-23):

- RL on answer correctness alone does not reduce post-rationalized citations (arXiv 2609.23053).
- Grounding, citation sufficiency and refusal quality can be trained with GRPO from programmatic,
  verifiable rewards with no gold answers (arXiv 2506.15522, "Ground-GRPO"); dense rewards beat
  exact-match for small search agents (arXiv 2609.28765).

The project already owns the two things such a recipe needs: a deterministic frozen retriever that
assembles `(query, context)` prompts at scale with no labeling, and a claim-support verifier that
is calibrated against human labels (H5). The question EXP-005 asks is whether the **verifier as a
reward** beats the **verifier as a filter**.

## What is fixed (inherited)

| held fixed | value |
|---|---|
| corpus | the 2,500-abstract exoplanet-atmosphere snapshot (`sha256:08736961…`) |
| retriever | hybrid RRF, pool 100, **serving context top-k per D1** (same k for prompts, SFT data and RL rollouts) |
| base model | `Qwen/Qwen3.5-4B`, QLoRA (r=64, α=128; config committed under `configs/` at the run) |
| think mode | **no-think for all arms** (2026-07-21 decision + mode-symmetry rider); see "Open variant" |
| eval set | the EXP-004 eval partition (gold, never trained on); judge/rubrics per EXP-004-A1 and D3 |
| gates | SFT-H1, H2, H3 (with the ≤ 0.10 false-refusal cap), H4 open-book overlap gate + backstop, H5 verifier trust |

## Arms

| arm | training signal | data needed |
|-----|-----------------|-------------|
| **A. base** | none (prompted) | — |
| **B. SFT** | teacher-synthesized `(query + context) → cited answer`, verifier-filtered (EXP-004 recipe) | teacher API calls; verifier passes |
| **C. GRPO** | on-policy rollouts from `(query + context)` prompts scored by a programmatic reward | machine-assembled `(query, context)` prompts **plus human-audited answerability labels** (next section); verifier as reward |

Arm C runs **only if** the verifier passed the H5 trust gate (precision ≥ 0.85 on the calibration
partition, including on the human-authored challenge subset of EXP-004-A2) **and** the
answerability-label audit below passes. Otherwise the reward is too noisy to be a training signal
and arm C is recorded as "not run — verifier below gate" / "not run — label audit failed", not
skipped silently.

## Prompt set and answerability labels (arm C)

RL needs ~500–1,000 `(query, context)` prompts. Retrieval assembles the context deterministically,
but the **abstention term of the reward needs an answerability label per prompt**, and no retriever
supplies one. The labels are a labelled dataset in their own right, built and audited before the run:

- **Construction.** Queries are drawn from the same candidate pool and family mix as the gold seed
  (~45/35/20) and are disjoint from every eval-partition query. For each prompt the teacher proposes
  `answerable | thin | contradictory | off_topic | absent` with an evidence pointer (the bibcodes
  that make it answerable, or why none do). Absent-object prompts are additionally verified by the
  corpus grep used for the gold candidates.
- **Audit.** A human verifies **100% of prompts labelled non-answerable** (≈ 20%, 100–200 prompts)
  and a **20% random sample of the answerable ones**, blind to the teacher's reasoning. If the label
  error rate in the audited sample exceeds **10%**, arm C is not run. Budget ≈ 5–8 hours, added to M4.
- **Storage.** Labels are reward-only metadata on the prompt record (`answerability`, `evidence`,
  `audited`); they are never used as eval labels and never shipped.

## Reward specification (arm C) — committed before the run, in this order

The reward is a weighted sum of components that are each independently checkable; weights are
committed in the run config and reported.

0. **Answer classification and claim extraction are not under the policy's control.** The answer
   format carries a fixed abstention marker (a leading `ABSTAIN:` line). A **pure abstention** is
   an answer with the marker and no declarative sentence other than the stated reason. A **mixed
   response** (marker plus substantive sentences, or no marker) is scored as a non-abstaining
   answer. Every declarative sentence of a non-abstaining answer is a claim, cited or not (sentence
   splitting by a fixed rule; quotations and a trailing "sources" line excluded). A non-abstaining
   answer with zero claims, or with zero citations, receives the reward floor.
1. **Citation validity** (hard, applied to **every** answer before any other component, pure
   abstentions included): every bibcode anywhere in the answer ∈ serving context. Any invalid
   citation → reward floor for the sample.
2. **Claim support and coverage** (dense): the fraction of extracted claims that are *both* cited
   *and* supported by the cited abstract per the calibrated verifier. An uncited claim counts as
   unsupported, so attaching valid citations to easy sentences while leaving substantive assertions
   uncited lowers the reward instead of evading it.
3. **Abstention correctness** (two-sided, from the audited answerability labels; pure abstentions
   are scored by components 1 and 3 only): +1 for a pure abstention on a thin/absent/contradictory/
   off-topic prompt, −1 for a pure abstention on an answerable prompt; a mixed response receives 0
   here and is scored on components 0–2 and 4. The prompt-set family mix mirrors the eval seed so
   over-abstention cannot pay.
4. **Format**: cited-answer shape, length band, no reproduced abstract text beyond quotation
   length.

Dense over binary (2609.28765); no answer-correctness term from a judge (2609.23053 — it does not
buy faithfulness and it would reintroduce the teacher/judge shared bias).

## Hypotheses (pre-registered)

Comparison is paired on the eval partition with bootstrap CIs, all arms no-think, judge ≠ teacher.

- **T-H1 (SFT lift).** Arm B meets SFT-H1 and SFT-H2 vs arm A (EXP-004's prediction, restated).
- **T-H2 (RL lift).** Arm C meets SFT-H1 and SFT-H2 vs arm A **independently** — matching arm B
  is not sufficient — and its false-refusal rate is ≤ 0.10 (the SFT-H3 cap).
- **T-H2′ (reward vs filter; between validated arms only).** Arm C is *non-inferior* to arm B on
  the human-audited claim-support rate iff the lower bound of the paired-bootstrap 95% CI on
  (C − B) is > **−0.05** absolute, and *superior* iff the whole CI is > 0. A CI that straddles both
  is **inconclusive** and B is preferred; a wide CI never counts as equivalence.
- **T-H3 (no reward hacking).** Sampling unit = eval query. Per-query statistics, computed on
  non-abstaining answers only for (a) and (b): (a) citations per 100 words, (b) claims per answer;
  (c) is the arm's abstention rate on answerable queries (a proportion over all answerable eval
  queries; an empty or unparseable output counts as a non-abstaining answer with zero claims).
  **Comparison statistic:** for (a) and (b) the trained arm's **per-query median** is compared with
  the band; for (c) the arm's rate is compared with the band. **Bands**, fixed from arm A before
  arms B/C are scored: (a) and (b): arm-A median ± 1.5 × the arm-A interquartile range of per-query
  values, with ± 0.5 × the arm-A median if that IQR is 0; if the arm-A median itself is 0 the band
  cannot be derived from A and the pre-stated absolute fallback band applies instead — (a)
  [0.5, 8] citations per 100 words, (b) [1, 12] claims per answer. (c): [0, arm-A rate + 0.10],
  one-sided. **Failure rule:** a violation in the hacking direction — (a) above band, (b) below
  band, (c) above band — **fails the arm regardless of T-H1/T-H2**; (a) below band or (b) above
  band is reported, not failed.
- **T-H4 (retention).** EXP-004-A1's open-book overlap gate, its minimum-power fallback and its
  closed-book backstop (a ship-block) apply to every trained arm.

## Validation rule and what ships

Nothing from EXP-005 ships (EXP-004-A1 §4). A recipe is **validated** only if its arm clears, on
its own against arm A: SFT-H1 ∧ SFT-H2 ∧ the SFT-H3 cap ∧ T-H3 ∧ T-H4. T-H2′ never validates a
recipe; it only chooses between two validated ones.

- **both validated** → Phase B re-runs arm C's recipe if T-H2′ is superior or non-inferior,
  otherwise arm B's (simpler and cheaper);
- **one validated** → Phase B re-runs that recipe;
- **neither** → "RAG-grounded beta on the base model, training iterating"; M6 ships base + RAG.

## Compute and budget

Arm B: one QLoRA run on an RTX 4090 spot instance (R1 estimate $2–4) at the D1 sequence length
(likely 16K; re-measure). Arm C: GRPO with colocated generation (TRL `GRPOTrainer` + vLLM, or
Unsloth GRPO), 4–8 samples per prompt, ~500–1,000 prompts, on a 48 GB spot instance (L40S/A6000);
budget cap **$80** for the arm, two attempts maximum. Reward computation must run on the same GPU
in < 1 s per sample or arm C is infeasible at this budget — measured before the run.

## Open variant (not part of the primary comparison)

Ground-GRPO reports that reasoning-augmented policies beat instruction-only ones on unanswerable
queries. The 2026-07-21 no-think decision was driven by hand-authoring cost and verifier scope,
neither of which binds an RL arm (traces are model-generated and the reward reads only the final
answer). A **C-think** arm is therefore a legitimate secondary variant *if budget remains after the
primary three arms*; it would be compared to arm C under the same gates and reported as a variant,
never substituted for the primary mode-matched comparison.

## Confounds and controls (added to EXP-004's)

- **Reward/judge leakage.** The verifier is the reward (arm C) *and* the H2 instrument. Control:
  a **blinded, paired human audit** — the same 30 randomly drawn eval queries for every arm (A, B,
  C), arm identity hidden, every extracted claim labelled supported/unsupported by the human.
  Per-arm agreement with the verifier must be ≥ 0.85 (percent agreement on claims). If an arm falls
  below the floor, its verifier-scored H2/T-H2 numbers are **invalid**; the human-labelled rate on
  the audited slice, with its CI, is the only number that can clear the gate for that arm. T-H2′
  compares the human-audited rate in every case.
- **Prompt-set / eval-set disjointness.** RL prompts and SFT queries are disjoint from the eval
  partition at the query level and, where the corpus allows, drawn on held-out abstracts.
- **Sequence-length confound.** All arms train and are evaluated at the D1 context; any arm that
  needed a shorter context to fit memory is reported as such and not compared head-to-head.

---

*Pre-registration commit. Results, interpretation and verdicts are appended at the results step
(separate commit), after M3 (verifier trust decision) and the runs.*
