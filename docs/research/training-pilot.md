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
| **C. GRPO** | on-policy rollouts from `(query + context)` prompts scored by a programmatic reward | prompts only (machine-assembled); verifier as reward |

Arm C runs **only if** the verifier passed the H5 trust gate (precision ≥ 0.85 on the calibration
partition). Below that, the reward is too noisy to be a training signal and arm C is recorded as
"not run — verifier below gate", not skipped silently.

## Reward specification (arm C) — committed before the run, in this order

The reward is a weighted sum of components that are each independently checkable; weights are
committed in the run config and reported.

1. **Citation validity** (hard): every bibcode in the answer ∈ serving context. Any invalid
   citation → reward floor for the sample.
2. **Claim support** (dense): fraction of extracted claims whose cited abstract supports them per
   the calibrated verifier. Claims are extracted by a fixed rule (one claim per sentence carrying a
   citation), never by the policy model.
3. **Abstention correctness** (two-sided): +1 for correctly abstaining on thin/absent/contradictory
   context (as labeled in the prompt set), −1 for abstaining on an answerable prompt. The prompt set
   mirrors the eval-seed family mix (~45/35/20) so over-abstention cannot pay.
4. **Format**: cited-answer shape, length band, no reproduced abstract text beyond quotation
   length.

Dense over binary (2609.28765); no answer-correctness term from a judge (2609.23053 — it does not
buy faithfulness and it would reintroduce the teacher/judge shared bias).

## Hypotheses (pre-registered)

Comparison is paired on the eval partition with bootstrap CIs, all arms no-think, judge ≠ teacher.

- **T-H1 (SFT lift).** Arm B meets SFT-H1 and SFT-H2 vs arm A. *(This is EXP-004's original
  prediction, restated so EXP-005 cannot pass on arm C alone.)*
- **T-H2 (reward ≥ filter).** Arm C's claim-support rate and faithfulness are ≥ arm B's (CI on
  C − B does not exclude 0 in B's favour), **and** arm C's false-refusal rate ≤ 0.10.
- **T-H3 (no reward hacking).** For arms B and C, citation density, claims per answer and
  abstention rate on answerable prompts stay within bands fixed from arm A's distribution
  (±1.5 × arm-A interquartile range). An arm outside any band **fails** regardless of T-H1/T-H2.
- **T-H4 (retention).** EXP-004-A1's open-book overlap gate and closed-book backstop apply to
  every trained arm.

## Kill criteria and what ships

Nothing from EXP-005 ships (EXP-004-A1 §4). The outcomes are:

- **recipe validated** (T-H1 or T-H2 with T-H3 and T-H4) → Phase B re-runs *that* recipe on the
  widened corpus with its own retention gate;
- **SFT only** (T-H1 holds, T-H2 fails or arm C not run) → Phase B uses the SFT recipe;
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

- **Reward/judge leakage.** The verifier is the reward (arm C) *and* the H2 instrument. H2 is
  therefore also spot-checked by a human on a fixed random slice with the pre-stated agreement
  floor; the human number is the one reported for T-H2.
- **Prompt-set / eval-set disjointness.** RL prompts and SFT queries are disjoint from the eval
  partition at the query level and, where the corpus allows, drawn on held-out abstracts.
- **Sequence-length confound.** All arms train and are evaluated at the D1 context; any arm that
  needed a shorter context to fit memory is reported as such and not compared head-to-head.

---

*Pre-registration commit. Results, interpretation and verdicts are appended at the results step
(separate commit), after M3 (verifier trust decision) and the runs.*
