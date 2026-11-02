# AstroLLM Research Log

A living document tracking experiments, findings, and decisions.

!!! tip "Claude Code users"
    Use the `/research-log` command to add structured entries automatically.

---

## Experiment Template

```
### EXP-XXX: [Title]
**Date**: YYYY-MM-DD
**Type**: training | retrieval | data | evaluation | infrastructure
**W&B Run**: [link]
**Config**: configs/[filename].yaml

**Hypothesis**: What we expect to happen and why.

**Setup**:
- Base model:
- Method:
- Dataset:
- GPU:
- Training time:

**Results**:
| Metric | Base Model | Previous Best | This Run | Delta |
|--------|-----------|---------------|----------|-------|
| AstroMLab-1 (subset) | | | | |
| Grounding accuracy | | | | |
| Tool routing F1 | | | | |
| Abstention recall | | | | |
| Pedagogy score | | | | |

**Observations**: What we learned.

**Next Steps**: What to try based on these results.
```

---

## Experiments

### EXP-000: Baseline Measurements
**Date**: TBD
**Status**: Pending

**Goal**: Establish baseline scores for unmodified base models on our evaluation suite.

**Models to evaluate**:
- [ ] Qwen3.5-9B (primary base model)
- [ ] Qwen3.5-4B (smaller experiment)
- [ ] Gemma 4 E4B (Track B cross-family comparison)
- [ ] Qwen 2.5 7B Instruct (prior-generation comparison)

**Metrics**: AstroMLab-1, Astro-QA, custom pedagogy eval, perplexity on held-out astro text

---

> **Retrieval experiments (EXP-001 – EXP-003) use a condensed entry.** The EXP-XXX template above is
> training-shaped (W&B run, GPU, AstroMLab metrics), which doesn't fit a retrieval ablation, so these
> log the date / type / status / outcome and link the per-experiment doc for full method,
> pre-registration, tables, and per-query results.

### EXP-001: Pilot retrieval ablation — dense vs lexical vs hybrid
**Date**: 2026-05-31 · **Type**: retrieval · **Status**: complete (PR #5) · **Detail**: [pilot-ablation.md](research/pilot-ablation.md)

500-abstract ADS corpus (`abs:"exoplanet atmosphere"`, year ≥ 2018), 29-query reviewed gold. Three
arms over one frozen index — dense (pgvector/BGE), lexical (FTS5/BM25), hybrid (RRF; k=10, pool=50,
RRF_K=60). Hybrid wins MRR on every split; the Recall@10 ordering sits within the paired-bootstrap
noise at n=29. The decision driver was **candidate-set recall**, not Recall@10: union recall at pool
depth = 0.966, and q12's dense-blind ERO (dense #338 / lexical #4) is an existence proof that lexical
recovers targets dense buries. **Outcome: hybrid selected as the stage-1 candidate generator.**

### EXP-002: Corpus widening (500 → 2,500), method-fixed
**Date**: 2026-05-31 · **Type**: retrieval · **Status**: complete (PR #6, merge `d84f3ff`) · **Detail**: [corpus-widening.md](research/corpus-widening.md)

Same query + year, deeper citation cut; the original 500 a byte-identical strict subset. Vary the
corpus, hold the method.

- **H5 confirmed** — every arm loses Recall@10 at fixed pool=50 (hybrid least, −0.098; dense −0.195;
  lexical −0.276), and hybrid becomes the best Recall@10 arm.
- **H6 confirmed, via incumbent displacement** — the single-arm-exclusive class scales 5 → 15, the
  union premium grows +0.069 → +0.172, and union R@50 holds 0.948.
- **H7 falsified by construction** — the coverage premise collapsed: q15 (LHS 475b) is
  phrase-excluded and q11's canonical inversion paper is year-excluded, so a deeper citation cut
  yields **zero coverage win** (the bottleneck is a query-recall bound, not corpus size).
- Edge robustness: the dense−hybrid Recall@10 edge is **fragile** (carried by 4 queries, collapses
  under leave-one-out); the lexical−hybrid edge is **robust** (survives all 29 LOO drops).

### EXP-003: Pool-depth sweep (frozen 2,500 index)
**Date**: 2026-06-01 · **Type**: retrieval · **Status**: complete (PR #7, merge `22d2793`) · **Detail**: [pool-sweep.md](research/pool-sweep.md)

Frozen-index ablation (no re-index); sweep only `pool ∈ {50, 100, 200, 500}`.

- **H8 confirmed (strongly)** — candidate-union recall rises to **1.000** with depth, but **fused
  top-10 is exactly pool-invariant** (0.592 all / 0.716 named) and fused top-50 barely moves
  (non-monotonically); the gap does not close. The bottleneck is **RRF fusion demotion of gold that
  is already in the candidate set**, not candidate absence.
- **H9 falsified** — both arm edges are pool-invariant, which is *entailed* by the top-10 invariance
  (not an independent result): the fragile dense−hybrid edge stays fragile and the robust
  lexical−hybrid edge stays robust at every depth.
- **H10 lever call** — the lever is stage-2 reranking, not pool depth; operating point **pool = 100**.

### EXP-004: RAG-SFT pilot — curation-recipe pre-registration (Phase A)
**Date**: 2026-06-01 · **Type**: data · **Status**: pre-registered (PR #10), no data generated · **Detail**: [sft-pilot.md](research/sft-pilot.md)

Executes the 2026-06-01 pivot (Decision Log below): from the closed retrieval thread to SFT data
curation, the active critical path for the week-12 beta. Phase A holds the exoplanet-atmosphere
corpus **frozen** (the 2,500-abstract pool-sweep index, context assembled hybrid RRF @ pool=100) and
treats the **curation recipe** as the single manipulated variable — gold seed (150–250 hand-curated,
disjoint verifier-calibration / eval-seed partitions) → teacher synthesis (Claude API, grounded on
the frozen `retrieve()`) → in-repo citation-grounding verification → manifest + 95/5 split by task
family. Same pilot→widen, single-variable discipline as the retrieval thread; Phase B (separate
commission) freezes the recipe and widens the corpus to a general astro-ph slice for the shippable v1
set. Every example is retrieval-augmented `(query + retrieved abstracts) → cited answer` across three
families — literature-grounded QA (~45%), citation-grounded summarization (~35%), explicit abstention
(~20%, deliberately over-weighted).

Five hypotheses registered **before any data is generated**; comparison is always QLoRA-SFT
Qwen3.5-4B vs the same base on the held-out eval-seed set, with paired-bootstrap CIs as in the
retrieval thread:

- **SFT-H1 (grounding lift)** — CI on (SFT − base) faithfulness excludes 0, target ≥ +0.10 absolute.
- **SFT-H2 (citation accuracy)** — cited-bibcode-supports-claim clears the V1 >80% target and beats base.
- **SFT-H3 (abstention, two-sided)** — refusal rate beats base, subject to a false-refusal cap ≤ 0.10.
- **SFT-H4 (no knowledge regression)** — AstroMLab-1 subset drops ≤ 2 pp vs base (larger = kill signal).
- **SFT-H5 (verifier validity)** — verifier precision/recall vs gold; trusted as a filter only at
  precision ≥ 0.85, else every "passed" example is human-reviewed.

**Gate**: fine-tune into the beta only if H1 ∧ H2 ∧ H4 hold; otherwise the honest status is
"RAG-grounded beta on the base model, SFT iterating" — the week-12 date does not override the gates.
Pre-registration only: nothing is fine-tuned and no SFT data is curated yet. Results /
Interpretation / Verdict append at the results step (separate commit).

### EXP-004-A1: SFT-H4 amendment — open-book gate + pre-data operationalization
**Decision dates**: 2026-06-02 (amendment, PR #12) and 2026-07-21 (operationalization, PR #13) ·
**Recorded**: 2026-07-21 · **Type**: data · **Status**: pre-data (0 of 150–250 gold examples
authored) · **Detail**: [sft-pilot.md § Amendments](research/sft-pilot.md#amendments)

Amends the EXP-004 entry above, which is append-only and therefore still carries the original
SFT-H4 wording. **Old gate**: closed-book AstroMLab-1 subset drops ≤ 2 pp vs base, larger = kill.
**New gate** (2026-06-02, commits `a60e1c9` + `6514900`): open-book topic-overlap kill (MCQ with
relevant retrieved context supplied, base and SFT in the same mode); closed-book reported as
expected drift with a catastrophe backstop; comprehensive general-astronomy retention gated at
Phase B. Rationale: closed-book MCQ conflates parametric knowledge loss with by-design abstention
hedging in a context-trained model. **Operationalization** (2026-07-21, after adversarial review):
the topic-overlap slice is selected by a committed pre-output keyword filter with n reported before
the eval; a minimum-power fallback (n < 40 or base CI half-width > 5 pp ⇒ point estimate ≥ base
− 5 pp) prevents a wide CI from passing by default; the backstop is defined (closed-book drop
> 10 pp or accuracy < 35% on 4-option MCQ) and is a **ship-block**; and no Phase-A fine-tune ships
into the beta under any result — Phase A validates the recipe, only a Phase-B run passing the
Phase-B pre-registered retention gate ships. H1/H3 rubrics and judge identities are committed
before the eval runs; H2 adds citation-density reporting + a spot-check agreement floor; H5 adds
recall + dropped-example distribution reporting.

**Convention** (stated here for future entries): amendments to pre-registered experiments are
recorded as appended `EXP-XXX-A#` entries plus a dated bullet in the experiment doc's Amendments
section; the original entry and pre-registration text are never modified.

### EXP-004-A2: gold seed = teacher-drafted, human-verified; model-facing context = top-k
**Decision date**: 2026-11-02 · **Type**: data · **Status**: pre-data (0 of 150–250 gold examples
authored) · **Detail**: [sft-pilot.md § Amendments](research/sft-pilot.md#amendments)

Second amendment to EXP-004 (original and A1 unchanged). Adopted with V1 Plan R2 after the
[2026-10 plan review](research/plan-review-2026-10.md). (1) The teacher (`claude-opus-5-5`,
recorded per record) drafts the cited answer and claims; the human verifies every claim's support
span and sets every label — the human labels, not the prose, remain the verifier-calibration and
eval reference. (2) `retrieved_context` keeps the full pool-100 output for provenance but every
consumer (worksheet, teacher, SFT/RL prompts, serving) sees the top-k slice, k fixed by decision
D1 before EXP-000 — the harness's frozen K=100 context (~25–30K tokens) contradicted the
`max_seq_length: 4096` in every committed training config. (3) Eval partition is authored first,
calibration second. (4) Eval metrics are judged against context, never against gold answer text;
judge ≠ teacher family. (5) Teacher-selection-bias control: every calibration negative and ≥ 25%
of calibration positives are human-authored from scratch (a challenge subset on which H5 is
reported separately and must also clear the ≥ 0.85 trust gate), and every teacher-drafted record
gets a context-wide answerability review. Targets, families, gates and "Phase A never ships" are
unchanged.

### EXP-005: Training pilot — base vs SFT vs GRPO-with-verifier-reward (Phase A, pre-registration)
**Date**: 2026-11-02 · **Type**: training · **W&B Run**: [pending] · **Config**: committed under
`configs/` at the run · **Status**: pre-registered, nothing run · **Detail**:
[training-pilot.md](research/training-pilot.md)

Extends EXP-004 with a third arm. Fixed: the frozen 2,500-abstract corpus, hybrid RRF @ pool 100
with the D1 serving context, Qwen3.5-4B QLoRA, no-think for all arms, the EXP-004 eval partition,
judge ≠ teacher. Arms: **A** prompted base; **B** SFT on teacher-synthesized, verifier-filtered
data (the EXP-004 recipe); **C** GRPO with a committed programmatic reward — citation validity
(hard), calibrated claim support (dense), two-sided abstention, format — and **no judge-scored
answer-correctness term**. Arm C runs only if the verifier clears the H5 trust gate (precision
≥ 0.85); budget cap $80, two attempts.

Pre-registered: **T-H1** arm B meets SFT-H1/H2 vs A; **T-H2** arm C meets SFT-H1/H2 vs A
*independently* with false-refusal ≤ 0.10; **T-H2′** (between validated arms only) C is
non-inferior to B on the human-audited claim-support rate iff the lower 95% CI bound of (C − B)
is > −0.05, superior iff the CI is > 0, else inconclusive and B is preferred; **T-H3**
reward-hacking tripwires with bands fixed from arm A before B/C are scored (the trained arm's
per-query median vs arm-A median ± 1.5 IQR for citation density and claims per answer, with
pre-stated absolute fallback bands if A's median is 0; a one-sided +0.10 band for abstention on
answerable queries; a violation in the hacking direction fails the arm regardless, the opposite
direction is reported); **T-H4** the
EXP-004-A1 retention gate, min-power fallback and closed-book ship-block apply to every trained
arm. A recipe is validated only by clearing SFT-H1 ∧ H2 ∧ the H3 cap ∧ T-H3 ∧ T-H4 on its own
against arm A; T-H2′ only chooses between validated recipes. Arm C additionally needs
human-audited answerability labels on its prompt set (100% of non-answerable, 20% of answerable;
> 10% label error ⇒ not run), claim extraction over every declarative sentence incl. quotations (uncited = unsupported; zero
claims, zero citations or zero words on a non-abstention ⇒ reward floor; citation validity is
checked on every answer, pure abstentions included, before any other component; answers are
classified by the shared D4 answer-format contract, implemented before M2; an arm with < 10
non-abstaining eval answers fails T-H3), and a blinded,
paired 30-query human audit per arm with a 0.85 agreement floor that invalidates verifier-scored
numbers below it. Nothing from EXP-005 ships. A C-think variant is a pre-registered secondary arm
if budget remains, never substituted for the mode-matched primary comparison.

---

## Key Learnings

### From AstroSage Papers
1. **Catastrophic forgetting is real**: Early AstroLLaMA (7B) scored LOWER than base LLaMA after CPT
2. **Data quality > quantity**: Curated, diverse SFT data matters more than raw corpus size
3. **Model merging saves capabilities**: SLERP/TIES merging of SFT model with instruct model preserves reasoning
4. **Loss masking is essential**: Train only on assistant completions, not user prompts
5. **Synthetic data works**: Generated Q&A pairs with randomized question styles improve diversity
6. **Metadata-based Q&A helps**: Questions about paper titles, dates, arXiv IDs build factual grounding

### From "Talking with the Latents" (2026)
1. **Teacher-student distillation**: Large LLM generates synthetic Q&A → smaller LLM learns from it
2. **LoRA is sufficient**: Lightweight adapters + frozen base achieves strong domain performance
3. **Latent feature fusion**: Physical features can be injected into LLMs via adapters

---

## Decision Log

| Date | Decision | Rationale |
|------|----------|-----------|
| TBD | Start with Qwen3-8B | Apache 2.0 license, strong fine-tuning results at 4B/8B scale, dual thinking mode |
| TBD | QLoRA for initial experiments | Cloud-only, need to minimize GPU costs per experiment |
| TBD | Researcher-first audience | Highest value differentiation, tool integration is unique angle |
| 2026-04-23 | Upgrade base models from Qwen3-4B/8B to **Qwen3.5-4B/9B** | Qwen3.5 released Feb 2026 (Apache 2.0, natively multimodal, 262K ctx). Same family preserves plan/tooling/eval continuity. V1_FINAL_PLAN.md remains frozen; this Decision Log entry overrides it. |
| 2026-04-23 | Add **Gemma 4 E4B** as Track B comparison arm | Gemma 4 released Apr 2, 2026. Unsloth day-0 support (~2× faster training, ~60% less memory). Cheap cross-family data point via existing week 7-8 experiment matrix. Dense variant only — Gemma 4 MoE has a known 3D-tensor QLoRA bug. |
| 2026-04-24 | **Watch**: Qwen3.6 dense 4B/9B variants | As of 2026-04-24, only Qwen3.6-27B (dense, coding-focused) and Qwen3.6-35B-A3B (MoE) are released as open weights; 4B/9B are "expected" but not shipped. If dense 4B/9B land before week 7, revisit base-model choice. Otherwise ship v1 on Qwen3.5 and evaluate 3.6 in Phase 2. |
| 2026-05-30 | Repo-wide reconcile to Qwen3.5 base + verified HF repo IDs | Propagated the Qwen3.5-4B/9B + Gemma 4 E4B Track-B decision across all configs, docs, personas, and skill commands, superseding the original Qwen3-4B/8B (and earlier Llama 3.1) references. Note 9B, not 8B — Qwen3.5 has no 8B dense. Verified exact HF repo IDs against Hugging Face: `Qwen/Qwen3.5-9B` and `Qwen/Qwen3.5-4B` (post-trained; no `-Instruct` suffix, `-Base` = pretrained), and `google/gemma-4-E4B-it` (capital `E4B`). |
| 2026-05-31 | Docker Compose project-name collision (resolved) | During pilot stack bringup, `docker compose` derived the project name "docker" from the `docker/` directory and collided with an unrelated local Docker project that derives the same name; Compose recreated that project's database container against the pilot volume. Remediated by rebinding the other project's data volume and re-isolating the pilot stack. Structural fix: added a top-level `name: astrollm-pilot` key to `docker/docker-compose.pilot.yml` so isolation is a property of the file, not a `-p` flag (landed in PR #3, merge cfcd21b). **OPEN**: independent integrity verification of the other project's database (row counts) is still pending — the self-remediation does not close it. |
| 2026-05-31 | Pilot label review: eval baseline updated | First-pass eval (PR #3): Recall@10 0.812 / MRR 0.776 over 16 queries with title-level entity-matched labels. After full review (PR #4; 29/30 scored, q15 has no in-corpus target): overall Recall@10 0.690 / MRR 0.623; named-target (n=17) 0.794 / 0.794; broad known-item (n=12) 0.542 / 0.381. The headline moved **down** by intent — the review added an honest q11 0.00 miss and corrected mislabels in both directions rather than inflating. **Decision**: the reviewed gold set is the eval baseline going forward. **Finding** (treat as DIRECTIONAL): hybrid retrieval is markedly weaker on conceptual/topical queries than on named entities — but the two sets used different labeling protocols (broad = single known-item landmarks, so its number is optimistic) and n is small (12 broad / 17 named), so the pattern is robust while the decimals are noisy. Motivates the dense-vs-lexical-vs-hybrid ablation and the stage-2 reranker. |
| 2026-06-01 | **Retrieval thread closed** — beta ships hybrid RRF stage-1 @ pool=100 | Stage-1 candidate generation is **not** the binding constraint (union recall ≥ 0.966, and 1.000 for named queries — see EXP-001/002/003); the bottleneck is **fusion ranking**. The beta default is therefore **hybrid RRF stage-1 at pool=100** (saturates the recoverable fused recall at ~157 candidates, per EXP-003). Fusion-ranking improvement — the ladder **RRF_K sweep → weighted/score-based fusion → cross-encoder reranking** — is **deferred to the post-beta retrieval cycle** (aligns with V1_FINAL_PLAN placing Stage-2 reranking post-week-12). The **query-widening** lever (to reach the query-excluded coverage targets q15/q11) is also deferred. The active critical path **pivots to SFT data curation** (plan weeks 5-6). Retrieval was stopped because it is **ship-good, not because it failed**. **Dissemination**: PR #6 → blog Post 3, PR #7 → blog Post 4 (nandan.me/writing/notes, "building on a budget" series); drafting deferred, tracked. |
| 2026-07-21 | **Plan re-anchor after 7-week pause; calendar-week labels retired from revisable docs** | Work paused 2026-06-02 and resumed 2026-07-21. The V1 plan's week numbers are effort-nominal, not calendar-locked (plan "weeks 1-2" of retrieval work took 3 calendar days; then 7 calendar weeks passed with zero plan-weeks of progress), so revisable docs now use milestone labels ("Phase A gold-seed authoring — in progress") instead of week numbers. "Week-12 beta" is read as the *beta milestone*; a calendar date is re-committed when Phase A completes — per the gates-over-dates principle already in the EXP-004 pre-registration. Also recorded: the plan's weeks-3-4 first-copilot demo (off-the-shelf Gemma 4 E4B + simple UI) was skipped in the 2026-06-01 pivot from retrieval directly to SFT; it remains unbuilt, deferred to the integration milestone. |
| 2026-07-21 | **Base-model decision re-affirmed: Qwen3.5-4B/9B primary + Gemma 4 E4B Track B** | Landscape re-check (HF-verified 2026-07-21): Qwen3.6 still ships no small dense model (27B / 35B-A3B only, unchanged since the 2026-04-24 watch entry); no new 3-10B open-weight release since 2026-05-30 beats Qwen3.5-4B/9B for RAG-grounded citation/abstention SFT; AstroSage line (Llama-3.1-based) and AstroMLab-1 unchanged — a Qwen3.5 astronomy tune would be first-of-family with a ready benchmark. **Watch item**: `deepreinforce-ai/Ornith-1.0-9B` (MIT, 2026-06-21) — an RL-reasoning tune *of* Qwen3.5-9B with GGUFs; candidate alternative *starting checkpoint* for the 9B arm, revisit at fine-tune time; not a base-family change. **Recorded nuance**: Qwen3.5 small models are multimodal with thinking-mode-on by default — `<think>`-trace handling (disable vs curate) must be decided **before** gold-seed authoring so the gold answers match the training-time template; serving must support the hybrid Gated-DeltaNet arch (vLLM + llama.cpp GGUF support confirmed). |
| 2026-07-21 | **Freeze-policy violation acknowledged; frozen baseline redefined at `1739826`** | The 2026-04-23 entry above states "V1_FINAL_PLAN.md remains frozen; this Decision Log entry overrides it" — but commits `73b2bc3` and `1739826` (PR #2, 2026-05-30) in fact edited `V1_FINAL_PLAN.md` (and `73b2bc3` edited `MASTER_PLAN.md`) to reconcile base-model references, contrary to the freeze rule. Resolution: the current text is **not** reverted (reverting would resurrect Qwen3-4B/8B references that no config or doc matches, and git preserves the original at `e2406e8`); the frozen baseline is redefined as the state at `1739826`; `.claude/rules/docs.md` now codifies that supersessions are recorded only in this Decision Log and that a Decision Log entry cannot authorize editing the frozen files. No banner is added to the frozen docs — that would be a further edit. |
| 2026-07-21 | **Docker DB integrity verification: still OPEN** (restating the 2026-05-31 item) | The independent row-count verification of the collided project's database was not performed during the pause. Remains OPEN. Trigger: verify before that project's DB is next used, and before treating this Docker host's state as trustworthy for anything load-bearing; the pilot stack itself was re-isolated by the `name: astrollm-pilot` fix (PR #3) and is unaffected. |
| 2026-07-21 | **Phase A trains and serves no-think** (Qwen3.5 `<think>`-trace handling) | Qwen3.5 small models default to thinking-mode-on; the gold seed defines the answer format, and Phase B re-runs the recipe frozen, so trace handling had to be decided pre-authoring. **Decision: thinking disabled** — gold answers are the cited answer + claims only, exactly the harness worksheet shape. Rationale: (1) hand-authoring plausible traces for 150–250 examples roughly doubles the bottleneck cost; (2) traces carry uncited intermediate assertions the pre-registered citation verifier cannot check without an un-pre-registered extension; (3) single-variable discipline — traces would be a second manipulated recipe variable; (4) the task is open-book grounded synthesis, not the closed-book parametric QA where AstroSage's trace-SFT precedent lives; (5) Gemma 4 E4B Track B has no native think block, so formats align across tracks. **Rider 1 (mode symmetry)**: every SFT-vs-base comparison (SFT-H1–H4, incl. the open-book MCQ slice) runs BOTH arms in no-think mode; base-with-thinking may be reported as a reference row but the gates read the mode-matched pair. **Rider 2 (template check)**: the concrete no-think mechanism (template flag vs empty think block — Qwen versions differ) is verified against the actual tokenizer before the fine-tune, not assumed. Trace-SFT remains a legitimate future recipe *variant*, pre-registered against the validated no-think baseline if ever tried. |
| 2026-11-02 | **V1 Plan Revision 2 adopted** — `docs/V1_PLAN_R2.md` supersedes `V1_FINAL_PLAN.md` (Revision 1) as the Phase-1 execution plan | Trigger: the [2026-10 plan review](research/plan-review-2026-10.md) after a second pause (07-21 → 10-23). the Master Plan's Phase 1–2 risk #1 (invisible progress → project death) materialized once Revision 1's protected week-4 demo was skipped; the current plan was spread across ten Decision Log rows and an amendments section. R2 is one document; `.claude/rules/docs.md` now defines revisions as whole, frozen-on-merge files. `MASTER_PLAN.md` vision unchanged. |
| 2026-11-02 | **Copilot v0 is the next milestone (M1); training is not a prerequisite for the beta** | Base Qwen3.5-9B + the as-built retriever as a tool + `astroquery-mcp` + a citation-enforcing prompt + a bibcode citation guard, CLI first. It is the skipped R1 week-4 demo and the plan's own fallback outcome. **First three M1 tasks**: (1) decide D1 by scoring fused recall at cuts {10, 20, 30, 50} with the pool-sweep harness; (2) a `uv run` chat CLI over the frozen retriever with the citation guard; (3) wire `astroquery-mcp` (pinned) for SIMBAD/ADS under R2's evidence contract (grounded mode = frozen top-k only, the evaluated path; live ADS results only in a separately logged, never-trained-on live mode). EXP-000 baselines follow immediately (M2). |
| 2026-11-02 | **Serving-context decision D1 opened: train/serve context is top-k of the pool-100 ranking, k to be measured before EXP-000** | Found in review: `context.py` freezes K=100 abstracts (~25–30K tokens) as each example's context while all three training configs cap `max_seq_length` at 4,096. The pre-registration's train/serve equality cannot hold as written. Full pool kept for provenance; every consumer sees top-k; `max_seq_length` raised to fit. R2 commits the measurement: delivered-context token budget 16,384 (Qwen3.5 tokenizer), whole-abstract overflow drops, recall scored on the delivered context, selection rule and fallback fixed before any model output is seen. |
| 2026-11-02 | **Tools layer adopted, not built**: NASA-IMPACT `astroquery-mcp` (pinned) for SIMBAD/ADS; `tools-integration` retired as a build target; the project's retriever to be published as an MCP server | Verified 2026-10-23: 141 functions / 14 services, Apache-2.0, Claude Code plugin, backend of NASA's AKD agent + AstroAKD-Bench; `cbyrohl/mcp-server-ads` for citation graphs. Tool access is now table stakes; differentiation moves to an open small model that cites faithfully + a public eval set. Serving loop calls astroquery directly; MCP is the interface for Claude clients. |
| 2026-11-02 | **EXP-004-A2** (gold = teacher-drafted, human-verified; top-k context; eval partition first) and **EXP-005** (three-arm training pilot incl. GRPO with the verifier as reward) recorded | See the entries above and `docs/research/training-pilot.md`. Basis: arXiv 2609.23053 (RL on answer correctness does not fix post-rationalized citations), 2506.15522 (Ground-GRPO), 2609.28765 (dense rewards for small search agents). |
| 2026-11-02 | **Landscape re-check (verified 2026-10-23)**: base-model decision holds; Gemma 4 12B dense (2026-05-23) added to the watch list, not as an arm; AstroSage unchanged (70B-20251009); prior art to cite: Pathfinder (2408.01556), Astro2Method (2610.30282) | Qwen3.6 and Qwen3.8 (08-2026) ship no dense model below 27B. `ornith-ai/Ornith-1.0-9B` remains the alternative 9B checkpoint. Next re-check at fine-tune time. |
| 2026-11-02 | **Process rules R2**: pre-registration only for runs that spend GPU money or gate a ship decision; every session ends with a runnable artifact or a measured number; per-session lab notes replace weekly reports; one Codex round per PR, run last | Two of the last four merged PRs were docs-only, and the two code PRs built and tested a harness that has produced zero examples. The research-log and amendment conventions are unchanged; the rigor is kept where it decides something. |
