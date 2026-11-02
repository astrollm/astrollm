# Plan Review — 2026-10-23 (recorded 2026-11-02)

*A status audit and plan review of AstroLLM after the second pause of 2026, with the model and
tooling landscape re-verified on 2026-10-23. The recommendations were accepted on 2026-11-02 and
became [V1 Plan Revision 2](../V1_PLAN_R2.md), amendment EXP-004-A2 and the EXP-005
pre-registration. The adversarial review at the end records the objections raised against the
recommendations and the adjustments they produced.*

---

## 1. Status at 2026-10-23

| Item | State |
|------|-------|
| Plan written | 2026-03-23 (7 months earlier) |
| Working days with commits | ~10, in three bursts: 04-05/06, 05-30 → 06-02, 07-20/21 |
| Pauses | 06-02 → 07-21 (7 weeks), 07-21 → 10-23 (13 weeks) |
| Python incl. tests | ~3,700 lines, 67 tests, CI green |
| Retrieval | Built and ablated (EXP-001–003): 2,500 ADS abstracts, hybrid RRF @ pool 100 |
| Gold SFT examples | 0 of 170–230 (220 candidate queries triaged, uncommitted) |
| Training script | Stub (imports commented out) |
| Model calls anywhere in the pipeline | None |
| Tools / inference / web chat | Empty / empty / starter template |
| Budget spent | ≈ $0 of $400/month |

**Solid.** The retrieval thread is honest science with a transferable finding (fusion ranking, not
candidate recall, is the bottleneck). The gold-seed harness is well engineered and tested. The
research log's pre-registration and amendment discipline is a genuine asset.

**Wrong.** The Master Plan's risk register puts "motivation loss from invisible progress" first
for Phases 1–2 and mitigates it with the week-4 demo that Revision 1 lists among its protected
milestones. The demo was skipped on 2026-06-01; both pauses followed. The
critical path then required hand-authoring 150–250 examples, each over a 100-abstract worksheet
(≈ 50–150 solo hours), for a Phase-A fine-tune that by pre-registration never ships. Two of the
last four merged PRs were documentation about process; the other two built and tested a harness
that has produced zero examples.

**A latent contradiction found during the review.** The harness freezes the *full* pool-100
output (≈ 25–30K tokens) as each example's context (`context.py`, `K = 100`), while all three
training configs set `max_seq_length: 4096`. The pre-registration's promise that the SFT context
distribution equals the serving distribution cannot hold as written. Fixed by decision D1 in R2.

## 2. Landscape, verified 2026-10-23

| Area | Finding | Consequence |
|------|---------|-------------|
| Qwen | Qwen3.6 (04-2026) = 27B dense + 35B-A3B; Qwen3.8 (08-2026) = 27B dense, 2.4T-A95B MoE, Flash-Next. **No dense model below 27B after Qwen3.5.** | Qwen3.5-4B/9B remain the right small bases |
| Gemma | **Gemma 4 12B dense** shipped 2026-05-23 (Apache 2.0, QAT q4_0 checkpoints, text-only "assistant" variants) | Added to the watch list; not an arm |
| Ornith | `ornith-ai/Ornith-1.0-9B` (MIT), RL tune of Qwen3.5-9B | Still a candidate 9B starting checkpoint |
| AstroSage | Latest `AstroMLab/AstroSage-70B-20251009`; AstroMLab 4 paper; no Qwen-family or new small model | "First Qwen3.5 astronomy tune" still true |
| Tools | **NASA-IMPACT `astroquery-mcp`**: 141 functions, 14 services (SIMBAD 18 fns, ADS), Apache-2.0, Claude Code plugin, backend of NASA's AKD data-search agent and AstroAKD-Bench. `cbyrohl/mcp-server-ads`: 11 ADS tools incl. citation graph | Adopt, don't build; tool access is no longer a differentiator |
| Citation faithfulness | arXiv 2609.23053: ~1 in 7 RAG citations post-rationalized; RLVR on answer correctness does not reduce it. VERITAS (TMLR 05-2026): faithfulness rewards in the RL loop | Validates the thesis; faithfulness needs its own objective and its own eval |
| Verifiable-reward RL | arXiv 2506.15522 (Ground-GRPO: rewards for correctness, citation sufficiency, refusal; no gold traces); 2609.28765 (GRPO + search on Qwen3.5-0.8B; dense rewards win) | The verifier can be a reward, not only a filter |
| Prior art | Pathfinder (arXiv 2408.01556): semantic literature review over 300K ADS abstracts. Astro2Method (2610.30282), ReplicationBench | Cite; position against |
| Claude | Opus 5.5 $4/$20, Sonnet 5.5 $2/$10, Haiku 5.5 $0.10/$0.50 per MTok | Teacher budget unchanged |

## 3. Recommendations (as accepted)

1. **Ship the skipped copilot v0 before any training** — base Qwen3.5-9B + the existing retriever
   as a tool + `astroquery-mcp` + a citation-enforcing prompt + an inference-time bibcode guard.
2. **Run EXP-000 baselines next, not last** — base ± RAG on the eval seed; the answer decides the
   role of fine-tuning.
3. **Gold seed = teacher-drafted, human-verified** (EXP-004-A2) — cuts per-example time ~5×;
   human judgment stays on the labels that calibrate the verifier.
4. **Add a GRPO arm with the verifier as reward** (EXP-005) — base / SFT / GRPO under the same
   gates; reward-hacking tripwires.
5. **Adopt the tool layer** — `astroquery-mcp` in, `tools-integration` out; publish the retriever
   as an MCP server.
6. **Make the eval set the public artifact** — queries, bibcodes, claim labels; no abstract text.
7. **Write a Plan Revision 2** — one current document; revisions frozen on merge.
8. **Lighten process** — pre-registration only for GPU spend or ship gates; every session ends
   runnable.

## 4. Adversarial review of the recommendations

Each objection was raised against the accepted list before R2 was drafted; the "adjustment" column
is what R2 / A2 / EXP-005 actually say as a result.

| # | Objection | Assessment | Adjustment made |
|---|-----------|------------|-----------------|
| 1 | **Teacher-drafted gold contaminates the eval.** If the same teacher drafts eval answers and synthesizes SFT data, the SFT arm is scored on teacher-shaped targets. | Partly real. The eval metrics (faithfulness, citation support, abstention) are judged against the *context*, not against gold answer text, so the gold *labels* matter more than the gold prose. The residual risk is judge/teacher style bias, which EXP-004 already controls. | A2: the drafting model is recorded per record; every claim's support span is verified by the human (not accepted by default); H1/H3 judge is a different family (D3); eval scoring never uses gold answer text as a reference. |
| 2 | **GRPO is a stretch for a solo learner on 5–10 h/week.** RL is finicky; a slow or noisy reward wastes the budget. | Real. | EXP-005: arm C is conditional on the verifier trust gate (precision ≥ 0.85); $80 cap, two attempts; reward latency measured before the run; SFT arm is the fallback; 4B only. |
| 3 | **Agentic retrieval breaks the train/serve contract.** A multi-call loop at serving time, with single-shot contexts at training time, re-creates the distribution mismatch EXP-004 was designed to avoid. | Real and important. | R2 scope: single-shot top-k is the serving default; multi-call loops are a flagged experiment, not the product path, until training contexts are assembled the same way. |
| 4 | **The 100-abstract context does not fit the training sequence length.** | Real (found during this review). | D1: serving context k chosen by measurement before EXP-000; full pool kept for provenance; `max_seq_length` raised to fit. |
| 5 | **Depending on an external v0.1 MCP server is fragile.** | Real but cheap to contain. | Pin the version; call astroquery directly inside the serving loop; MCP is the interface for Claude clients, not the serving path. |
| 6 | **About a hundred examples (the eval partition minimum is 95) is not a "benchmark".** | Fair. | Called *eval set v0*; n and bootstrap CIs reported everywhere. |
| 7 | **Lightening process could erode the rigor that makes the project publishable.** | Partly real. | Pre-registration kept for GPU runs and ship gates (where it matters); dropped for harness/labeling/UI work; research-log conventions untouched. |
| 8 | **Adding Gemma 4 12B dilutes a tiny budget.** | Real. | Watch list only. |
| 9 | **No-think may handicap the RL arm** (Ground-GRPO finds reasoning helps on unanswerables). | Plausible. | EXP-005 keeps no-think for the primary mode-matched comparison; a C-think variant is a pre-registered secondary arm if budget remains. |
| 10 | **No local GPU for serving a 9B.** | Not blocking. | llama.cpp Q4_K_M on Apple Silicon, or a hosted Qwen3.5-9B endpoint for v0; quantization recorded on every eval row. |
| 11 | **If base + RAG already clears 80% citation support, the training thread is hollow.** | That is a good outcome for the product and the honest one to find early. | Training continues under the learning mandate with unchanged gates; the beta ships either way. |
| 12 | **R2 could itself become a docs-only burst followed by a pause.** | The real risk. | M1 is specified so the next session starts building; this PR names the first three M1 tasks (see Research Log 2026-11-02). |

## 5. External (Codex) review

Targeted plan-review rounds (design only), run on the committed diff against `main` under the
local review policy (`CODEX_ROUND=targeted:v1-plan-r2`, effort high, read-only).

### Round 1 — verdict NEEDS-ATTENTION: 2 P1, 8 P2, 3 P3. All accepted.

| # | Sev | Finding (file) | Change made |
|---|-----|----------------|-------------|
| F3 | P1 | EXP-005's outcome rule "T-H1 or T-H2" let arm C validate a recipe by merely matching arm B, even if both failed the inherited gates (`training-pilot.md`) | Validation is now per recipe, each arm on its own vs arm A: SFT-H1 ∧ H2 ∧ H3-cap ∧ T-H3 ∧ T-H4. T-H2′ only chooses between validated recipes |
| F9 | P1 | R2 said AstroMLab-1 is "never the ship gate", contradicting A1's open-book retention gate and closed-book ship-block (`V1_PLAN_R2.md`) | Rewritten: open-book slice = Phase-A retention gate; closed-book catastrophe thresholds = ship-block; only ordinary closed-book drift is ungated |
| F1 | P2 | D1 was not operationalized: no tokenizer, overflow policy, prompt/answer reserve, or fallback when no k qualifies; recall scored on ranks, not delivered context | D1 now fixes the 16,384-token delivered-context budget, reserves, whole-abstract overflow drops, delivered-context recall, selection rule and fallback, all before any model output |
| F2 | P2 | Human verification of teacher-chosen claims does not recover what the teacher omitted; H5 could certify a verifier on teacher-shaped examples (`sft-pilot.md` A2) | A2 point 6: human-authored challenge subset (all calibration negatives + ≥ 25% of positives), context-wide answerability review per record, H5 reported and gated on the challenge subset separately |
| F4 | P2 | "CI does not exclude 0 in B's favour" turned failure to detect inferiority into non-inferiority | T-H2′: non-inferior iff lower 95% CI bound of (C − B) > −0.05; superior iff CI > 0; otherwise inconclusive and B preferred |
| F5 | P2 | Only cited sentences became claims, so the policy controlled what the reward examined; zero-claim answers undefined | Every declarative sentence is a claim; uncited = unsupported; zero claims or zero citations on a non-abstention = reward floor |
| F6 | P2 | "Prompts only" hid a required labelled dataset: the abstention reward needs answerability labels for 500–1,000 prompts | New section: teacher-proposed labels with evidence pointers, 100% human audit of non-answerable and 20% of answerable, > 10% error ⇒ arm C not run, reward-only metadata, budgeted |
| F7 | P2 | T-H3 "±1.5 × IQR" had no centre, no unit, and a zero-IQR failure mode for binary abstention | Per-query median ± 1.5 IQR (or ± 0.5 median if IQR = 0) for density and claim count; one-sided +0.10 proportion band for abstention; hacking direction defined; bands fixed from arm A first |
| F8 | P2 | The human reward audit had no size, sampling unit or consequence | Blinded, paired 30-query audit per arm, 0.85 per-arm agreement floor, verifier numbers invalid below it; T-H2′ always uses the human rate |
| F10 | P2 | Live ADS evidence had no defined relationship to the D1 context contract or the citation guard | R2 evidence contract: grounded mode (frozen top-k only; default; the evaluated path) vs live mode (opt-in; separate log; never evaluated or trained on); SIMBAD = display only |
| — | P3 | Risk register misattributed to Revision 1 (it is `MASTER_PLAN.md`); "three of the last four PRs" (it is two); "60–95 examples" (eval minimum is 95) | Fixed here, in R2 and in the Decision Log rows |

Three-questions triage: every P1/P2 was judged real, introduced by this diff, and worth curing;
none was deferred.

### Round 2 — verdict NEEDS-ATTENTION: F1–F4, F6, F8, F9 and the P3s confirmed FIXED; F5, F7, F10 PARTIAL. 1 P1, 3 P2. All accepted.

| # | Sev | Finding | Change made |
|---|-----|---------|-------------|
| R1 | P1 | R2's Evaluation section said an arm outside *any* T-H3 band fails, while the pre-registration makes the non-hacking direction report-only; the two documents gave opposite verdicts for the same result | R2 now states the directional failure rule verbatim and names `training-pilot.md` as the normative text |
| R2 | P2 | T-H3 never said which B/C aggregate is compared with the band; a zero arm-A median collapsed the fallback band to [0, 0] | Comparison statistic = the trained arm's per-query median (rate for (c)); (a)/(b) computed on non-abstaining answers; empty outputs defined; pre-stated absolute fallback bands when A's median is 0 |
| R3 | P2 | "An abstaining answer is scored by component 3 only" let a fabricated citation inside an abstention escape the hard citation-validity floor; mixed refusal-plus-claims responses were unclassified | Citation validity now applies to every answer first; pure abstention defined by a fixed marker and no substantive sentences; mixed responses scored as non-abstaining |
| R4 | P2 | The architecture diagram still routed the query through SIMBAD alias expansion before the frozen retriever, contradicting the evidence contract and `author.py prepare` | Diagram redrawn: grounded path takes the verbatim query; alias expansion and live ADS appear only on the live-mode branch |

### Round 3 — no P1 remaining (review closes under the policy); R1, R3, R4 FIXED, R2 PARTIAL; 3 P2. All accepted and folded in.

| # | Sev | Finding | Change made |
|---|-----|---------|-------------|
| F1 | P2 | The `ABSTAIN:` marker was required by the reward but not by the gold schema, worksheet or fixtures, so equivalent refusals could be classified differently across arms | New pre-registered constant **D4** in R2: one answer-format contract (marker + reason code + ≤ 1 sentence) implemented in the harness before M2 and used by the gold seed, synthesis, serving, H3/T-H3 and the reward |
| F2 | P2 | Zero-word density was 0/0; an arm with no non-abstaining answers left the T-H3 median undefined | Density of an empty output = 0; an arm with < 10 non-abstaining eval answers fails T-H3; if arm A has < 10, the absolute fallback bands apply |
| F3 | P2 | Quotations and a "sources" line were excluded from claim extraction, so unsupported assertions could hide there; the abstention "reason" had no machine rule | Quoted material is a claim; only a bibcode-only `Sources:` line is excluded; pure abstention is machine-defined by D4 (marker, reason code, ≤ 1 sentence, ≤ 40 words) |

**Outcome.** Three targeted rounds, 2 + 1 + 0 P1s. Every finding across all rounds was accepted;
none was deferred. The pre-registration text that results is the one the next session builds
against. A native (light) Codex pass remains available before merge under the local policy.

---

### Sources

- [NASA-IMPACT astroquery-mcp](https://github.com/NASA-IMPACT/astroquery-mcp) ·
  [AstroAKD-Bench](https://nasa-impact.github.io/astroakd-bench/) ·
  [cbyrohl/mcp-server-ads](https://github.com/cbyrohl/mcp-server-ads)
- [Attributable post-rationalization in RAG citations (2609.23053)](https://arxiv.org/abs/2609.23053) ·
  [VERITAS (2510.13272)](https://arxiv.org/html/2510.13272) ·
  [Lessons from training grounded LLMs with verifiable rewards (2506.15522)](https://arxiv.org/abs/2506.15522v1) ·
  [RLVR for small search agents (2609.28765)](https://arxiv.org/pdf/2609.28765)
- [Pathfinder (2408.01556)](https://arxiv.org/pdf/2408.01556) · [Astro2Method (2610.30282)](https://arxiv.org/html/2610.30282) ·
  [AstroMLab 4 (2505.17592)](https://arxiv.org/html/2505.17592)
- [Qwen/Qwen3.8-27B](https://hf.co/Qwen/Qwen3.8-27B) · [google/gemma-4-12B-it](https://hf.co/google/gemma-4-12B-it) ·
  [ornith-ai/Ornith-1.0-9B](https://hf.co/ornith-ai/Ornith-1.0-9B) ·
  [AstroMLab/AstroSage-70B-20251009](https://hf.co/AstroMLab/AstroSage-70B-20251009)
