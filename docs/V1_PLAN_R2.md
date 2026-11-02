# AstroLLM v1 Plan — Revision 2

*Date: 2026-11-02. Supersedes [V1_FINAL_PLAN.md](V1_FINAL_PLAN.md) (Revision 1, frozen at commit
`1739826`) as the execution plan for Phase 1. The long-term vision in
[MASTER_PLAN.md](MASTER_PLAN.md) is unchanged. Adopted via the 2026-11-02 Decision Log rows in the
[Research Log](RESEARCH_LOG.md); the reasoning is in the
[2026-10 plan review](research/plan-review-2026-10.md).*

Revision 1 is kept verbatim for the record. This document is the one place the *current* plan
lives; nothing here needs to be reconstructed from Decision Log rows. Once merged, a revision is
frozen too: changes go in a Revision 3 plus a Decision Log row (see `.claude/rules/docs.md`).

---

## Why a revision

Seven months after Revision 1 the project has a well-ablated retriever, a tested gold-seed
harness, a disciplined research log, zero gold examples, no model call anywhere in the pipeline,
and two long pauses. The Master Plan's risk register puts "motivation loss from invisible progress"
first for Phases 1–2 and mitigates it with the week-4 demo that Revision 1 lists among its protected
milestones; the demo was skipped on 2026-06-01 and both pauses followed. The current critical path (hand-author 150–250 examples, each over 100 abstracts, for a
fine-tune that by pre-registration never ships) makes the next visible artifact months away.

Three things also changed in the field since March (verified 2026-10-23, see the review):

- **Tool access is table stakes.** NASA-IMPACT publishes an MCP server exposing 141 astroquery
  functions (SIMBAD, ADS, MAST, Gaia, …) with a Claude Code plugin, plus its own data-search agent
  and benchmark. "AstroSage has no tools" is no longer a differentiator.
- **Citation faithfulness is an open, named problem.** A 2026 study finds roughly one RAG citation
  in seven is post-rationalized and that RL on answer correctness does not reduce it; faithfulness
  must be trained and measured as its own objective. That is exactly what the gold seed and
  verifier are for.
- **Verifiable-reward RL is a standard recipe.** Published recipes train grounding, citation
  sufficiency and refusal quality with GRPO from programmatic rewards and no gold answers. The
  project's verifier is such a reward.

---

## Product definition (unchanged core, sharpened edge)

**AstroLLM v1** is a retrieval-grounded astronomy copilot for graduate students and early-career
researchers: cited literature answers over NASA ADS, object resolution via SIMBAD, honest
abstention when evidence is thin.

**What it is competing on now.** An *open, small* model that cites faithfully and abstains
honestly, that runs on a laptop, and that ships with a public evaluation set measuring exactly
those behaviours. Tool access and RAG are necessary, not distinctive.

It is still NOT: a benchmark competitor to AstroSage, a model family, multimodal, or a
free-running autonomous agent.

---

## Scope lock (R2)

### In

- One model tier: **Core** — Qwen3.5-9B primary, Qwen3.5-4B for the training pilot; served
  no-think (Decision Log 2026-07-21).
- One retriever: the as-built hybrid BM25+dense RRF, stage-1 pool 100, over ADS abstracts; a
  **serving context of top-k** abstracts (k fixed by decision D1 below).
- Live tools **adopted, not built**: SIMBAD and ADS through the NASA-IMPACT `astroquery-mcp`
  server (pinned version), with `mcp-server-ads` for citation-graph queries if needed. The
  project's own retriever is also published as an MCP server so Claude Code / Desktop can use it.
  Live tools and the frozen retriever meet under one **evidence contract** (Architecture, below):
  in grounded mode the only citable evidence is the frozen top-k context; live ADS results are
  evidence only in a separately logged live mode that is never evaluated against the gates or used
  as training data; SIMBAD returns object cards for display, not evidence for claims.
- An **inference-time citation guard**: every bibcode in an answer must be in the retrieved
  context (hard rule from day one); claim-support checking once the verifier exists (M3).
- Evaluation: the EXP-004 eval seed (lit_qa / summarization / abstention) + the EXP-004-A1
  AstroMLab-1 retention gates + the astronomy error taxonomy from Revision 1.
- One training pilot with **three arms** (base / SFT / GRPO-with-verifier-reward) on the frozen
  exoplanet-atmosphere corpus (EXP-005). Phase-A weights never ship (EXP-004-A1 §4 stands).
- Audience-adaptive depth as a prompt feature, not a training target.

### Out (deferred, unchanged unless noted)

- Nano / Pro / Ultra tiers; NED / PDS / MAST / Gaia *custom* integrations (they come free with
  `astroquery-mcp`, but are not exercised or evaluated in v1); multimodal; custom MoE.
- **Multi-call agentic retrieval loops as the default serving path** — allowed as a flagged
  experiment only, because training contexts are assembled single-shot and the train/serve
  context distribution must match (adversarial review, point 3).
- Cross-encoder / ColBERT reranking, SPECTER2, query widening: post-beta retrieval cycle.
- Tool-use SFT, DPO, model merging, CPT: Phase 2.
- Hand-authored pedagogy data; Streamlit (R1's UI choice is dropped: CLI first, then the
  TanStack/Elysia web app that already exists as a scaffold).

---

## Architecture (as built → target)

```
GROUNDED MODE (default; the evaluated path — identical to `author.py prepare`)
query (verbatim) ─► hybrid RRF stage-1 (pool=100, frozen) ─► top-k serving context (D1)
                 ─► Qwen3.5-9B (no-think) + system prompt: cite bibcodes from context only; abstain if thin
                 ─► citation guard: bibcode ∈ top-k (hard) · claim-support verifier (M3+)
                 ─► answer + bibcode links · SIMBAD object card (display only) · logged, mode=grounded

LIVE MODE (opt-in flag; flagged experiment, separate log, never evaluated or trained on)
query ─► SIMBAD alias expansion + live ADS search (astroquery-mcp) ─► frozen top-k ∪ live results
      ─► same model + guard with the union as the allowlist ─► logged, mode=live
```

Serving: llama.cpp GGUF Q4_K_M on Apple Silicon or a hosted Qwen3.5-9B endpoint for v0 (same
weights; quantization noted in every eval row). vLLM when the web beta needs throughput.

### Evidence contract (grounded mode vs live mode)

| | **Grounded mode** (default; the only evaluated product path) | **Live mode** (opt-in flag; flagged experiment) |
|---|---|---|
| Retrieval input | the user query verbatim — no alias expansion in the retrieval path, so serving matches authoring (`author.py prepare`) exactly | query plus SIMBAD alias expansion and/or live ADS search |
| Citable evidence | the frozen retriever's top-k (D1) only | frozen top-k ∪ live ADS results, each tagged with its source |
| Citation-guard allowlist | the top-k bibcodes | the union above |
| SIMBAD | object card shown to the user; never cited as evidence | same |
| Transcript | the product log; eligible for eval review and (M4+) training prompts | a separate log; never evaluated against the gates, never used as training data |

---

## Pre-registered constants to lock before EXP-000

| ID | Decision | How it is decided | Why it blocks |
|----|----------|-------------------|---------------|
| **D1** | Serving context **k** (top-k of the fused pool-100 ranking) and the context token budget | Extend the pool-sweep harness to score fused hybrid recall at cuts {10, 20, 30, 50} on the frozen 29-query gold (today it scores only 10 and 50: 0.592 / 0.862 overall at pool 100) **and** to render each cut exactly as the model will see it (bibcode line + title + abstract) and count tokens with the Qwen3.5 tokenizer. Budget: **16,384 context tokens** per query; prompt overhead (system + question) reserved at 1,024 and answer allowance 2,048, so `max_seq_length` = 20,480. Overflow policy: k is a fixed count; if a query's top-k rendering exceeds the budget, drop whole abstracts from the bottom of the ranking until it fits, never truncate inside an abstract, and log the drop. Recall is scored on the *delivered* context after overflow handling. Selection rule: the smallest k ∈ {10, 20, 30, 50} whose delivered-context recall is within 0.05 of delivered R@50; if none qualifies, the largest k that fits the budget on ≥ 95% of queries, with the recall gap recorded. All of this is committed, and the chosen k reported, before any model output is seen. | The harness currently freezes **all 100** abstracts (~25–30K tokens) as context while every training config caps sequences at 4,096 tokens. Train and serve must see the same context. Gold records keep all 100 for provenance; the labeler/teacher/model see top-k. |
| **D2** | No-think template mechanism verified against the real tokenizer | Rider 2 of the 2026-07-21 decision | Must precede the first base-model eval, not just the fine-tune |
| **D3** | Teacher, judge, verifier identities | Teacher = Claude Opus 5.5 (`claude-opus-5-5`). Judge for H1/H3 = a different family (open-weight 27–31B via a provider, or human). Verifier = NLI cross-encoder + bibcode rule, calibrated per H5. Drafting model recorded in every gold record's provenance. | The pre-registered shared-bias control |
| **D4** | Answer-format contract shared by every arm, the gold seed, teacher synthesis, serving and the eval | One response grammar, implemented in the harness **before M2**: an abstaining answer begins with `ABSTAIN: <thin|contradictory|off_topic|absent>` followed by at most one sentence (≤ 40 words); any further sentence makes it a *mixed* response, scored as a non-abstaining answer. Non-abstaining answers carry inline bibcode citations and may end with one `Sources:` line containing bibcodes only. The schema's rule 1 is extended at implementation: abstention family ⇒ the answer starts with the marker; otherwise ⇒ no marker. The same parser classifies A/B/C outputs for H3/T-H3 and the RL reward. | EXP-005's reward and T-H3 classify answers by this grammar; the current gold fixtures and worksheet shape carry no marker, so without a shared contract equivalent refusals would be scored differently across arms (Codex round 3, F1). |

---

## Milestones

Every milestone ends in something that can be run with one `uv run` command and is written up in
one lab note. Calendar dates are re-committed at the end of each milestone (gates over dates).

### M1 — Copilot v0 (the skipped R1 week-4 demo)

Build: base Qwen3.5-9B behind a CLI chat; the retriever as a tool returning top-k with bibcodes;
`astroquery-mcp` for SIMBAD object cards and, behind a flag, live ADS search (per the evidence
contract: grounded mode is the default and the only evaluated path); the citation-enforcing system
prompt; the bibcode citation guard; conversation logging with error-taxonomy tags and the mode
recorded per transcript. No training, no agent loop, no web.

Exit: D1 and D2 locked; 25 logged conversations reviewed; `QUICKSTART.md` has a working
end-to-end command. Spend: $0–20.

### M2 — Eval seed + EXP-000 baselines

Author the **eval partition** (targets unchanged: ≥40 lit_qa / ≥30 summarization / ≥25
abstention) under EXP-004-A2: teacher drafts the cited answer and claims from the rendered top-k
context, the human verifies every support span and label. Then run EXP-000 for real: Qwen3.5-4B
and 9B (Gemma 4 E4B optional), closed-book + open-book AstroMLab-1 per EXP-004-A1, and the
eval-seed metrics (faithfulness, citation support, abstention) for the *base* model with RAG.

Exit: the EXP-000 table is filled; a written answer to "does base + RAG already clear the 80%
citation-support target?" Spend: $20–50 (API).

### M3 — Verifier, calibration partition, public eval set v0

Author the **calibration partition** (≥40 positives, 40–60 negatives covering all four failure
modes) the same way. Build the claim-support verifier in-repo; report H5 precision/recall on the
calibration partition; wire it into the copilot as a soft guard. Release **eval set v0** on
HuggingFace: queries, retrieved bibcode lists, claim-support labels, abstention labels — no
abstract text (publisher copyright).

Exit: H5 reported; verifier trust decision made (precision ≥ 0.85 → filter / reward; else
advisory only). Spend: $0–20.

### M4 — EXP-005 training pilot (Phase A, three arms)

Pre-registered in [training-pilot.md](research/training-pilot.md). Arms on Qwen3.5-4B QLoRA:
**base** (prompted), **SFT** on teacher-synthesized, verifier-filtered data (the EXP-004 recipe),
**GRPO** with the calibrated verifier as reward. Gates H1–H4 as amended. The GRPO arm runs only
if the verifier passed its trust gate. Nothing from Phase A ships.

Exit: EXP-005 results appended; each recipe independently validated or killed against the
inherited gates. Spend: $50–150 (GPU).

### M5 — Phase B (recipe frozen, corpus widened) — only if M4 validates a recipe

As in EXP-004: re-run the validated recipe on a general astro-ph slice with its own pre-registered
retention gate. This is the only run whose weights may ship.

### M6 — Public beta at astrollm.org

Web UI on the existing TanStack Start scaffold + Elysia API; ships Phase-B weights if they passed,
otherwise base + RAG + tools (an honest, documented outcome). Architecture page, known-limitations
page, feedback form, failed-query log. Target: 5 beta users.

---

## Model matrix (R2)

| Role | Model | Status |
|------|-------|--------|
| Primary serving / 9B arm | `Qwen/Qwen3.5-9B` | Confirmed 2026-10-23: no smaller Qwen3.6/3.8 dense exists |
| Training pilot | `Qwen/Qwen3.5-4B` | Unchanged |
| Track B | `google/gemma-4-E4B-it` | Optional; run only if M2 budget allows |
| Watch | `google/gemma-4-12B-it` (dense, 2026-05-23, QAT checkpoints) | Candidate for Phase 2; **not** an arm (budget) |
| Watch | `ornith-ai/Ornith-1.0-9B` (RL tune of Qwen3.5-9B) | Alternative 9B starting checkpoint at fine-tune time |
| Teacher | `claude-opus-5-5` | Synthesis + eval-seed drafting (A2) |
| Judge | different family from the teacher | Locked at D3 |

---

## Evaluation (unchanged gates, one addition)

- EXP-004 hypotheses H1–H5 and the EXP-004-A1 operationalization stand verbatim.
- Addition for EXP-005 (T-H3, normative text in [training-pilot.md](research/training-pilot.md)):
  reward-hacking tripwires reported for every arm — citation density, claims per answer,
  abstention rate on answerable queries — with bands fixed from the base arm before any trained arm
  is scored. A violation **in the hacking direction** (density above band, claim count below band,
  abstention on answerable queries above band) fails the arm regardless of headline metrics; the
  opposite direction is reported, not failed.
- AstroMLab-1 remains the external comparability number. Per EXP-004-A1 its **open-book
  topic-overlap slice is the Phase-A retention gate** and its **closed-book catastrophe thresholds
  (> 10 pp drop, or < 35% on 4-option MCQ) are a ship-block**; only the ordinary closed-book score
  is reported as expected drift rather than gated. The comprehensive general-astronomy retention
  gate belongs to Phase B.

---

## Budget

$400/month unchanged. Expected Phase-1 total through M6: $200–400, dominated by M4 GPU time and
M6 hosting. Rule from Revision 1 stands: no single run above 50% of a month's budget without a
documented hypothesis.

---

## Process rules (R2)

1. **Every work session ends with a runnable artifact** or a measured number. A session that only
   produces documents is recorded as such in the lab note and does not advance a milestone.
2. **Pre-registration is for runs that spend GPU money or gate a ship decision.** Harnesses,
   labeling, tooling and UI are build-measure-log.
3. **Amendments** keep the EXP-XXX-A# convention; originals are never edited.
4. **Plan revisions** are whole documents (`V1_PLAN_R<N>.md`), frozen on merge, superseded only by
   the next revision plus a Decision Log row.
5. **Lab notes** replace weekly lab reports: one short note per session in `docs/lab/`.
6. **One Codex round per PR**, run last, per the local review policy.

---

## Risks (R2)

| Risk | Mitigation |
|------|------------|
| R2 becomes another docs-only burst followed by a pause | M1 is defined so the next session can start building; the PR that adopts R2 names the first three M1 tasks |
| Base + RAG already meets the targets, hollowing out the training thread | Good outcome for the product; training continues under the learning mandate with the same gates |
| RL arm is unstable or reward-hacked | Conditional on verifier trust gate; tripwires; SFT arm is the fallback; 4B first |
| External MCP server churn (v0.1) | Pin the version; call astroquery directly inside the serving loop; MCP is for Claude clients |
| Train/serve context mismatch | D1 locks k before any eval; agentic multi-call stays a flagged experiment |
| Eval set too small to be a "benchmark" | Call it eval set v0; always report n and bootstrap CIs |

---

## Supersession table (Revision 1 → Revision 2)

| Revision 1 item | Disposition |
|-----------------|-------------|
| Weeks 1–2 retrieval foundation | Done (EXP-001–003); thread closed 2026-06-01 |
| Weeks 3–4 first copilot (Gemma 4 E4B, Streamlit) | Skipped in R1; **now M1** on Qwen3.5-9B, CLI |
| Weeks 5–6 SFT data, 5,000–8,000 examples, five families | Replaced by EXP-004 (three families, pilot scale) + A2 |
| Weeks 7–8 QLoRA SFT on 4B/9B | **M4 / EXP-005**, three arms incl. GRPO; Phase A never ships |
| Weeks 9–10 integration + reranking | Integration moves into M1/M6; reranking stays post-beta |
| Weeks 11–12 beta | **M6**; date re-committed after M4 |
| Three-stage retrieval (SPECTER2, ColBERTv2) | Post-beta cycle (unchanged decision) |
| Two live tools built in `tools-integration` | **Adopted** via `astroquery-mcp`; package retired as a build target |
| "No agent loop" | Still no autonomous loop; single-shot tool calls are the baseline |
| Week-8 model gate | Replaced by the EXP-004/005 gates |
