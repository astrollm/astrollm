# Lab Reports

Weekly lab reports documenting observations, learnings, and informal notes from building AstroLLM. Unlike the [Research Log](../RESEARCH_LOG.md) (structured experiment entries) or [Experiment Reports](../research/experiments/index.md) (detailed analysis), lab reports are an informal record of the journey.

## Purpose

- Track what you learned each week (even if it didn't produce a result)
- Document surprises, dead ends, and "aha" moments
- Create a record that feeds the technical blog posts and papers later
- Help future-you understand why decisions were made

## Lab Report Index

| Week | Date | Phase | Title | Key Takeaway |
|------|------|-------|-------|-------------|
| [01](week-01.md) | 2026-05-31 | 1 | Pilot retrieval prototype | Hybrid BM25+dense over 500 ADS abstracts: Recall@10 = 0.812, MRR = 0.776 (16 grounded queries, first-pass labels) |
| [note](2026-11-02-plan-revision-2.md) | 2026-11-02 | 1 | Plan Revision 2 | Docs-only session: R2 adopted, EXP-004-A2 + EXP-005 pre-registered, three Codex rounds; next session builds copilot v0 |

> **Gaps**: no reports for calendar weeks 02–07 (pause 2026-06-02 → 07-21) and none between
> 2026-07-21 and 2026-11-02 (second pause). Under [Plan R2](../V1_PLAN_R2.md) process rule 5,
> per-session lab notes replace weekly reports from 2026-11-02.

## Weekly Lab Report Template

Create a new file: `docs/lab/week-XX.md`

```markdown
# Week XX Lab Report

**Date**: YYYY-MM-DD to YYYY-MM-DD
**Phase**: 0 / 1 / 2 / 3 / 4+
**Hours spent**: X hrs

## What I Built

- Bullet list of concrete outputs (code, data, configs)

## What I Learned

- Key insights from study materials
- Surprising findings from experiments or data exploration
- Things that didn't work and why

## Observations

Informal notes — things noticed while working that might matter later.

## Blockers / Questions

- Open questions that need investigation
- Blockers waiting on external resources (API access, GPU availability, etc.)

## Next Week

- What I plan to work on
- Study materials to cover

## Reading Log

| Material | Chapters/Sections | Notes |
|----------|-------------------|-------|
| Raschka | Ch. X-Y | Key insight: ... |
| Paper: ... | Full | Relevant because: ... |
```

## Guidelines

- Write lab reports **at the end of each week**, even if it's just a few bullets
- Honest > polished — these are for you, not reviewers
- Link to specific commits, W&B runs, or files when referencing work
- If you hit a dead end, document it — dead ends are data
- Photos of whiteboard sketches, hand-drawn diagrams, etc. go in `docs/lab/assets/`
