# Lab Note — 2026-11-02: Plan Revision 2

**Phase**: 1 · **Milestone**: none advanced (documents only — recorded as such per R2 process rule 1)

## What was done

- Status audit after the 07-21 → 10-23 pause; landscape re-verified 10-23
  ([plan review](../research/plan-review-2026-10.md)).
- [V1 Plan Revision 2](../V1_PLAN_R2.md) adopted: copilot v0 first, EXP-000 next, tools adopted
  via `astroquery-mcp`, three-arm training pilot, nothing from Phase A ships.
- EXP-004-A2 (teacher-drafted, human-verified gold; top-k context) and EXP-005 pre-registered.
- Three Codex plan-review rounds: 2 P1 + 8 P2 + 3 P3 → 1 P1 + 3 P2 → 0 P1 + 3 P2. All applied.

## What I learned

- The gold-seed harness froze all 100 abstracts per example while every training config caps
  sequences at 4,096 tokens. Nobody had put the two numbers next to each other. Decision D1 fixes
  the measurement; the configs change when it is made.
- An adversarial reviewer earns its keep on *pre-registrations*: three of its findings (gate
  bypass via "T-H1 or T-H2", non-inferiority by wide CI, reward only looking at cited sentences)
  would each have let a bad recipe through, and none was visible to the author.
- Teacher-drafted gold is fine for *prose* but not for *claim selection* — hence the human-authored
  challenge subset.

## Next session (M1, the first runnable artifact)

1. Extend the pool-sweep harness to score cuts {10, 20, 30, 50} and count delivered-context
   tokens; decide D1 and record it.
2. The `uv run` chat CLI over the frozen retriever with the citation guard (grounded mode).
3. Pin and wire `astroquery-mcp` for SIMBAD object cards; live ADS behind the live-mode flag.
