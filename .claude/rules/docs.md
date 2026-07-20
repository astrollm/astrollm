---
globs: ["docs/**"]
---

# Documentation Rules

- **Never modify** `docs/V1_FINAL_PLAN.md` or `docs/MASTER_PLAN.md` — authoritative planning documents, frozen as of commit `1739826` (see the 2026-07-21 Decision Log entry acknowledging the earlier base-model-reconcile edits). Supersessions are recorded **only** in the `RESEARCH_LOG.md` Decision Log; a Decision Log entry cannot authorize editing the frozen files themselves
- **Never modify** files in `docs/archive/` — these are superseded and kept for historical reference
- `docs/RESEARCH_LOG.md` is append-only — add new entries, never edit past experiments
- Use the `/research-log` skill for structured experiment entries
- Keep `CLAUDE.md` Current Status section up to date when phases change
- All docs use GitHub-flavored markdown
- Relative links between docs: `[V1 Plan](V1_FINAL_PLAN.md)` not absolute paths
- Dates in YYYY-MM-DD format (ISO 8601)
