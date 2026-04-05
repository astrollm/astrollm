---
globs: ["docs/**"]
---

# Documentation Rules

- **Never modify** `docs/V1_FINAL_PLAN.md` or `docs/MASTER_PLAN.md` — these are authoritative planning documents frozen at their creation date
- **Never modify** files in `docs/archive/` — these are superseded and kept for historical reference
- `docs/RESEARCH_LOG.md` is append-only — add new entries, never edit past experiments
- Use the `/research-log` skill for structured experiment entries
- Keep `CLAUDE.md` Current Status section up to date when phases change
- All docs use GitHub-flavored markdown
- Relative links between docs: `[V1 Plan](V1_FINAL_PLAN.md)` not absolute paths
- Dates in YYYY-MM-DD format (ISO 8601)
