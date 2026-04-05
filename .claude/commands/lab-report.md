# /lab-report — Write a Weekly Lab Report

Create a weekly lab report in `docs/lab/` documenting what was built, learned, and observed.

## Usage
```
/lab-report [week_number]
```

Examples:
- `/lab-report 1` (first week of Phase 0)
- `/lab-report 5` (week 5 of Phase 1)

## Workflow

1. Determine the week number and date range
2. Ask the user about:
   - **What they built** this week (code, data, configs)
   - **What they learned** (from study materials, experiments, data exploration)
   - **Observations** (surprises, things noticed, informal notes)
   - **Blockers** (open questions, waiting on resources)
   - **Reading log** (what materials were covered)
3. Create `docs/lab/week-{XX}.md` using the template from `docs/lab/index.md`
4. Add a row to the Lab Report Index table in `docs/lab/index.md`
5. Add the new report to the `nav:` section in `mkdocs.yml` under "Lab Reports"
5. If any experiment results are mentioned, suggest creating an experiment report via the experiment template in `docs/research/experiments/index.md`

## Guidelines
- Honest over polished — lab reports are for the builder, not reviewers
- Link to specific commits, W&B runs, or files
- Document dead ends — they're data
- Keep it concise: 5-15 minutes to write, not an essay
- Include reading log entries for study materials covered that week
