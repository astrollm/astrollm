---
globs: ["packages/web/**"]
---

# Web Package Rules

- Use `bun` for all package management and script execution, never npm/yarn
- TypeScript strict mode — no `any` types unless truly unavoidable
- Use Biome for formatting and linting (configured in `packages/web/biome.json`)
- TanStack Start file-based routing: routes go in `src/routes/`
- Tailwind CSS 4 for styling — use utility classes, avoid custom CSS where possible
- Dark mode is the default — all components must support it
- Use `lucide-react` for icons (already in deps)
- Components go in `src/components/` with PascalCase naming
- Prefer server-side rendering where possible (TanStack Start SSR)
- Dev server runs on port 3000: `bun dev`
