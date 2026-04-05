# Web Package

Chat interface and dashboard for AstroLLM at astrollm.org.

## Stack
- **Frontend**: TanStack Start + Shadcn/ui + Tailwind CSS
- **Backend**: Elysia (Bun runtime)
- **Styling**: Dark mode default (astronomers work at night)

## Features (Planned)

### Chat Interface
- Streaming LLM responses via SSE
- Inline citations linking to arXiv papers via ADS
- Tool call visualization (SIMBAD results, coordinate calculations)
- LaTeX rendering for equations (KaTeX)
- Code blocks with Astropy syntax highlighting
- Conversation history (local storage)

### Paper Explorer
- Semantic search over indexed papers
- Paper cards with title, authors, abstract, citation count
- Related papers via embedding similarity
- Direct links to arXiv and ADS

### Object Lookup
- Search by object name (M31, NGC 4151, Proxima Centauri)
- SIMBAD data card with coordinates, type, redshift, magnitudes
- Sky survey image cutout (from ESASky or SDSS)
- Available observations (HST, JWST, Kepler)

### Dashboard (Phase 4)
- Model comparison (base vs fine-tuned on benchmark)
- Training metrics from W&B
- RAG retrieval quality metrics
- User feedback aggregation

## Development

```bash
cd packages/web
bun install
bun dev  # http://localhost:3000
```

## Design Principles
1. **Information density**: Researchers want data, not whitespace
2. **Keyboard-first**: Shortcuts for common actions (Cmd+K for search)
3. **Citation-linked**: Every factual claim traces to a source
4. **Progressive disclosure**: Simple chat upfront, power features accessible
5. **Dark mode**: Non-negotiable for astronomers
