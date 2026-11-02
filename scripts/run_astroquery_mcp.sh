#!/usr/bin/env bash
# Launch NASA-IMPACT/astroquery-mcp (Apache-2.0) over stdio, pinned to one commit.
#
# Why a launcher and not `uvx --from git+...@<sha> astroquery-mcp`: at the pinned commit the
# upstream wheel ships only the `models/` package (flat-layout modules such as server.py are not
# packaged), so the console script dies with `ModuleNotFoundError: No module named 'server'`.
# The README's supported launch form is `python server.py` from a checkout, which is what this
# does -- with the checkout verified against the SHA and deps taken from upstream's uv.lock.
#
# Usage (from .mcp.json):  scripts/run_astroquery_mcp.sh <full-40-char-commit-sha>
# ADS: the server reads API_DEV_KEY, falling back to ADS_API_KEY (export it; .env is not read).
# stdout is the MCP channel -- everything here logs to stderr.
set -euo pipefail

SHA="${1:?usage: run_astroquery_mcp.sh <full-commit-sha>}"
if [[ ! "$SHA" =~ ^[0-9a-f]{40}$ ]]; then
  echo "run_astroquery_mcp.sh: need a full 40-char commit SHA, got '$SHA'" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIR="$ROOT/.vendor/astroquery-mcp"

if [[ ! -d "$DIR/.git" ]]; then
  mkdir -p "$ROOT/.vendor"
  git clone --quiet https://github.com/NASA-IMPACT/astroquery-mcp.git "$DIR" >&2
fi

if [[ "$(git -C "$DIR" rev-parse HEAD)" != "$SHA" ]]; then
  git -C "$DIR" cat-file -e "${SHA}^{commit}" 2>/dev/null || git -C "$DIR" fetch --quiet origin >&2
  git -C "$DIR" checkout --quiet --detach "$SHA" >&2
fi

exec uv run --quiet --frozen --no-dev --directory "$DIR" python server.py
