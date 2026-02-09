---
name: demongrep
description: Semantic code search for coding agents. Fast, local, no API calls.
---

# demongrep

## Agent Policy (CRITICAL)
1. **One search at a time** - Use `demongrep_hybrid_search` once per query
2. **Default params** - limit=4, per_file=1
3. **No chaining** - Don't call multiple tools for same query
4. **Be concise** - Return: `path:lines (score)` only

## Tools
- `demongrep_hybrid_search` - Primary (vector + BM25 fusion)
  - Default: limit=4, per_file=1
  - Use `compact=true` for minimal output (path:lines format)
  - Reranking: cached model, only use if needed (adds ~300ms after first load)
- `demongrep_semantic_search` - Fallback if hybrid empty
- `demongrep_index_status` - Diagnostics only

## Compact Mode
For minimal token usage, add `compact: true`:
```json
{"query": "auth handling", "limit": 4, "compact": true}
```
Returns: `[{"path": "src/auth.rs:10-25", "score": 0.89}]`

## Reranking
- Model is cached after first use (subsequent calls: ~300ms)
- Default: rerank_top=20, max=50
- Only enable if results need reordering: `{"rerank": true, "rerank_top": 20}`

## Daily CLI
```bash
demongrep search "auth handling"
demongrep search "retry logic" --filter-path src/
```

## Setup (if broken)
```bash
cd /project/path
demongrep setup && demongrep index
demongrep install-claude-code --project-path /project/path
# Restart agent after
```

## Troubleshooting
- "No database": Run `demongrep index`
- Wrong results: Check `which -a demongrep` for PATH issues
