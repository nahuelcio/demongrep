---
name: demongrep-agent-workflows
description: Semantic code search with demongrep. Use alongside grep/rg: grep for exact text, demongrep for concepts. Also use for setup and troubleshooting in Codex, Claude Code, and OpenCode.
---

# What demongrep does

Finds code by meaning.  
Use this skill when users ask questions like:
- "Where do we handle auth?"
- "Where is this validated?"
- "What code orchestrates retries?"

Use both tools:
- `rg`/`grep`: exact string or symbol match
- `demongrep`: concept match and semantic discovery

# Primary commands

```bash
# Semantic search
demongrep search "where do we validate user permissions"

# Agent-optimized output
demongrep search --agent "where do we validate user permissions"

# JSON output for pipelines/tools
demongrep search --json --quiet "authentication flow"
```

# Setup and integration workflow

1. Confirm installation:
```bash
demongrep --version
```
2. Warm model cache:
```bash
demongrep setup
```
3. Build/update index:
```bash
demongrep index
```
4. Validate environment:
```bash
demongrep doctor
```
5. Configure agent integration:
```bash
demongrep install-codex
demongrep install-claude-code
demongrep install-opencode
```
6. Tell the user to restart the agent application.

If a specific project must be pinned, use:
```bash
demongrep install-codex --project-path /absolute/path/to/project
```

Use `--dry-run` first when you need a safe preview of config changes.

# Architecture workflow for agents

1. Start semantic:
```bash
demongrep search "where do requests enter the server"
```
2. Narrow by area:
```bash
demongrep search "jwt validation" --filter-path src/auth
```
3. Increase precision when needed:
```bash
demongrep search "permission checks in handlers" --rerank
```
4. Use compact mode when only file locations are needed:
```bash
demongrep search "authentication middleware" --compact
```

# Output guidance

- Prioritize high-score results first.
- Prefer focused reads over full-file reads.
- Combine demongrep results with `rg` for exact follow-up.

# Failure handling

- If setup/model fails: run `demongrep doctor` and use `references/troubleshooting.md`.
- If integration fails: rerun install command with `--dry-run`, then without it.
- If results look stale: run `demongrep search "query" --sync`.

# Agent-specific references

- Codex: `references/codex.md`
- Claude Code: `references/claude-code.md`
- OpenCode: `references/opencode.md`
- Troubleshooting: `references/troubleshooting.md`
