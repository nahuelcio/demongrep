---
name: demongrep
description: Install, configure, validate, and troubleshoot demongrep for coding agents (Codex, Claude Code, OpenCode). Standalone workflow with no external references.
---

# demongrep Agent Workflows

Use this skill when the user asks to:
- install demongrep for an agent
- configure MCP integration
- fix a broken integration
- resolve version/path conflicts

## MCP vs CLI

- In coding agents (OpenCode/Codex/Claude), demongrep runs via MCP tools (`demongrep_*`) through `demongrep mcp`.
- In terminal usage, demongrep runs via direct CLI (`demongrep search ...`).
- CLI flags are not automatically available in MCP unless explicitly exposed by MCP tool schema.

## Agent execution policy (important)

When operating as a coding agent, keep tool usage strict and short:

1. Start with exactly one `demongrep_hybrid_search` call.
2. Use `limit` 8-12 and `per_file=1` for broad discovery.
3. Use `demongrep_semantic_search` only if hybrid returns empty/weak results.
4. Do not chain multiple search tools for the same query unless the user asks for deeper digging.
5. Do not narrate internal reasoning ("Thinking..."). Return concise findings with file paths and line ranges.
6. Only read full files after search when the user asks for full context or implementation details.

## Daily usage (CLI)

Use these commands when the user asks to find code by meaning:

```bash
# Basic semantic search
demongrep search "where do we validate auth tokens"

# Better precision on top candidates
demongrep search "permission checks in handlers" --rerank --rerank-top 50

# Restrict to an area
demongrep search "retry backoff logic" --filter-path src/

# Agent-optimized output
demongrep search "request entrypoint" --agent

# Machine-readable output
demongrep search "jwt middleware" --json --quiet
```

If user provides legacy shorthand (`demongrep "query" .`), treat it as:
`demongrep search "query" --path .`

## Standard setup flow

1. Verify binary health:
```bash
demongrep --version
which -a demongrep
```
2. Prepare project index:
```bash
cd /absolute/path/to/project
demongrep setup
demongrep index
```
3. Install agent integration:
```bash
demongrep install-codex --project-path /absolute/path/to/project
# or
# demongrep install-claude-code --project-path /absolute/path/to/project
# demongrep install-opencode --project-path /absolute/path/to/project
```
4. Validate:
```bash
demongrep doctor
```
5. Restart the agent app/process.

Use `--dry-run` first if you need a safe config preview.

## PATH and version hygiene

If multiple binaries appear in PATH, enforce one canonical binary first:

```bash
which -a demongrep
demongrep --version
```

If needed, prefer release install path first:

```bash
export PATH="$HOME/.local/bin:$PATH"
hash -r
which -a demongrep
demongrep --version
```

## Troubleshooting playbook

1. Integration missing in agent:
- rerun install command with explicit `--project-path`
- run once with `--dry-run`, then apply real run

2. Agent still behaves like old version:
- re-check `which -a demongrep`
- fix PATH order
- restart agent process

3. OpenCode config schema mismatch:
- support either legacy `mcpServers.demongrep` or new `mcp.demongrep`
- rerun `demongrep install-opencode --project-path /absolute/path/to/project`

4. Search relevance issues:
- rebuild index with `demongrep index`
- prefer hybrid MCP search tool
- enable MCP rerank parameters when available

## Expected completion criteria

- `demongrep doctor` has no blocking failures
- agent config includes a valid `demongrep` MCP entry
- active binary/version matches intended installation path
