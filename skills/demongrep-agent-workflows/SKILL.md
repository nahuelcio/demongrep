---
name: demongrep-agent-workflows
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
