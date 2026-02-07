---
name: demongrep-agent-workflows
description: Setup, verify, and troubleshoot demongrep MCP integration for Codex, Claude Code, and OpenCode. Use when users ask how to install demongrep for coding agents, configure MCP, validate integration, or recover from integration failures.
---

# demongrep Agent Workflows

Use this skill when a user asks for demongrep integration with Codex, Claude Code, or OpenCode.

## Workflow

1. Confirm demongrep is installed.
2. Confirm the project path that should be indexed.
3. Run `demongrep setup` to warm model cache.
4. Run indexing if needed.
5. Run the matching install command.
6. Run `demongrep doctor` and resolve failures.
7. Ask the user to restart the coding agent.

## Agent Selection

- For Codex requests, read `references/codex.md`.
- For Claude Code requests, read `references/claude-code.md`.
- For OpenCode requests, read `references/opencode.md`.
- For any setup errors, read `references/troubleshooting.md`.

## Canonical Commands

```bash
demongrep install-claude-code --project-path /absolute/path/to/project
demongrep install-codex --project-path /absolute/path/to/project
demongrep install-opencode --project-path /absolute/path/to/project
```

Use `--dry-run` first when you need to preview config changes safely.
