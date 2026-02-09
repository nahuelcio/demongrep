# Agent Integration Troubleshooting

## 1. Check local installation

```bash
demongrep --version
demongrep doctor
```

If `doctor` reports failures, fix them first.

## 2. Verify your MCP config entry

Codex and Claude Code need:

```json
"mcpServers": {
  "demongrep": {
    "command": "/absolute/path/to/demongrep",
    "args": ["mcp", "/absolute/path/to/project"]
  }
}
```

OpenCode needs:

```json
"mcp": {
  "demongrep": {
    "type": "local",
    "enabled": true,
    "command": ["/absolute/path/to/demongrep", "mcp", "/absolute/path/to/project"]
  }
}
```

## 3. Re-run install with explicit project path

```bash
demongrep install-claude-code --project-path /absolute/path/to/project
demongrep install-codex --project-path /absolute/path/to/project
demongrep install-opencode --project-path /absolute/path/to/project
```

## 4. Use dry-run before writing

```bash
demongrep install-codex --project-path /absolute/path/to/project --dry-run
```

## 5. Restart the agent

After config changes, fully restart the coding agent process.
