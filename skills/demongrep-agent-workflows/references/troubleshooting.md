# Troubleshooting

## Common Fixes

1. Re-run install with explicit project path.
2. Validate JSON in config file.
3. Confirm `mcpServers.demongrep` exists.
4. Run `demongrep doctor` and address failures.
5. Restart the coding agent.

## Safe Preview

```bash
demongrep install-codex --project-path /absolute/path/to/project --dry-run
```
