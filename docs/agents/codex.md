# demongrep + Codex

## Quick Setup

```bash
# 1) Install demongrep
curl -sSL https://raw.githubusercontent.com/nahuelcio/demongrep/main/install.sh | bash

# 2) Set up local model cache (recommended)
demongrep setup

# 3) Index your project
cd /absolute/path/to/project
demongrep index

# 4) Install Codex MCP integration
demongrep install-codex --project-path /absolute/path/to/project

# 5) Validate
demongrep doctor
```

## Config File

Codex config path:

`~/.codex/config.json`

Expected entry:

```json
{
  "mcpServers": {
    "demongrep": {
      "command": "/absolute/path/to/demongrep",
      "args": ["mcp", "/absolute/path/to/project"]
    }
  }
}
```

## Safe Preview

Use dry-run before writing:

```bash
demongrep install-codex --project-path /absolute/path/to/project --dry-run
```
