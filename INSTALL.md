# Installing demongrep

## Quick Start

The easiest way to install demongrep is using the provided installation script:

```bash
curl -sSL https://raw.githubusercontent.com/nahuelcio/demongrep/main/install.sh | bash
```

Then run the recommended first-use setup:

```bash
demongrep setup
```

## Installation Methods

### Method 1: Automated Installation (Recommended)

Download and run the installation script:

```bash
curl -sSL https://raw.githubusercontent.com/nahuelcio/demongrep/main/install.sh | bash
```

This script:
- Detects your OS and architecture
- Downloads the latest release
- Extracts and installs to `/usr/local/bin` or `~/.local/bin`
- Verifies the installation
- Provides helpful next steps

### Method 2: Custom Installation Directory

Install to a specific directory using the `INSTALL_DIR` environment variable:

```bash
INSTALL_DIR=~/.cargo/bin bash install.sh
```

Or:

```bash
INSTALL_DIR=~/bin bash install.sh
```

### Method 3: Specific Version

Install a specific version using the `VERSION` environment variable:

```bash
VERSION=1.0.0 bash install.sh
```

### Method 4: Manual Installation

1. Visit the [GitHub Releases](https://github.com/nahuelcio/demongrep/releases) page
2. Download the binary for your platform:
   - **Linux x86_64**: `demongrep-VERSION-x86_64-unknown-linux-gnu.tar.gz`
   - **macOS Intel**: `demongrep-VERSION-x86_64-apple-darwin.tar.gz`
   - **macOS Apple Silicon**: `demongrep-VERSION-aarch64-apple-darwin.tar.gz`

3. Extract and install:
```bash
tar -xzf demongrep-VERSION-TARGET.tar.gz
sudo mv demongrep /usr/local/bin/
chmod +x /usr/local/bin/demongrep
```

4. Verify installation:
```bash
demongrep --version
```

## Permissions

### Without sudo (Recommended)

If `/usr/local/bin` is not writable, the script will automatically fallback to `~/.local/bin`. This requires adding the directory to your PATH.

Add to your shell profile (`~/.bashrc`, `~/.zshrc`, `~/.profile`, etc.):

```bash
export PATH="$PATH:$HOME/.local/bin"
```

Then reload your shell:
```bash
source ~/.bashrc  # or ~/.zshrc
```

### With sudo

If you have administrative privileges and want to install system-wide:

```bash
sudo bash install.sh
```

## Troubleshooting

### "Command not found" after installation

The installation directory is not in your PATH. Add it:

```bash
export PATH="$PATH:$HOME/.local/bin"
```

Add this line to your shell profile to make it permanent.

### Permission denied

Try installing to a different directory:

```bash
INSTALL_DIR=~/.local/bin bash install.sh
```

Or use sudo:

```bash
sudo bash install.sh
```

### Download fails

The installation script has automatic retry logic. If downloads repeatedly fail:

1. Check your internet connection
2. Verify the release exists: `https://github.com/nahuelcio/demongrep/releases`
3. Try specifying a known version: `VERSION=0.1.0 bash install.sh`
4. Install manually from GitHub Releases

### Archive verification failed

Your download was corrupted. The script will automatically retry. If this persists:

1. Clear your package manager cache
2. Try from a different network
3. Download manually and extract

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `VERSION` | `latest` | Specific version to install (e.g., `1.0.0`) |
| `INSTALL_DIR` | Auto-detect | Custom installation directory |

## Script Options

```bash
bash install.sh [OPTIONS]
```

| Option | Description |
|--------|-------------|
| `--help` | Show help message |
| `--version` | Show installer version |

## Supported Platforms

| OS | Architecture | Status |
|----|--------------|--------|
| Linux | x86_64 | ✓ Supported |
| macOS | x86_64 (Intel) | ✓ Supported |
| macOS | aarch64 (Apple Silicon) | ✓ Supported |

## Coding Agent Integration

After installation, you can configure MCP integration for supported coding agents:

```bash
# Claude Code
demongrep install-claude-code --project-path /absolute/path/to/project

# Codex
demongrep install-codex --project-path /absolute/path/to/project

# OpenCode
demongrep install-opencode --project-path /absolute/path/to/project
```

Validate setup:

```bash
demongrep doctor
```

Troubleshooting guides:
- `docs/agents/claude-code.md`
- `docs/agents/codex.md`
- `docs/agents/opencode.md`
- `docs/agents/troubleshooting.md`

Install Codex skill pack (optional):

```bash
demongrep add-skills
```

## Uninstalling

Remove demongrep from your system:

```bash
# If installed in /usr/local/bin
sudo rm /usr/local/bin/demongrep

# If installed in ~/.local/bin
rm ~/.local/bin/demongrep

# Or check where it was installed
which demongrep
rm $(which demongrep)
```

## Building from Source

To build demongrep from source:

```bash
git clone https://github.com/nahuelcio/demongrep.git
cd demongrep
cargo build --release
./target/release/demongrep --version
```

## Getting Help

- **GitHub Issues**: [Report bugs](https://github.com/nahuelcio/demongrep/issues)
- **Documentation**: [View README](README.md)

## Script Safety

The installation script is designed to be safe for `curl | bash` usage:

- Uses `set -euo pipefail` for error handling
- Validates downloads with tar integrity checks
- Cleans up temporary files automatically
- Includes retry logic for failed downloads
- Provides clear error messages
- No destructive operations without explicit user action
- Source available for audit on GitHub

To review the script before running:

```bash
curl -sSL https://raw.githubusercontent.com/nahuelcio/demongrep/main/install.sh | less
```
