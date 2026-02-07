#!/bin/bash
set -euo pipefail

REPO_OWNER="nahuelcio"
REPO_NAME="demongrep"
SKILL_NAME="demongrep-agent-workflows"
REF="${REF:-main}"
DEST_DIR="${CODEX_SKILLS_DIR:-$HOME/.codex/skills}"
TEMP_DIR=""

cleanup() {
  if [ -n "$TEMP_DIR" ] && [ -d "$TEMP_DIR" ]; then
    rm -rf "$TEMP_DIR"
  fi
}
trap cleanup EXIT

log() {
  printf "%s\n" "$1"
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Error: missing required command '$1'" >&2
    exit 1
  fi
}

main() {
  require_cmd curl
  require_cmd tar

  TEMP_DIR=$(mktemp -d)
  local tarball="$TEMP_DIR/repo.tar.gz"
  local url="https://github.com/${REPO_OWNER}/${REPO_NAME}/archive/refs/tags/${REF}.tar.gz"

  # If REF is not a tag, fall back to branch/archive ref style.
  if ! curl -fsSL "$url" -o "$tarball"; then
    url="https://github.com/${REPO_OWNER}/${REPO_NAME}/archive/${REF}.tar.gz"
    curl -fsSL "$url" -o "$tarball"
  fi

  tar -xzf "$tarball" -C "$TEMP_DIR"

  local repo_dir
  repo_dir=$(find "$TEMP_DIR" -maxdepth 1 -type d -name "${REPO_NAME}-*" | head -n 1)
  if [ -z "$repo_dir" ]; then
    echo "Error: could not find extracted repository directory" >&2
    exit 1
  fi

  local skill_src="$repo_dir/skills/$SKILL_NAME"
  if [ ! -d "$skill_src" ]; then
    echo "Error: skill not found at $skill_src" >&2
    exit 1
  fi

  mkdir -p "$DEST_DIR"
  rm -rf "$DEST_DIR/$SKILL_NAME"
  cp -R "$skill_src" "$DEST_DIR/"

  log "Installed skill: $SKILL_NAME"
  log "Destination: $DEST_DIR/$SKILL_NAME"
  log "Restart Codex to pick up new skills."
}

main "$@"
