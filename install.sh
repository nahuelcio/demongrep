#!/bin/bash
set -euo pipefail

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
REPO_OWNER="nahuelcio"
REPO_NAME="demongrep"
RELEASE_BASE_URL="https://github.com/${REPO_OWNER}/${REPO_NAME}/releases/download"

# Script information
SCRIPT_VERSION="1.0.0"
TEMP_DIR=""

# Cleanup function
cleanup() {
    if [ -n "$TEMP_DIR" ] && [ -d "$TEMP_DIR" ]; then
        rm -rf "$TEMP_DIR"
    fi
}
trap cleanup EXIT

# Help message
show_help() {
    cat << EOF
${BLUE}demongrep installer${NC}

USAGE:
    install.sh [OPTIONS]

OPTIONS:
    --help              Show this help message
    --version           Show installer version
    
ENVIRONMENT VARIABLES:
    VERSION             Specific version to install (default: latest)
    INSTALL_DIR         Custom installation directory (default: /usr/local/bin or ~/.local/bin)

EXAMPLES:
    # Install latest version
    curl -sSL https://raw.githubusercontent.com/${REPO_OWNER}/${REPO_NAME}/main/install.sh | bash

    # Install specific version
    VERSION=1.0.0 bash install.sh

    # Install to custom directory
    INSTALL_DIR=~/.cargo/bin bash install.sh

EOF
}

# Logging functions
log_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

log_success() {
    echo -e "${GREEN}✓${NC} $1"
}

log_error() {
    echo -e "${RED}✗${NC} $1" >&2
}

log_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

# Parse arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --help)
                show_help
                exit 0
                ;;
            --version)
                echo "Installer version: $SCRIPT_VERSION"
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
}

# Detect OS
detect_os() {
    case "$(uname -s)" in
        Linux*)
            echo "linux"
            ;;
        Darwin*)
            echo "macos"
            ;;
        MINGW*|MSYS*|CYGWIN*)
            echo "windows"
            ;;
        *)
            log_error "Unsupported operating system: $(uname -s)"
            exit 1
            ;;
    esac
}

# Detect architecture
detect_arch() {
    local machine
    machine=$(uname -m)
    case "$machine" in
        x86_64|amd64)
            echo "x86_64"
            ;;
        aarch64|arm64)
            echo "aarch64"
            ;;
        *)
            log_error "Unsupported architecture: $machine"
            exit 1
            ;;
    esac
}

# Map OS and architecture to target triple
get_target_triple() {
    local os=$1
    local arch=$2
    
    case "${os}-${arch}" in
        linux-x86_64)
            echo "x86_64-unknown-linux-gnu"
            ;;
        macos-x86_64)
            echo "x86_64-apple-darwin"
            ;;
        macos-aarch64)
            echo "aarch64-apple-darwin"
            ;;
        *)
            log_error "Unsupported OS and architecture combination: ${os}-${arch}"
            exit 1
            ;;
    esac
}

# Get latest version from GitHub API
get_latest_version() {
    if command -v curl &> /dev/null; then
        local response
        response=$(curl -sSL "https://api.github.com/repos/${REPO_OWNER}/${REPO_NAME}/releases/latest" 2>/dev/null || true)
        
        # Check if response contains a valid tag_name
        if echo "$response" | grep -q '"tag_name"'; then
            # Extract version without 'v' prefix
            echo "$response" | grep -o '"tag_name":"[^"]*' | cut -d'"' -f4 | sed 's/^v//'
        else
            log_warning "Could not fetch latest version from GitHub API, using 'latest' tag"
            echo "latest"
        fi
    else
        echo "latest"
    fi
}

# Validate URL (check if it returns 200)
validate_url() {
    local url=$1
    
    if command -v curl &> /dev/null; then
        if curl -sSL -I "$url" 2>/dev/null | head -1 | grep -q "200\|302"; then
            return 0
        else
            return 1
        fi
    fi
    
    # If curl is not available, assume URL is valid
    return 0
}

# Download file with retry logic
download_file() {
    local url=$1
    local output=$2
    local max_retries=3
    local retry_count=0
    
    while [ $retry_count -lt $max_retries ]; do
        log_info "Downloading from $url..."
        
        if curl -sSL -f "$url" -o "$output"; then
            log_success "Downloaded successfully"
            return 0
        else
            retry_count=$((retry_count + 1))
            if [ $retry_count -lt $max_retries ]; then
                log_warning "Download failed, retrying... (attempt $((retry_count + 1))/$max_retries)"
                sleep 2
            fi
        fi
    done
    
    log_error "Failed to download after $max_retries attempts"
    return 1
}

# Verify download integrity (using tar integrity check)
verify_archive() {
    local archive=$1
    
    log_info "Verifying archive integrity..."
    
    if tar -tzf "$archive" > /dev/null 2>&1; then
        log_success "Archive verification passed"
        return 0
    else
        log_error "Archive verification failed - corrupted download"
        return 1
    fi
}

# Extract archive
extract_archive() {
    local archive=$1
    local extract_dir=$2
    
    log_info "Extracting archive..."
    
    if tar -xzf "$archive" -C "$extract_dir"; then
        log_success "Extraction successful"
        return 0
    else
        log_error "Extraction failed"
        return 1
    fi
}

# Find extracted binary
find_binary() {
    local extract_dir=$1
    
    # Look for demongrep binary in common locations
    if [ -f "$extract_dir/demongrep" ]; then
        echo "$extract_dir/demongrep"
    elif [ -f "$extract_dir/bin/demongrep" ]; then
        echo "$extract_dir/bin/demongrep"
    else
        # Try to find it anywhere in the extracted directory
        find "$extract_dir" -maxdepth 2 -type f -name "demongrep" -print -quit
    fi
}

# Determine installation directory
determine_install_dir() {
    # If INSTALL_DIR is set, use it
    if [ -n "${INSTALL_DIR:-}" ]; then
        echo "$INSTALL_DIR"
        return 0
    fi
    
    # Try /usr/local/bin first (usually requires sudo)
    if [ -w "/usr/local/bin" ]; then
        echo "/usr/local/bin"
        return 0
    fi
    
    # Try ~/.local/bin as fallback
    local local_bin="$HOME/.local/bin"
    if [ ! -d "$local_bin" ]; then
        mkdir -p "$local_bin"
    fi
    
    if [ -w "$local_bin" ]; then
        echo "$local_bin"
        return 0
    fi
    
    # No writable directory found
    return 1
}

# Check if we need sudo and offer help
handle_permissions() {
    local install_dir=$1
    
    if [ ! -w "$install_dir" ]; then
        if [ -d "$install_dir" ] || [ -d "$(dirname "$install_dir")" ]; then
            log_warning "Installation directory requires elevated permissions"
            log_info "You can either:"
            log_info "  1. Run with sudo: sudo bash install.sh"
            log_info "  2. Set INSTALL_DIR to a writable directory:"
            log_info "     INSTALL_DIR=~/.local/bin bash install.sh"
            return 1
        fi
    fi
    
    return 0
}

# Install binary
install_binary() {
    local binary=$1
    local install_dir=$2
    
    # Ensure install directory exists
    if [ ! -d "$install_dir" ]; then
        mkdir -p "$install_dir"
    fi
    
    # Check permissions
    if ! handle_permissions "$install_dir"; then
        return 1
    fi
    
    # Make binary executable
    chmod +x "$binary"
    
    # Copy to installation directory
    log_info "Installing to $install_dir..."
    
    if cp "$binary" "$install_dir/demongrep"; then
        log_success "Installed to $install_dir/demongrep"
        
        # Verify installation
        if [ -x "$install_dir/demongrep" ]; then
            return 0
        else
            log_error "Installed file is not executable"
            return 1
        fi
    else
        log_error "Failed to copy binary to $install_dir"
        return 1
    fi
}

# Get installed version
get_installed_version() {
    local binary=$1
    
    if "$binary" --version 2>/dev/null | head -1; then
        return 0
    elif "$binary" -v 2>/dev/null | head -1; then
        return 0
    else
        echo "unknown"
        return 0
    fi
}

# Main installation flow
main() {
    parse_args "$@"
    
    log_info "Starting demongrep installation..."
    
    # Detect system
    local os
    local arch
    local target
    os=$(detect_os)
    arch=$(detect_arch)
    target=$(get_target_triple "$os" "$arch")
    
    log_info "Detected: ${os} (${arch})"
    log_info "Target triple: ${target}"
    
    # Determine version
    local version="${VERSION:-}"
    if [ -z "$version" ]; then
        log_info "Fetching latest version..."
        version=$(get_latest_version)
    fi
    
    log_info "Version to install: v${version}"
    
    # Construct download URL
    local download_url="${RELEASE_BASE_URL}/v${version}/demongrep-${version}-${target}.tar.gz"
    log_info "Download URL: $download_url"
    
    # Create temporary directory
    TEMP_DIR=$(mktemp -d)
    trap cleanup EXIT
    
    # Download
    local archive="$TEMP_DIR/demongrep-${version}.tar.gz"
    if ! download_file "$download_url" "$archive"; then
        log_error "Installation failed: could not download demongrep"
        exit 1
    fi
    
    # Verify archive
    if ! verify_archive "$archive"; then
        exit 1
    fi
    
    # Extract
    local extract_dir="$TEMP_DIR/extract"
    mkdir -p "$extract_dir"
    if ! extract_archive "$archive" "$extract_dir"; then
        exit 1
    fi
    
    # Find binary
    local binary
    binary=$(find_binary "$extract_dir")
    if [ -z "$binary" ] || [ ! -f "$binary" ]; then
        log_error "Could not find demongrep binary in extracted archive"
        exit 1
    fi
    
    log_info "Found binary: $binary"
    
    # Determine installation directory
    local install_dir=$(determine_install_dir)
    if [ -z "$install_dir" ]; then
        log_error "Could not determine writable installation directory"
        log_info "Please set INSTALL_DIR environment variable and try again"
        exit 1
    fi
    
    # Install
    if ! install_binary "$binary" "$install_dir"; then
        exit 1
    fi
    
    # Verify installation and get version
    local installed_path="$install_dir/demongrep"
    local installed_version=$(get_installed_version "$installed_path")
    
    echo ""
    log_success "Installation completed successfully!"
    echo ""
    echo "  ${BLUE}Binary:${NC} $installed_path"
    echo "  ${BLUE}Version:${NC} $installed_version"
    echo ""
    
    # Check if install_dir is in PATH
    if [[ ":$PATH:" == *":$install_dir:"* ]]; then
        log_info "demongrep is available in your PATH"
        echo ""
        log_info "You can now run: ${GREEN}demongrep${NC}"
    else
        log_warning "Installation directory is not in your PATH"
        echo ""
        log_info "To use demongrep, either:"
        log_info "  1. Add $install_dir to your PATH"
        log_info "  2. Run $installed_path directly"
        echo ""
        log_info "To add to PATH, add this line to your shell profile (~/.bashrc, ~/.zshrc, etc.):"
        log_info "    ${GREEN}export PATH=\"\$PATH:$install_dir\"${NC}"
    fi
    
    echo ""
}

main "$@"
