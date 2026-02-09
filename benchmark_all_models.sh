#!/bin/bash
# Legacy wrapper: delegates to `demongrep bench` full profile.

set -euo pipefail

DEMONGREP="${DEMONGREP:-./target/release/demongrep}"
RESULTS_DIR="benchmarks/model_comparison_$(date +%Y%m%d_%H%M%S)"
RESULTS_MD="$RESULTS_DIR/benchmark_results.md"
RESULTS_JSON="$RESULTS_DIR/benchmark_results.json"

mkdir -p "$RESULTS_DIR"

echo "⚠️  [DEPRECATED] benchmark_all_models.sh now wraps 'demongrep bench'."
echo "   Recommended direct command:"
echo "   $DEMONGREP bench --profile full --path . --output benchmarks/benchmark_results.md"
echo ""

# Helpful default for local macOS + Homebrew setups.
if [[ -z "${ORT_DYLIB_PATH:-}" && -f "/opt/homebrew/opt/onnxruntime/lib/libonnxruntime.1.24.1.dylib" ]]; then
  export ORT_DYLIB_PATH="/opt/homebrew/opt/onnxruntime/lib/libonnxruntime.1.24.1.dylib"
fi

echo "Running full benchmark profile..."
echo "  Markdown: $RESULTS_MD"
echo "  JSON:     $RESULTS_JSON"
echo ""

has_opt() {
  local name="$1"
  shift
  for arg in "$@"; do
    if [[ "$arg" == "$name" || "$arg" == "$name="* ]]; then
      return 0
    fi
  done
  return 1
}

cmd=("$DEMONGREP" bench)
if ! has_opt "--profile" "$@" && ! has_opt "--models" "$@"; then
  cmd+=(--profile full)
fi
if ! has_opt "--path" "$@"; then
  cmd+=(--path .)
fi
if ! has_opt "--output" "$@"; then
  cmd+=(--output "$RESULTS_MD")
fi
if ! has_opt "--json" "$@"; then
  cmd+=(--json)
fi
cmd+=("$@")

"${cmd[@]}" > "$RESULTS_JSON"

echo ""
echo "✅ Benchmark completed."
echo "   - $RESULTS_MD"
echo "   - $RESULTS_JSON"
