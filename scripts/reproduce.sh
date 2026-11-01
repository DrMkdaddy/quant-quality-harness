#!/usr/bin/env bash
# Reproduce the committed sweep from scratch.
set -euo pipefail

cd "$(dirname "$0")/.."

CONFIG="${CONFIG:-configs/qwen2.5-3b.json}"
LLAMA_BIN="${QUANTLAB_LLAMA_BIN:-$HOME/llama.cpp/llama-b11050}"

python3 -m quantlab fetch --config "$CONFIG"
python3 -m quantlab run --config "$CONFIG" --llama-bin "$LLAMA_BIN" --out-dir results
