#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
free -g
if pgrep -af '(^|/)(qwenox[^/[:space:]]*|flash_serve|serve_api\.py)([[:space:]]|$)'; then
  echo 'Stop the existing model process before starting a probe.' >&2
  exit 1
fi
export QWENOX_KVSNAP=0
export QWENOX_NGRAM_MIN="${QWENOX_NGRAM_MIN:-4}"
export QWENOX_NGRAM_MAX="${QWENOX_NGRAM_MAX:-16}"
mkdir -p logs
MODEL_DIR="${MODEL_DIR:-models}"
nohup bash tools/run_capped.sh 86 -- "${PROBE_BINARY:-build/qwenox-engine}" \
  "$MODEL_DIR/qwen38-flash-next-w4b.hgn" \
  "$MODEL_DIR/qwen38-flash-next-w4b.overlay.hgn" \
  --serve --port 8732 --maxctx "${PROBE_CONTEXT:-8192}" \
  > "logs/${PROBE_TAG:-ngram-probe}.log" 2>&1 < /dev/null &
