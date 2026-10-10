#!/usr/bin/env bash
# Start ONE engine (no API front end) with the production kernel env of
# start_hgn.sh, on a test port, for protocol-level A/B tests (kvsnap_ab.py etc.).
# Keep the export block in sync with tools/serve_common.sh (serve_run).
#   PROBE_TAG=name PROBE_CONTEXT=65536 KVSNAP_DIR=data/kvsnap-ab \
#   MTP_FILE=models/xxx-mtp.hgn bash tools/run_prod_probe.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if pgrep -af '(^|/)(qwenox[^/[:space:]]*|flash_serve|serve_api\.py)([[:space:]]|$)'; then
  echo 'Stop the existing model process before starting a probe.' >&2
  exit 1
fi
export QWENOX_QSA_KV_BF16=1 QWENOX_QSA_WMMA=1 QWENOX_QSA_WMMA_BTV=1
export QWENOX_GEMM_WMMA=1 QWENOX_GDN_FUSED=1
export QWENOX_MOE_LT=1 QWENOX_MOE_LT_BF16=1 QWENOX_GR_BF16=1
export QWENOX_GDN_STREAM=1 QWENOX_GDN_WAVE=1 QWENOX_NOWARMUP=1
export QWENOX_PREFILL_CHUNK="${QWENOX_PREFILL_CHUNK:-16384}"
export QWENOX_INDEX_FUSED2=1 QWENOX_PP_MOE_OUT=1 QWENOX_INDEX_STREAM_SELECT=1
export QWENOX_KVSNAP=1 QWENOX_KVSNAP_DIR="${KVSNAP_DIR:-data/kvsnap-ab}"
mkdir -p logs
MODEL_DIR="${MODEL_DIR:-models}"
engine=("${PROBE_BINARY:-build/qwenox-engine}" "$MODEL_DIR/qwen38-flash-next-w4b.hgn"
        "$MODEL_DIR/qwen38-flash-next-w4b.overlay.hgn")
[[ -z "${MTP_FILE:-}" ]] || engine+=("$MTP_FILE")
engine+=(--serve --port 8732 --maxctx "${PROBE_CONTEXT:-65536}")
nohup bash tools/run_capped.sh 86 -- "${engine[@]}" \
  > "logs/${PROBE_TAG:-prod-probe}.log" 2>&1 < /dev/null &
echo "engine starting, log: logs/${PROBE_TAG:-prod-probe}.log (wait for 'serve: listening')"
