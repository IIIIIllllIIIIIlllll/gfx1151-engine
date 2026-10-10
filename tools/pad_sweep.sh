#!/usr/bin/env bash
# 扫 QWENOX_PAD_BATCH 对 (2560,6144) oproj gemm 的影响（32k tokens + 128k 布局）
set -u
cd "$(dirname "$0")/.."
for PAD in "$@"; do
  LOG="logs/pad_${PAD}.log"
  env QWENOX_PAD_BATCH=$PAD QWENOX_KPROF=1 \
      QWENOX_QSA_KV_BF16=1 QWENOX_QSA_WMMA=1 QWENOX_QSA_WMMA_BTV=1 \
      QWENOX_MOE_LT=1 QWENOX_MOE_LT_BF16=1 QWENOX_GR_BF16=1 \
      QWENOX_GDN_STREAM=1 QWENOX_GDN_WAVE=1 \
      QWENOX_PREFILL_CHUNK=16384 QWENOX_GEMM_WMMA=1 QWENOX_GDN_FUSED=1 \
      QWENOX_INDEX_FUSED2=1 QWENOX_PP_MOE_OUT=1 QWENOX_INDEX_STREAM_SELECT=1 \
      QWENOX_KVSNAP=1 QWENOX_KVSNAP_MAX_GB=20 QWENOX_PROF=1 \
      build/qwenox-engine-win models/qwen38-flash-next-v2.hgn models/qwen38-flash-next-ngram.hgn \
      --tokens-file data/qsa-oracle/32768.tokens --gen 1 --maxctx 139264 >"$LOG" 2>&1
  rc=$?
  # 取 base=16384（第 3 个 chunk，稳态）行的 oproj 两段
  LINE=$(grep "base=16384" "$LOG" | head -1)
  GOP=$(echo "$LINE" | grep -o "\[gdn:oproj [0-9.]*" | grep -o "[0-9.]*")
  QOP=$(echo "$LINE" | grep -o "\[qsa:oproj [0-9.]*" | grep -o "[0-9.]*")
  GPR=$(echo "$LINE" | grep -o "\[gdn:proj [0-9.]*" | grep -o "[0-9.]*")
  echo "pad=$PAD rc=$rc gdn:oproj=$GOP qsa:oproj=$QOP gdn:proj=$GPR"
done
