#!/usr/bin/env bash
# ht 重编码的剩余部分（lm_head 之后降档）：断点续跑 -> 重组装 -> 离线验证。
# 用法: HTENC_BEAM=192 bash tools/v2repro/run_ht_rest.sh
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export PYTHONPATH=/home/mark/Workspace/pylib
export HTENC_BEAM=${HTENC_BEAM:-192}
export HTENC_SWEEPS=${HTENC_SWEEPS:-3}
export HTENC_GUIDED=${HTENC_GUIDED:-4}
PY=python3
W=/home/mark/Models/v2repro

step() { echo; echo "===== $(date +%H:%M:%S) $* ====="; }

step "ht resume beam=$HTENC_BEAM sweeps=$HTENC_SWEEPS guided=$HTENC_GUIDED"
$PY tools/v2repro/v2_encode.py ht || exit 1

step "assemble"
if [ -f "$W/out/qwen38-flash-next-v2.hgn" ]; then
  mv "$W/out/qwen38-flash-next-v2.hgn" "$W/out/qwen38-flash-next-v2.hgn.bak2"
fi
$PY tools/v2repro/v2_assemble.py || exit 1

step "offline verify"
$PY tools/v2repro/v2_verify.py
RC=$?
step "done rc=$RC"
exit $RC
