#!/usr/bin/env bash
# ht 重编码（beam 384 x 3 sweeps + guided LNS）-> 重新组装 -> 离线验证。
# 旧的 beam-128 parts 与旧的 out/ 产物一律 mv 保留为 .bak，不删除。
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export PYTHONPATH=/home/mark/Workspace/pylib
export HTENC_BEAM=${HTENC_BEAM:-384}
export HTENC_SWEEPS=${HTENC_SWEEPS:-3}
export HTENC_GUIDED=${HTENC_GUIDED:-4}
PY=python3
W=/home/mark/Models/v2repro

step() { echo; echo "===== $(date +%H:%M:%S) $* ====="; }

step "ht re-encode beam=$HTENC_BEAM sweeps=$HTENC_SWEEPS guided=$HTENC_GUIDED"
if [ -d "$W/parts/ht" ]; then
  mv "$W/parts/ht" "$W/parts/ht-b128-bak"
fi
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
