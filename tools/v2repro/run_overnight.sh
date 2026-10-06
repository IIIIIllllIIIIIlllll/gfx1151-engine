#!/usr/bin/env bash
# 通宵复刻管线：BF16 -> v2 hgn（编码 -> 组装 -> 离线验证）。
# 用法: bash tools/v2repro/run_overnight.sh
# 日志: logs/v2repro_overnight.log
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export PYTHONPATH=/home/mark/Workspace/pylib
PY=python3
ENC=tools/v2repro/v2_encode.py
ASM=tools/v2repro/v2_assemble.py
VER=tools/v2repro/v2_verify.py

step() { echo; echo "===== $(date +%H:%M:%S) $* ====="; }

step "symbols"
$PY $ENC symbols || exit 1
step "q6g64"
$PY $ENC q6g64 || exit 1
step "ht (C++ htenc, 全核)"
$PY $ENC ht || exit 1
step "i4r (numpy x12)"
OMP_NUM_THREADS=2 $PY $ENC i4r --workers 12 || exit 1
step "assemble --dry-run"
$PY $ASM --dry-run || exit 1
step "assemble"
$PY $ASM || exit 1
step "offline verify"
$PY $VER
RC=$?
step "done rc=$RC"
exit $RC
