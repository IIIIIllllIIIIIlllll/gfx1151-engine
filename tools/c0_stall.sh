#!/usr/bin/env bash
# c0_stall.sh — D0-M5 并发卡顿基线（约 3 分钟）：
# PARALLEL=2 起引擎（端口 8732），A 持续 decode + B 提交 32K prompt，
# 报告 A 的最大 token 间隔与 B 的 TTFT。
#   BIN=build/qwenox_cc bash tools/c0_stall.sh
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source tools/probe_lib.sh
BIN="${BIN:-build/qwenox_cc}"
export PROBE_BINARY="$BIN"
probe_precheck || exit 1
chk="$(bash start_hgn.sh --check 2>&1)" || { echo "$chk"; exit 1; }
mapfile -t PENV < <(sed -n 's/^ENV //p' <<<"$chk")
CMDLINE="$(sed -n 's/^CMD //p' <<<"$chk")"
eval "BASE=($CMDLINE)"
trap 'probe_stop' EXIT
for e in $(compgen -e | grep '^QWENOX_'); do unset "$e"; done
for e in "${PENV[@]}"; do export "$e"; done
export QWENOX_KVSNAP=0 QWENOX_NGRAM_CHUNK=16 QWENOX_PARALLEL=2
cmd=("${BASE[@]}")
cmd[0]="$BIN"
for i in "${!cmd[@]}"; do
  [[ "${cmd[$i]}" == --port ]] && cmd[$((i + 1))]=8732
done
probe_start "c0-stall" "${cmd[@]}" || exit 1
python3 tools/c0_stall.py
rc=$?
exit $rc
