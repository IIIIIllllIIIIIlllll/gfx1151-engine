#!/usr/bin/env bash
# draft_desync_verify.sh — 一键验证 draft 链去同步（QWENOX_DRAFT_DESYNC，默认开）：
#   γ 步 draft 的 argmax 走 D2D 接力，每轮最后一次 D2H 收齐；关闭：QWENOX_DRAFT_DESYNC=0
# 检查（greedy，SPEC=256 GAMMA=3）：
#   1. 8K MTP / 8K chain / 32K MTP：BIN vs BASE 的 ids: 完全一致
#   2. opt-out（QWENOX_DRAFT_DESYNC=0）与 BASE ids 一致
#   3. 打印各组 tok/s 供肉眼对比
# 用法: bash tools/draft_desync_verify.sh    （约 4 分钟，结尾输出 PASS / FAIL）
#   BIN=build/qwenox-engine（默认）  BASE=build/qwenox.base（默认）
# 日志: logs/dsv_*.log，汇总 logs/draft_desync_verify.out
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
BIN="${BIN:-build/qwenox-engine}"
BASE="${BASE:-build/qwenox.base}"
OUT=logs/draft_desync_verify.out
mkdir -p logs
exec > >(tee "$OUT") 2>&1
fail=0
bad() { echo "  FAIL: $*"; fail=1; }
ok() { echo "  OK: $*"; }
[[ -x "$BIN" && -x "$BASE" ]] || { echo "缺二进制（先 bash build.sh）"; echo FAIL; exit 1; }
echo "== draft_desync_verify  $(date '+%F %T')  BIN=$BIN BASE=$BASE"

run() {  # run LABEL BIN PROMPT [K=V...]
  local L=$1 B=$2 P=$3; shift 3
  BIN=$B SPEC=256 GAMMA=3 bash tools/pp_prod.sh "$P" "$L" "$@" >"logs/$L.stdout" 2>&1
  local rc=$?
  if (( rc )) || ! grep -q '^ids:' "logs/$L.log"; then
    echo "  $L: 运行失败 rc=$rc"; grep -h 'rror\|abort\|exception' "logs/$L.log" | head -3
    return 1
  fi
}
ids() { grep '^ids:' "logs/$1.log"; }
specline() { grep -h 'spec: \| chain: ' "logs/$1.log" | tail -1 | cut -c1-90; }

echo "-- 1. ids 一致：8K MTP / 8K chain / 32K MTP"
for t in "8k mtp" "8k chain" "32k mtp"; do
  set -- $t; P=$1; M=$2
  A=(); [[ $M == chain ]] && A=(QWENOX_SPEC_CHAIN=1)
  if run dsv_${P}_${M}_base "$BASE" "$P" "${A[@]}" && run dsv_${P}_${M}_new "$BIN" "$P" "${A[@]}"; then
    echo "  base: $(specline dsv_${P}_${M}_base)"
    echo "  new : $(specline dsv_${P}_${M}_new)"
    [[ "$(ids dsv_${P}_${M}_base)" == "$(ids dsv_${P}_${M}_new)" ]] \
      && ok "$P $M ids 一致" || bad "$P $M ids 不一致"
  else
    bad "$P $M 运行失败"
  fi
done

echo "-- 2. opt-out 与 BASE 一致"
if run dsv_off "$BIN" 8k QWENOX_DRAFT_DESYNC=0; then
  [[ "$(ids dsv_off)" == "$(ids dsv_8k_mtp_base)" ]] \
    && ok "QWENOX_DRAFT_DESYNC=0 ids 一致" || bad "opt-out ids 不一致"
else
  bad "opt-out 运行失败"
fi

echo
(( fail )) && echo FAIL || echo PASS
