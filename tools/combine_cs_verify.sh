#!/usr/bin/env bash
# combine_cs_verify.sh — 一键验证 gr_combine_b_hc_bf16 列分块（QWENOX_COMBINE_CS，默认开）：
#   P<=8 时从 P 块改成 dim3(P,4) 块（每块 d/4 列）。每列单线程独立计算，逐 bit 等价。
#   关闭：QWENOX_COMBINE_CS=0
# 检查：
#   1. ktest 单测：gr_combine_colsplit 必须 mismatches=0（build.sh test）
#   2. greedy ids：8K MTP / 32K MTP，BIN vs BASE 完全一致；opt-out 一致
# 用法: bash tools/combine_cs_verify.sh   （约 5 分钟，结尾输出 PASS / FAIL）
#   BIN=build/qwenox-engine（默认）  BASE=build/qwenox.base（默认）  SKIP_KTEST=1 跳过单测
# 日志: logs/ccv_*.log，汇总 logs/combine_cs_verify.out
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
BIN="${BIN:-build/qwenox-engine}"
BASE="${BASE:-build/qwenox.base}"
OUT=logs/combine_cs_verify.out
mkdir -p logs
exec > >(tee "$OUT") 2>&1
fail=0
bad() { echo "  FAIL: $*"; fail=1; }
ok() { echo "  OK: $*"; }
[[ -x "$BIN" && -x "$BASE" ]] || { echo "缺二进制（先 bash build.sh）"; echo FAIL; exit 1; }
echo "== combine_cs_verify  $(date '+%F %T')  BIN=$BIN BASE=$BASE"

if [[ -z "${SKIP_KTEST:-}" ]]; then
  echo "-- 1. ktest 单测（gr_combine_colsplit 需 mismatches=0）"
  if bash build.sh test >logs/ccv_ktest.log 2>&1; then
    grep -E "gr_bf16_combine|gr_combine_colsplit" logs/ccv_ktest.log | sed 's/^/  /'
    grep -A 1 "gr_combine_colsplit" logs/ccv_ktest.log | grep -q "mismatches=0" \
      && ok "colsplit 逐 bit" || bad "colsplit 有差异（logs/ccv_ktest.log）"
  else
    bad "ktest 失败（logs/ccv_ktest.log）"
  fi
fi

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
specline() { grep -h 'spec: ' "logs/$1.log" | tail -1 | cut -c1-90; }

echo "-- 2. ids 一致：8K / 32K MTP"
for P in 8k 32k; do
  if run ccv_${P}_base "$BASE" "$P" && run ccv_${P}_new "$BIN" "$P"; then
    echo "  base: $(specline ccv_${P}_base)"
    echo "  new : $(specline ccv_${P}_new)"
    [[ "$(ids ccv_${P}_base)" == "$(ids ccv_${P}_new)" ]] \
      && ok "$P ids 一致" || bad "$P ids 不一致"
  else
    bad "$P 运行失败"
  fi
done

echo "-- 3. opt-out 与 BASE 一致"
if run ccv_off "$BIN" 8k QWENOX_COMBINE_CS=0; then
  [[ "$(ids ccv_off)" == "$(ids ccv_8k_base)" ]] \
    && ok "QWENOX_COMBINE_CS=0 ids 一致" || bad "opt-out ids 不一致"
else
  bad "opt-out 运行失败"
fi

echo
(( fail )) && echo FAIL || echo PASS
