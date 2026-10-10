#!/usr/bin/env bash
# brow_r_verify.sh — 一键验证 k_q8g32_gemv_brow_r 行平铺（默认开，QWENOX_BROW_R=0 回退）：
#   q8g32 brow 分支（xbf16 && cols>=4096 && rows<=4096：HC down 320x10240、out_proj
#   2560x6144 的 verify 档）从每块 1 行改成每块 RPB=4 行，x 每组只取一次全块复用。
#   每 (row,p) 的 group 顺序/表达式/归约树逐字复刻 brow → 构造性逐 bit。
#   实测整机收益 ≈0（x 重读与 DRAM 权重流本就重叠）；保留作 L2 流量削减，零风险。
# 检查：
#   1. ktest 单测：brow_r_bitexact 必须 mismatches=0（含 319 尾行）
#   2. greedy ids：8K MTP / 8K chain / 32K MTP，BIN vs BASE 完全一致；opt-out 一致
# 用法: bash tools/brow_r_verify.sh   （约 7 分钟，结尾输出 PASS / FAIL）
#   BIN=build/qwenox-engine（默认）  BASE=build/qwenox.base（默认）  SKIP_KTEST=1 跳过单测
# 日志: logs/brv_*.log，汇总 logs/brow_r_verify.out
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
BIN="${BIN:-build/qwenox-engine}"
BASE="${BASE:-build/qwenox.base}"
OUT=logs/brow_r_verify.out
mkdir -p logs
exec > >(tee "$OUT") 2>&1
fail=0
bad() { echo "  FAIL: $*"; fail=1; }
ok() { echo "  OK: $*"; }
[[ -x "$BIN" && -x "$BASE" ]] || { echo "缺二进制（先 bash build.sh）"; echo FAIL; exit 1; }
echo "== brow_r_verify  $(date '+%F %T')  BIN=$BIN BASE=$BASE"

if [[ -z "${SKIP_KTEST:-}" ]]; then
  echo "-- 1. ktest 单测（brow_r_bitexact 需 mismatches=0）"
  if bash build.sh test >logs/brv_ktest.log 2>&1; then
    grep -E "brow_r" logs/brv_ktest.log | sed 's/^/  /'
    grep -q "brow_r_bitexact .*PASS" logs/brv_ktest.log \
      && ok "brow_r 逐 bit" || bad "brow_r 有差异（logs/brv_ktest.log）"
  else
    bad "ktest 失败（logs/brv_ktest.log）"
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

echo "-- 2. ids 一致：8K MTP / 8K chain / 32K MTP"
if run brv_8_base "$BASE" 8k && run brv_8_new "$BIN" 8k; then
  [[ "$(ids brv_8_base)" == "$(ids brv_8_new)" ]] \
    && ok "8k ids 一致" || bad "8k ids 不一致"
else
  bad "8k 运行失败"
fi
if run brv_c_base "$BASE" 8k QWENOX_SPEC_CHAIN=1 && run brv_c_new "$BIN" 8k QWENOX_SPEC_CHAIN=1; then
  [[ "$(ids brv_c_base)" == "$(ids brv_c_new)" ]] \
    && ok "8k chain ids 一致" || bad "8k chain ids 不一致"
else
  bad "8k chain 运行失败"
fi
if run brv_32_base "$BASE" 32k && run brv_32_new "$BIN" 32k; then
  [[ "$(ids brv_32_base)" == "$(ids brv_32_new)" ]] \
    && ok "32k ids 一致" || bad "32k ids 不一致"
else
  bad "32k 运行失败"
fi

echo "-- 3. opt-out 与 BASE 一致"
if run brv_off "$BIN" 8k QWENOX_BROW_R=0; then
  [[ "$(ids brv_off)" == "$(ids brv_8_base)" ]] \
    && ok "QWENOX_BROW_R=0 ids 一致" || bad "opt-out ids 不一致"
else
  bad "opt-out 运行失败"
fi

echo
(( fail )) && echo FAIL || echo PASS
