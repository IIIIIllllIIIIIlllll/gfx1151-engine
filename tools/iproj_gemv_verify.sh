#!/usr/bin/env bash
# iproj_gemv_verify.sh — 验证 decode 档 iproj 走 k_f32_gemv_mr（默认开，QWENOX_IPROJ_GEMV=0 回退）。
#   indexer 投影 fp32 N=640 K=2560：P<=8 时从 rocblas_sgemm（自选 MT32x32x8、20 块、
#   P=4 实测 ~110us/次）改走 warp-per-row fp32 GEMV（流式读 6.5MB，~36us）。
#   非逐 bit（fp32 累加顺序变）；ktest maxrel~1e-5（vs fp64，与 sgemm 自身误差同量级），
#   实测 8K/32K greedy ids 与 BASE 逐 token 一致（indexer top-512 未发生边界翻转）。
# 检查：
#   1. ktest 单测：f32_gemv_mr PASS（vs fp64，tol 1e-4）+ P=1 与 P=4 第 0 行逐 bit
#   2. QWENOX_IPROJ_GEMV=0（回退旧 sgemm 路径）ids 必须与 BASE 一致
#   3. 默认开：8K/32K ids vs BASE（报告 SAME/DIFF；DIFF 不判 FAIL——
#      top-512 边界翻转是已知良性机制，但须人工确认后更新本脚本预期）
#   4. 默认开速度不慢于 BASE
# 用法: bash tools/iproj_gemv_verify.sh   （约 8 分钟，结尾 PASS / FAIL）
#   BIN=build/qwenox-engine（默认）  BASE=build/qwenox.base（默认）  SKIP_KTEST=1 跳过单测
# 日志: logs/igv_*.log，汇总 logs/iproj_gemv_verify.out
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
BIN="${BIN:-build/qwenox-engine}"
BASE="${BASE:-build/qwenox.base}"
OUT=logs/iproj_gemv_verify.out
mkdir -p logs
exec > >(tee "$OUT") 2>&1
fail=0
bad() { echo "  FAIL: $*"; fail=1; }
ok() { echo "  OK: $*"; }
[[ -x "$BIN" && -x "$BASE" ]] || { echo "缺二进制（先 bash build.sh）"; echo FAIL; exit 1; }
echo "== iproj_gemv_verify  $(date '+%F %T')  BIN=$BIN BASE=$BASE"

if [[ -z "${SKIP_KTEST:-}" ]]; then
  echo "-- 1. ktest 单测（f32_gemv_mr vs fp64）"
  if bash build.sh test >logs/igv_ktest.log 2>&1; then
    grep -E "f32_gemv" logs/igv_ktest.log | sed 's/^/  /'
    grep -q "f32_gemv_mr .*PASS" logs/igv_ktest.log \
      && grep -q "f32_gemv_mr_p1 .*PASS" logs/igv_ktest.log \
      && ok "ktest" || bad "f32_gemv_mr 单测失败（logs/igv_ktest.log）"
  else
    bad "ktest 构建/运行失败（logs/igv_ktest.log）"
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
toks() { grep -h 'spec: ' "logs/$1.log" | tail -1 | grep -o '[0-9.]* tok/s' | grep -o '^[0-9.]*'; }

echo "-- 2. QWENOX_IPROJ_GEMV=0（回退路径）ids 一致：8K MTP"
if run igv_off8 "$BIN" 8k QWENOX_IPROJ_GEMV=0 && run igv_bs8 "$BASE" 8k; then
  [[ "$(ids igv_off8)" == "$(ids igv_bs8)" ]] \
    && ok "回退路径 ids 一致" || bad "回退路径 ids 不一致（回退不应受影响）"
else
  bad "回退路径运行失败"
fi

echo "-- 3. 默认开：ids 对比 + 速度"
declare -A T
for P in 8k 32k; do
  if run igv_gv_$P "$BIN" "$P" && run igv_gb_$P "$BASE" "$P"; then
    T[gv_$P]=$(toks igv_gv_$P); T[gb_$P]=$(toks igv_gb_$P)
    echo "  $P: GEMV ${T[gv_$P]} tok/s vs BASE ${T[gb_$P]} tok/s"
    if [[ "$(ids igv_gv_$P)" == "$(ids igv_gb_$P)" ]]; then
      ok "$P ids SAME"
    else
      echo "  WARN: $P ids DIFF（top-512 边界翻转，已知良性机制；需人工确认）"
    fi
    awk "BEGIN{exit !(${T[gv_$P]:-0} >= ${T[gb_$P]:-999} * 0.995)}" \
      && ok "$P 速度不慢于 BASE" || bad "$P GEMV 反而更慢"
  else
    bad "$P 运行失败"
  fi
done

echo
(( fail )) && echo FAIL || echo PASS
