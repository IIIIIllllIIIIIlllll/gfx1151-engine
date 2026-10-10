#!/usr/bin/env bash
# gamma_sweep.sh — MTP 投机 γ 扫描（生产 HQ 权重，pp_prod.sh SPEC=256）
#   greedy：γ ∈ GAMMAS × 长度 ∈ LENS
#   采样：  8K，γ ∈ SGAMMAS × 种子 ∈ SEEDS（QWENOX_SPEC_SAMPLE=0.7,20,0.8,seed）
# 每格输出 tok/s、commit/round、ms/round。只做测速，不判 ids；结尾 DONE / FAIL（运行失败或非 HQ）。
# 用法: bash tools/gamma_sweep.sh   （约 25 分钟）
#   BIN=build/qwenox-engine  GAMMAS="4 5 6 7"  LENS="8k 32k 64k"  SGAMMAS="3 4 5 6 7"  SEEDS="1 2"
#   TAG=_x（日志名前缀后缀）  EXTRA="QWENOX_X=0"（greedy 附加开关）  OUT=logs/xxx.out
#   只跑某一段：GAMMAS="" 跳过 greedy，SEEDS="" 跳过采样
# 日志: logs/gs${TAG}_*.log，汇总 logs/gamma_sweep.out
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
BIN="${BIN:-build/qwenox-engine}"
GAMMAS="${GAMMAS-4 5 6 7}"; LENS="${LENS:-8k 32k 64k}"
SGAMMAS="${SGAMMAS-3 4 5 6 7}"; SEEDS="${SEEDS-1 2}"
TAG="${TAG:-}"; EXTRA="${EXTRA:-}"
OUT="${OUT:-logs/gamma_sweep${TAG}.out}"
mkdir -p logs
exec > >(tee "$OUT") 2>&1
[[ -x "$BIN" ]] || { echo "缺二进制: $BIN"; echo FAIL; exit 1; }
echo "== gamma_sweep  $(date '+%F %T')  BIN=$BIN"
fail=0

# one LABEL LEN GAMMA [K=V...] -> 打印一行结果
one() {
  local L=$1 N=$2 G=$3; shift 3
  SPEC=256 GAMMA=$G BIN=$BIN bash tools/pp_prod.sh "$N" "$L" "$@" > "logs/$L.stdout" 2>&1
  local rc=$? s
  s=$(grep -aE '^spec(-sample)?:' "logs/$L.log" 2>/dev/null | tail -1)
  if (( rc )) || [[ -z "$s" ]]; then echo "  $L: 运行失败 rc=$rc"; fail=1; return; fi
  grep -q 'dense: 8-bit' "logs/$L.log" || { echo "  $L: 不是 HQ 权重，停止"; echo FAIL; exit 1; }
  # spec: 257 tokens in 4.83 s = 53.2 tok/s | rounds=56 commit/round=4.59 rollbacks=14
  echo "$s" | awk -v l="$N" -v g="$G" -v x="$*" '{
    for (i=1;i<=NF;i++){ if($i=="s") t=$(i-1); if($i=="tok/s") v=$(i-1);
      if($i~/^rounds=/){split($i,a,"=");r=a[2]} if($i~/^commit\/round=/){split($i,b,"=");c=b[2]} }
    printf "  %-4s γ=%s %-28s %6.1f tok/s  commit/round %s  %5.1f ms/round\n", l, g, x, v, c, t*1000/r }'
}

echo "-- greedy"
# shellcheck disable=SC2086
for N in $LENS; do for G in $GAMMAS; do one "gs${TAG}_${N}_g$G" "$N" "$G" $EXTRA; done; done
echo "-- 采样 8K（temp 0.7 / top_k 20 / top_p 0.8）"
for S in $SEEDS; do for G in $SGAMMAS; do
  one "gs${TAG}_s${S}_g$G" 8k "$G" "QWENOX_SPEC_SAMPLE=0.7,20,0.8,$S" $EXTRA
done; done

echo "== 结束 $(date '+%T')"
(( fail )) && { echo FAIL; exit 1; }
echo DONE
