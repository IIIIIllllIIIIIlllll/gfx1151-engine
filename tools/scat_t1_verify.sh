#!/usr/bin/env bash
# scat_t1_verify.sh — 一键验证 scatter_norm_inj 的 verify 档 T=1（QWENOX_SCAT_T1，默认开）：
#   P<=8 时 k_gr_scatter_norm_inj_bf16 从 T=4（4 块，4 token 串行）换成 T=1（4P 块）。
#   逐 (t,b) 算术与 T 无关（proto 实测 R/Rhat 逐 bit 不变），纯并行度提升。
#   关闭：QWENOX_SCAT_T1=0
# 检查：
#   1. 单元原型 tools/injfuse_proto.cu（含 T=1）：R/Rhat 逐 bit + w4 误差 → PASS
#   2. greedy ids：8K MTP / 8K chain / 32K MTP，BIN vs BASE 完全一致；opt-out 一致
#   3. 打印各组 tok/s 供肉眼对比
# 用法: bash tools/scat_t1_verify.sh     （约 6 分钟，结尾输出 PASS / FAIL）
#   BIN=build/qwenox-engine（默认）  BASE=build/qwenox.base（默认）
# 日志: logs/stv_*.log，汇总 logs/scat_t1_verify.out
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
BIN="${BIN:-build/qwenox-engine}"
BASE="${BASE:-build/qwenox.base}"
HIPCC="${HIPCC:-/opt/rocm/bin/hipcc}"
OUT=logs/scat_t1_verify.out
mkdir -p logs
exec > >(tee "$OUT") 2>&1
fail=0
bad() { echo "  FAIL: $*"; fail=1; }
ok() { echo "  OK: $*"; }
[[ -x "$BIN" && -x "$BASE" ]] || { echo "缺二进制（先 bash build.sh）"; echo FAIL; exit 1; }
echo "== scat_t1_verify  $(date '+%F %T')  BIN=$BIN BASE=$BASE"

echo "-- 1. 单元原型（T=1/4/8/16 的 R/Rhat 逐 bit + P=4 计时）"
sed -n '/\[gr-injfuse-begin\]/,/\[gr-injfuse-end\]/p' src/gpu/parts/22_kernels_prefill.inc > tools/gr_injfuse_kernel.inc
if "$HIPCC" -O3 --offload-arch=gfx1151 -o build/injfuse_proto tools/injfuse_proto.cu >logs/stv_proto_build.log 2>&1; then
  build/injfuse_proto 5 >logs/stv_proto.log 2>&1
  grep -E "P=4 timing" -A 5 logs/stv_proto.log | sed 's/^/  /'
  tail -1 logs/stv_proto.log | grep -q "INJFUSE PROTO: PASS" && ok "原型 PASS（含 T=1 逐 bit）" || bad "原型 FAIL（logs/stv_proto.log）"
else
  bad "原型编译失败（logs/stv_proto_build.log）"
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
specline() { grep -h 'spec: \| chain: ' "logs/$1.log" | tail -1 | cut -c1-90; }

echo "-- 2. ids 一致：8K MTP / 8K chain / 32K MTP"
for t in "8k mtp" "8k chain" "32k mtp"; do
  set -- $t; P=$1; M=$2
  A=(); [[ $M == chain ]] && A=(QWENOX_SPEC_CHAIN=1)
  if run stv_${P}_${M}_base "$BASE" "$P" "${A[@]}" && run stv_${P}_${M}_new "$BIN" "$P" "${A[@]}"; then
    echo "  base: $(specline stv_${P}_${M}_base)"
    echo "  new : $(specline stv_${P}_${M}_new)"
    [[ "$(ids stv_${P}_${M}_base)" == "$(ids stv_${P}_${M}_new)" ]] \
      && ok "$P $M ids 一致" || bad "$P $M ids 不一致"
  else
    bad "$P $M 运行失败"
  fi
done

echo "-- 3. opt-out 与 BASE 一致"
if run stv_off "$BIN" 8k QWENOX_SCAT_T1=0; then
  [[ "$(ids stv_off)" == "$(ids stv_8k_mtp_base)" ]] \
    && ok "QWENOX_SCAT_T1=0 ids 一致" || bad "opt-out ids 不一致"
else
  bad "opt-out 运行失败"
fi

echo
(( fail )) && echo FAIL || echo PASS
