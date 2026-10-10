#!/usr/bin/env bash
# vbase_verify.sh — 验证 P3-A 第一段（batch 路径 base 位置设备化，d_vbase；QWENOX_VBASE=0 回退）
# 和 iproj GEMV 长上下文，全部用生产权重（start_hgn.sh --check 取自 service.conf）。
#   set_vbase 的暂存是 4 格 pinned 环 + event：多 chunk prefill 连续改 base 时，
#   前一个 chunk 的异步拷贝不会读到后一个 chunk 的值（不依赖别处碰巧的同步）。
# 检查（ids 必须逐 token 一致的判 FAIL；iproj 为非逐 bit 改动，DIFF 只报 WARN）：
#   0. 权重必须是 HQ（日志含 "dense: 8-bit"），否则直接 FAIL——防止 service.conf 被覆盖后白测
#   1. 8K  greedy MTP γ=4：默认 vs QWENOX_VBASE=0
#   2. 8K  greedy chain γ=4：默认 vs QWENOX_VBASE=0
#   3. 32K greedy MTP γ=4（prefill 2 个 chunk）：默认 vs QWENOX_VBASE=0
#   4. 8K  普通 decode GEN=32，QWENOX_PREFILL_CHUNK=2048（≥3 个 chunk）：默认 vs QWENOX_VBASE=0
#   5. 64K greedy MTP γ=4：默认 vs QWENOX_IPROJ_GEMV=0（SAME/DIFF，DIFF 记 WARN）
# 用法: bash tools/vbase_verify.sh   （约 12 分钟，结尾 PASS / FAIL）
#   BIN=build/qwenox-engine（默认）  SKIP_64K=1 跳过第 5 步
# 日志: logs/vbv_*.log，汇总 logs/vbase_verify.out
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
BIN="${BIN:-build/qwenox-engine}"
OUT=logs/vbase_verify.out
mkdir -p logs
exec > >(tee "$OUT") 2>&1
fail=0; warn=0
bad() { echo "  FAIL: $*"; fail=1; }
ok() { echo "  OK: $*"; }
[[ -x "$BIN" ]] || { echo "缺二进制: $BIN（先 bash build.sh）"; echo FAIL; exit 1; }
echo "== vbase_verify  $(date '+%F %T')  BIN=$BIN"

# run LABEL LEN [K=V...]（环境变量 SPEC/GEN 由调用方前置）-> logs/LABEL.log
run() {
  local L=$1 N=$2; shift 2
  BIN=$BIN bash tools/pp_prod.sh "$N" "$L" "$@" > "logs/$L.stdout" 2>&1
  local rc=$?
  if (( rc )) || ! grep -q '^ids:' "logs/$L.log"; then
    echo "  $L: 运行失败 rc=$rc"; grep -h 'rror\|abort\|exception' "logs/$L.log" | head -3
    return 1
  fi
  grep -q 'dense: 8-bit' "logs/$L.log" || { echo "  $L: 不是 HQ 权重（无 dense: 8-bit）"; return 2; }
}
# 只取 token 数字：GEN 模式下 stdout 的 ids 行不带换行，stderr 的
# "decode: N tokens in T ms" 会接在同一行，耗时每次不同 → 必须截掉。
ids() { grep -a '^ids:' "logs/$1.log" | sed -E 's/^ids://; s/[^0-9 ].*$//'; }
speed() { grep -h 'spec: \| chain: \|decode' "logs/$1.log" | tail -1 | sed 's/.*tokens in/in/' | cut -c1-70; }
nchunk() { grep -c '\] prefill: ' "logs/$1.log"; }  # 只数带时间戳的 chunk 行（不含 "moe prefill:"）

# pair STEP LABEL LEN "默认额外开关" "对照开关" [strict=1]
pair() {
  local S=$1 L=$2 N=$3 A=$4 B=$5 strict=${6:-1}
  echo "-- $S"
  # shellcheck disable=SC2086
  if run "${L}_a" "$N" $A && run "${L}_b" "$N" $A $B; then
    echo "  默认: $(speed "${L}_a")  [prefill chunk×$(nchunk "${L}_a")]"
    echo "  对照: $(speed "${L}_b")  [$B]"
    if [[ "$(ids "${L}_a")" == "$(ids "${L}_b")" ]]; then ok "ids 一致"
    elif (( strict )); then bad "ids 不一致"
    else echo "  WARN: ids 不一致（非逐 bit 改动，近似并列 argmax 翻转，需人工看）"; warn=1
    fi
  else
    local rc=$?
    (( rc == 2 )) && { bad "权重不是 HQ，检查 service.conf"; echo FAIL; exit 1; }
    bad "运行失败"
  fi
}

SPEC=256 GAMMA=4 pair "1. 8K MTP γ=4：VBASE on/off"   vbv_8k   8k  "" "QWENOX_VBASE=0"
SPEC=256 GAMMA=4 pair "2. 8K chain γ=4：VBASE on/off" vbv_8kc  8k  "QWENOX_SPEC_CHAIN=1" "QWENOX_VBASE=0"
SPEC=256 GAMMA=4 pair "3. 32K MTP γ=4：VBASE on/off"  vbv_32k  32k "" "QWENOX_VBASE=0"
GEN=32 pair "4. 8K 多 chunk（QWENOX_PREFILL_CHUNK=2048）：VBASE on/off" vbv_8kch 8k "QWENOX_PREFILL_CHUNK=2048" "QWENOX_VBASE=0"
if [[ -f logs/vbv_8kch_a.log ]] && (( $(nchunk vbv_8kch_a) < 3 )); then
  bad "第 4 步 prefill 只有 $(nchunk vbv_8kch_a) 个 chunk，没测到多 chunk"
fi
if [[ -z "${SKIP_64K:-}" ]]; then
  SPEC=256 GAMMA=4 pair "5. 64K MTP γ=4：IPROJ_GEMV on/off" vbv_64k 64k "" "QWENOX_IPROJ_GEMV=0" 0
fi

echo "== 汇总 $(date '+%T')"
if (( fail )); then echo FAIL; exit 1; fi
(( warn )) && echo "（有 WARN，见上）"
echo PASS
