#!/usr/bin/env bash
# tg_verify.sh — 一键验证 TG/MTP decode 改动（09-28）：
#   D'  draft lm_head 用 4-bit q4cp 副本        关闭：QWENOX_DRAFT_LM_Q4=0
#   B2  两段 argmax（k_argmax_p1/p2）          关闭：QWENOX_ARGMAX_OLD=1
#   A   采样准备搬到 GPU（修正 + radix top-k） 关闭：QWENOX_SMS_GPU=0
# 步骤：
#   1. 单测 tools/tg_kernels_test.cu（kernel 从当前 21_kernels_ple.inc 现抽）：
#      argmax / top-k / corr_apply 对 CPU 参考，含词表尾部最大值、并列、全 -inf、溢出回退
#   2. greedy 逐 token：8K prompt，--spec-gen 256 γ=3，MTP 与 chain 两种 drafter，
#      默认 vs 三个开关全关，ids 必须完全一致（BASE=旧二进制 时再和它比一次）
#   3. 采样分布自检：QWENOX_SMS_CHECK=1（每行把稀疏分布和 dense prepare() 比对），
#      MTP、chain、chain+presence/frequency 各一次，bad 必须为 0
#   4. 采样速度：chain drafter（API 默认），temp 1.0 / top_k 20 / top_p 0.95，
#      3 个 seed，默认 vs QWENOX_SMS_GPU=0，按 ms/轮 比较，默认需快 ≥5%
# 用法: bash tools/tg_verify.sh       （约 4 分钟，结尾输出 PASS / FAIL）
#   BIN=build/qwenox-engine（默认）  BASE=build/qwenox.base（可选，旧代码二进制）
# 日志: logs/tgv_*.log，汇总 logs/tg_verify.out
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
BIN="${BIN:-build/qwenox-engine}"
BASE="${BASE:-}"
HIPCC="${HIPCC:-/opt/rocm/bin/hipcc}"
OUT=logs/tg_verify.out
mkdir -p logs
exec > >(tee "$OUT") 2>&1
fail=0
bad() { echo "  FAIL: $*"; fail=1; }
ok() { echo "  OK: $*"; }
OFF="QWENOX_ARGMAX_OLD=1 QWENOX_DRAFT_LM_Q4=0 QWENOX_SMS_GPU=0"
SAMPLE="1.0,20,0.95"

[[ -x "$BIN" ]] || { echo "缺二进制: $BIN（先 bash build.sh）"; echo FAIL; exit 1; }
[[ -z "$BASE" || -x "$BASE" ]] || { echo "缺二进制: $BASE"; echo FAIL; exit 1; }
echo "== tg_verify  $(date '+%F %T')  BIN=$BIN ${BASE:+BASE=$BASE}"

# run LABEL BIN [K=V...]  -> logs/LABEL.log
run() {
  local L=$1 B=$2; shift 2
  BIN=$B SPEC=256 GAMMA=3 bash tools/pp_prod.sh 8k "$L" "$@" > "logs/$L.stdout" 2>&1
  local rc=$?
  if (( rc )) || ! grep -q '^ids:' "logs/$L.log"; then
    echo "  $L: 运行失败 rc=$rc"; grep -h 'rror\|abort\|exception' "logs/$L.log" | head -3
    return 1
  fi
}
ids() { grep '^ids:' "logs/$1.log"; }
specline() { grep -h 'spec: \|spec-sample: \|chain-sample: \| chain: ' "logs/$1.log" | tail -1; }
# "N tokens in T s = R tok/s | rounds=K" -> "T K"
t_r() { specline "$1" | sed -E 's/.* in ([0-9.]+) s = .*rounds=([0-9]+).*/\1 \2/'; }

echo "-- 1. kernel 单测"
T=$(mktemp -d)
sed -n '/\[tgk-begin\]/,/\[tgk-end\]/p' src/gpu/parts/21_kernels_ple.inc > "$T/tg_kernels.inc"
if "$HIPCC" --offload-arch=gfx1151 -O3 -I"$T" tools/tg_kernels_test.cu -o "$T/t" > "$T/cc.log" 2>&1; then
  "$T/t" | tee "$T/run.log" | sed 's/^/  /'
  [[ "$(tail -1 "$T/run.log")" == PASS ]] && ok "kernel 单测" || bad "kernel 单测"
else
  cat "$T/cc.log" | head -20; bad "单测编译失败"
fi
rm -rf "$T"

echo "-- 2. greedy 逐 token（默认 vs 开关全关）"
for c in 0 1; do
  if run tgv_g_new_c$c "$BIN" QWENOX_SPEC_CHAIN=$c && run tgv_g_off_c$c "$BIN" QWENOX_SPEC_CHAIN=$c $OFF; then
    echo "  chain=$c 默认: $(specline tgv_g_new_c$c | sed 's/.*tokens in/in/' | cut -c1-80)"
    echo "  chain=$c 关闭: $(specline tgv_g_off_c$c | sed 's/.*tokens in/in/' | cut -c1-80)"
    [[ "$(ids tgv_g_new_c$c)" == "$(ids tgv_g_off_c$c)" ]] && ok "chain=$c ids 一致" || bad "chain=$c ids 不一致"
  else
    bad "chain=$c greedy 运行失败"
  fi
done
if [[ -n "$BASE" ]]; then
  if run tgv_g_base "$BASE"; then
    echo "  base  : $(specline tgv_g_base | sed 's/.*tokens in/in/' | cut -c1-80)"
    [[ "$(ids tgv_g_new_c0)" == "$(ids tgv_g_base)" ]] && ok "与 BASE ids 一致" || bad "与 BASE ids 不一致"
  else
    bad "BASE 运行失败"
  fi
fi

echo "-- 3. 采样分布自检（QWENOX_SMS_CHECK=1）"
# 名称:chain:采样参数（第三项带 presence/frequency，走 k_corr_apply 与草稿前缀惩罚）
for spec in "mtp:0:$SAMPLE,7" "chain:1:$SAMPLE,7" "chain_pen:1:0.8,40,0.9,5,0.5,0.3"; do
  IFS=: read nm c sp <<<"$spec"
  L=tgv_chk_$nm
  if run $L "$BIN" QWENOX_SPEC_CHAIN=$c QWENOX_SMS_CHECK=1 QWENOX_SPEC_SAMPLE=$sp; then
    last=$(grep -h '^\[sms-check\] n=' "logs/$L.log" | tail -1)
    nmis=$(grep -c 'MISMATCH' "logs/$L.log")
    n=$(sed -E 's/.* n=([0-9]+) .*/\1/' <<<"$last")
    echo "  $nm ($sp): ${last:-无 sms-check 输出}"
    if [[ -n "$last" ]] && (( n >= 64 && nmis == 0 )) && grep -q ' bad=0 ' <<<"$last"; then
      ok "$nm 稀疏分布 = dense 分布"
    else
      bad "$nm 自检未通过（MISMATCH $nmis 行）"
    fi
  else
    bad "$nm 采样运行失败"
  fi
done

echo "-- 4. 采样速度（chain drafter，3 个 seed）"
tn=0; rn=0; to=0; ro=0; sok=1
for s in 7 11 23; do
  if run tgv_s_new_$s "$BIN" QWENOX_SPEC_CHAIN=1 QWENOX_SPEC_SAMPLE=$SAMPLE,$s &&
     run tgv_s_off_$s "$BIN" QWENOX_SPEC_CHAIN=1 QWENOX_SPEC_SAMPLE=$SAMPLE,$s QWENOX_SMS_GPU=0; then
    read a b <<<"$(t_r tgv_s_new_$s)"; read x y <<<"$(t_r tgv_s_off_$s)"
    echo "  seed $s: 默认 $(specline tgv_s_new_$s | sed -E 's/.*= ([0-9.]+ tok\/s).*/\1/')（${b} 轮）  SMS 关 $(specline tgv_s_off_$s | sed -E 's/.*= ([0-9.]+ tok\/s).*/\1/')（${y} 轮）"
    tn=$(awk "BEGIN{print $tn+$a}"); rn=$((rn + b)); to=$(awk "BEGIN{print $to+$x}"); ro=$((ro + y))
  else
    sok=0
  fi
done
if (( sok && rn && ro )); then
  mn=$(awk "BEGIN{printf \"%.1f\", $tn*1000/$rn}"); mo=$(awk "BEGIN{printf \"%.1f\", $to*1000/$ro}")
  echo "  ms/轮：默认 $mn  vs  QWENOX_SMS_GPU=0 $mo"
  awk "BEGIN{exit !($mn <= 0.95*$mo)}" && ok "GPU 采样每轮快 ≥5%" || bad "GPU 采样提速不足 5%"
else
  bad "采样速度运行失败"
fi

echo
(( fail )) && echo FAIL || echo PASS
