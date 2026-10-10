#!/usr/bin/env bash
# gamma_adapt_verify.sh — 自适应 γ（Model::GammaCtl，--gamma 0 / API 未设 QWENOX_SPEC_GAMMA）一键验证
#   1. 固定 γ 路径不变：BIN vs BASE（改动前二进制），8K greedy γ=4（MTP、chain）+ 采样 γ=3，ids 必须一致
#   2. 速度：自适应 vs 旧 auto 默认（greedy γ=4 / 采样 γ=3），SPEC=512，多样本汇总
#      greedy 5 个上下文（32K 文本的 6k..22k 前缀）× γ{4,7,自适应}；采样 8K × 6 seed × γ{3,4,7,自适应}
#      汇总 tok/s（总 token / 总秒）自适应 ≥ 0.98× 旧默认；64K（重复 prompt，接受率差）单样本 ≥ 0.95× γ=4
#   3. 采样分布自检：自适应下 QWENOX_SMS_CHECK=1（MTP、chain），bad 必须为 0
# 生产 HQ 权重（日志须含 "dense: 8-bit"）。用法: bash tools/gamma_adapt_verify.sh （约 16 分钟，结尾 PASS / FAIL）
#   BIN=build/qwenox-engine  BASE=build/qwenox.pre-adapt  SKIP_64K=1（跳过 64K）
#   PROMPTS="6000 10000 14000 18000 22000"  SEEDS="1 2 3 4 5 6"
# 注：64K 用的 tok65536.txt 是 tok32768.txt 重复两遍，接受率偏低（正好用来测"接受率差时不掉速"）
# 日志: logs/gav_*.log，汇总 logs/gamma_adapt_verify.out
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
BIN="${BIN:-build/qwenox-engine}"
BASE="${BASE:-build/qwenox.pre-adapt}"
OUT=logs/gamma_adapt_verify.out
mkdir -p logs
exec > >(tee "$OUT") 2>&1
fail=0
bad() { echo "  FAIL: $*"; fail=1; }
ok() { echo "  OK: $*"; }
SMP="0.7,20,0.8"
[[ -x "$BIN" ]] || { echo "缺二进制: $BIN"; echo FAIL; exit 1; }
[[ -x "$BASE" ]] || { echo "缺二进制: $BASE（改动前的 build/qwenox-engine 备份）"; echo FAIL; exit 1; }
echo "== gamma_adapt_verify  $(date '+%F %T')  BIN=$BIN BASE=$BASE"

# run LABEL BIN LEN SPEC GAMMA [K=V...] -> logs/LABEL.log
run() {
  local L=$1 B=$2 N=$3 S=$4 G=$5; shift 5
  SPEC=$S GAMMA=$G BIN=$B bash tools/pp_prod.sh "$N" "$L" "$@" > "logs/$L.stdout" 2>&1
  local rc=$?
  if (( rc )) || ! grep -aq '^ids:' "logs/$L.log"; then
    echo "  $L: 运行失败 rc=$rc"; grep -ah 'rror\|abort\|exception' "logs/$L.log" | head -3
    return 1
  fi
  grep -aq 'dense: 8-bit' "logs/$L.log" || { echo "  $L: 不是 HQ 权重，停止"; echo FAIL; exit 1; }
}
ids() { grep -a '^ids:' "logs/$1.log" | sed -E 's/^ids://; s/[^0-9 ].*$//'; }
specline() { grep -ahE '^spec(-sample)?: |chain(-sample)?: [0-9]+ tokens' "logs/$1.log" | tail -1; }
tps() { specline "$1" | sed -E 's/.* = ([0-9.]+) tok\/s.*/\1/'; }
cpr() { specline "$1" | sed -E 's/.*commit\/round=([0-9.]+).*/\1/'; }
adline() { grep -ah 'gamma-adapt:' "logs/$1.log" | tail -1 | sed 's/^.*gamma-adapt: //'; }

echo "-- 1. 固定 γ 路径与改动前逐 token 一致（8K，SPEC=256）"
for spec in "mtp_g4:4:QWENOX_SPEC_CHAIN=0" "chain_g4:4:QWENOX_SPEC_CHAIN=1" "smp_g3:3:QWENOX_SPEC_SAMPLE=$SMP,1"; do
  IFS=: read nm g ex <<<"$spec"
  if run gav_fix_${nm}_new "$BIN" 8k 256 "$g" $ex && run gav_fix_${nm}_base "$BASE" 8k 256 "$g" $ex; then
    if [[ -n "$(ids gav_fix_${nm}_new)" && "$(ids gav_fix_${nm}_new)" == "$(ids gav_fix_${nm}_base)" ]]; then
      ok "$nm ids 一致"
    else
      bad "$nm ids 不一致"
    fi
    grep -aq 'gamma-adapt:' "logs/gav_fix_${nm}_new.log" && bad "$nm 固定 γ 却打印了 gamma-adapt"
  else
    bad "$nm 运行失败"
  fi
done

echo "-- 2. 速度：自适应（γ=0）vs 旧默认，多样本汇总（SPEC=512）"
# 换 γ 会改变生成轨迹（greedy 近并列翻转 / 采样 RNG 对齐），单 prompt 单 seed 的
# tok/s 差 ±10% 都可能是轨迹噪声，所以按多个上下文 / seed 汇总：总 token / 总秒数。
# greedy：32K 文本的不同长度前缀当不同上下文；采样：8K × 多个 seed。
PROMPTS="${PROMPTS:-6000 10000 14000 18000 22000}"
SEEDS="${SEEDS:-1 2 3 4 5 6}"
PD=logs/gav_prompts; mkdir -p "$PD"
SRC=""; for d in data/ppbench "$HOME/ppbench"; do [[ -f "$d/tok32768.txt" ]] && SRC="$d/tok32768.txt" && break; done
[[ -n "$SRC" ]] || { echo "缺 tok32768.txt"; echo FAIL; exit 1; }
for n in $PROMPTS; do tr -s ' \n' '\n' < "$SRC" | grep -v '^$' | head -n "$n" | tr '\n' ' ' > "$PD/p$n.txt"; done
tsec() { specline "$1" | sed -E 's/.* in ([0-9.]+) s = .*/\1/'; }
ntok() { specline "$1" | sed -E 's/^.*: ([0-9]+) tokens in .*/\1/'; }
# pool NAME "γ列表" "样本列表" 取样本的命令模板 -> 打印每 γ 汇总 tok/s，存到 POOL[γ]
declare -A POOL
pool() {
  local nm=$1 gs=$2 samples=$3 kind=$4 G x L tt ts
  for G in $gs; do
    tt=0; ts=0; local row=""
    for x in $samples; do
      L=gav_${nm}_${x}_g$G
      if [[ $kind == greedy ]]; then run $L "$BIN" "$PD/p$x.txt" 512 "$G" || { bad "$L 运行失败"; continue; }
      else run $L "$BIN" 8k 512 "$G" "QWENOX_SPEC_SAMPLE=$SMP,$x" || { bad "$L 运行失败"; continue; }; fi
      tt=$((tt + $(ntok $L))); ts=$(awk "BEGIN{print $ts+$(tsec $L)}")
      row+=" $(tps $L)"
    done
    POOL[$nm$G]=$(awk -v t="$tt" -v s="$ts" 'BEGIN{printf "%.2f", (s > 0 ? t / s : 0)}')
    printf "  %-6s γ=%-2s 汇总 %6s tok/s | 各样本:%s\n" "$nm" "$([[ $G == 0 ]] && echo 自适应 || echo $G)" "${POOL[$nm$G]}" "$row"
  done
}
pool greedy "4 7 0" "$PROMPTS" greedy
pool smp "3 4 7 0" "$SEEDS" smp
for x in $PROMPTS; do echo "    greedy p$x 自适应: $(adline gav_greedy_${x}_g0)"; done
for x in $SEEDS; do echo "    smp s$x 自适应: $(adline gav_smp_${x}_g0)"; done
grep -aL 'gamma-adapt:' logs/gav_greedy_*_g0.log logs/gav_smp_*_g0.log | grep -q . && bad "有自适应运行没打印 gamma-adapt"
chk() {  # chk NAME 自适应 基准 基准名
  local d; d=$(awk "BEGIN{printf \"%+.1f\", ($2/$3-1)*100}")
  awk "BEGIN{exit !($2 >= 0.98*$3)}" && ok "$1 自适应 $2 vs $4 $3 tok/s（$d%）" || bad "$1 自适应 $2 比 $4 $3 慢 >2%（$d%）"
}
chk greedy "${POOL[greedy0]}" "${POOL[greedy4]}" "旧默认 γ=4"
chk 采样 "${POOL[smp0]}" "${POOL[smp3]}" "旧默认 γ=3"
if [[ "${SKIP_64K:-0}" != 1 ]]; then
  echo "  -- 64K（重复 prompt，接受率差的场景；单样本，容差 5%）"
  if run gav_64k_g4 "$BIN" 64k 512 4 && run gav_64k_g7 "$BIN" 64k 512 7 && run gav_64k_g0 "$BIN" 64k 512 0; then
    echo "    γ=4 $(tps gav_64k_g4)  γ=7 $(tps gav_64k_g7)  自适应 $(tps gav_64k_g0) tok/s | $(adline gav_64k_g0)"
    awk "BEGIN{exit !($(tps gav_64k_g0) >= 0.95*$(tps gav_64k_g4))}" && ok "64K 自适应不明显慢于 γ=4" || bad "64K 自适应比 γ=4 慢 >5%"
  else
    bad "64K 运行失败"
  fi
fi

echo "-- 3. 自适应下采样分布自检（QWENOX_SMS_CHECK=1）"
for spec in "mtp:0" "chain:1"; do
  IFS=: read nm c <<<"$spec"
  L=gav_chk_$nm
  if run $L "$BIN" 8k 256 0 QWENOX_SPEC_CHAIN=$c QWENOX_SMS_CHECK=1 "QWENOX_SPEC_SAMPLE=1.0,20,0.95,7"; then
    last=$(grep -ah '^\[sms-check\] n=' "logs/$L.log" | tail -1)
    nmis=$(grep -ac 'MISMATCH' "logs/$L.log")
    n=$(sed -E 's/.* n=([0-9]+) .*/\1/' <<<"$last")
    echo "  $nm: ${last:-无 sms-check 输出} | $(adline $L)"
    if [[ -n "$last" ]] && (( n >= 64 && nmis == 0 )) && grep -q ' bad=0 ' <<<"$last"; then
      ok "$nm 稀疏分布 = dense 分布"
    else
      bad "$nm 自检未通过（MISMATCH $nmis 行）"
    fi
    [[ -n "$(adline $L)" ]] || bad "$nm 自适应没有生效"
  else
    bad "$nm 运行失败"
  fi
done

echo "== 结束 $(date '+%T')"
(( fail )) && { echo FAIL; exit 1; }
echo PASS
