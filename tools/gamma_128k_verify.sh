#!/usr/bin/env bash
# gamma_128k_verify.sh — 128K 真实（不重复）上下文下的 MTP 投机 decode：固定 γ vs 自适应 γ
#   greedy γ{4,7,自适应} + 采样 γ{3,4,7,自适应}（每个 seed），SPEC=512，生产 HQ 权重
#   打印 tok/s、commit/round、ms/round（看 verify 成本是否随上下文上涨）、各深度接受率、自适应 γ 分布
#   PASS：全部跑通、HQ、无 SVM 死锁；自适应 ≥ 0.95× 旧默认（greedy γ=4 / 采样 γ=3，单样本所以放宽到 5%）
# 用法: bash tools/gamma_128k_verify.sh （约 16 分钟，结尾 PASS / FAIL）
#   BIN=build/qwenox-engine  TOK=data/qsa-oracle/131072.tokens  SEEDS="1"  SPEC=512
# 注：tok65536.txt 是 32K 重复两遍，不能代表真实长上下文；131072.tokens 无重复 64-gram。
# 日志: logs/g128_*.log，汇总 logs/gamma_128k_verify.out
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
BIN="${BIN:-build/qwenox-engine}"
TOK="${TOK:-data/qsa-oracle/131072.tokens}"
SEEDS="${SEEDS:-1}"
NS="${SPEC:-512}"
OUT=logs/gamma_128k_verify.out
mkdir -p logs
exec > >(tee "$OUT") 2>&1
T0="$(date '+%Y-%m-%d %H:%M:%S')"
fail=0
bad() { echo "  FAIL: $*"; fail=1; }
ok() { echo "  OK: $*"; }
SMP="0.7,20,0.8"
[[ -x "$BIN" ]] || { echo "缺二进制: $BIN"; echo FAIL; exit 1; }
[[ -f "$TOK" ]] || { echo "缺 token 文件: $TOK"; echo FAIL; exit 1; }
echo "== gamma_128k_verify  $T0  BIN=$BIN TOK=$TOK ($(wc -w <"$TOK") tokens) SPEC=$NS"

# 防 amdgpu SVM 死锁：先把 GGUF 权重逐出 page cache（同 pp128k_bench.sh）
mapfile -t GF < <(bash start_gguf.sh --check 2>/dev/null | sed -n 's/^CMD //p' | tr ' ' '\n' | grep -E '\.gguf$')
(( ${#GF[@]} )) && python3 - "${GF[@]}" <<'PY'
import os, sys
for f in sys.argv[1:]:
    try:
        fd = os.open(f, os.O_RDONLY); os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED); os.close(fd)
    except OSError:
        pass
PY
svm_check() {
  if journalctl -k --since "$T0" --no-pager 2>/dev/null | grep -q svm_range_cpu_invalidate_pagetables; then
    echo "!! amdgpu SVM 死锁（svm_range_cpu_invalidate_pagetables），中止，需重启"; echo FAIL; exit 1
  fi
}

specline() { grep -ahE '^spec(-sample)?: |^chain(-sample)?: [0-9]+ tokens' "logs/$1.log" | tail -1; }
tps() { specline "$1" | sed -E 's/.* = ([0-9.]+) tok\/s.*/\1/'; }
cpr() { specline "$1" | sed -E 's/.*commit\/round=([0-9.]+).*/\1/'; }
msr() { specline "$1" | sed -E 's/.* in ([0-9.]+) s .*rounds=([0-9]+).*/\1 \2/' | awk '{printf "%.1f", $1 * 1000 / $2}'; }
dacc() { grep -ah 'depth acc:' "logs/$1.log" | tail -1 | sed 's/^.*depth acc: //'; }
adline() { grep -ah 'gamma-adapt:' "logs/$1.log" | tail -1 | sed 's/^.*gamma-adapt: //'; }
declare -A R
run() {  # run LABEL GAMMA [K=V...]
  local L=$1 G=$2; shift 2
  echo "  .. $L（$(date +%T)）"
  SPEC=$NS GAMMA=$G BIN=$BIN bash tools/pp_prod.sh "$TOK" "$L" "$@" > "logs/$L.stdout" 2>&1
  local rc=$?
  svm_check
  if (( rc )) || [[ -z "$(specline $L)" ]]; then
    bad "$L 运行失败 rc=$rc"; grep -ah 'rror\|abort\|exception\|HSA' "logs/$L.log" "logs/$L.stdout" 2>/dev/null | head -3
    return 1
  fi
  grep -aq 'dense: 8-bit' "logs/$L.log" || { echo "  $L: 不是 HQ 权重，停止"; echo FAIL; exit 1; }
  R[$L]=$(tps $L)
  printf "     %-16s %6s tok/s  commit/round %-5s  %6s ms/round | acc %s\n" "$L" "${R[$L]}" "$(cpr $L)" "$(msr $L)" "$(dacc $L)"
  [[ $G == 0 ]] && echo "     └ 自适应: $(adline $L)"
  return 0
}
chk() {  # chk NAME 自适应标签 基准标签
  local a=${R[$2]:-} b=${R[$3]:-}
  [[ -n $a && -n $b ]] || { bad "$1 缺数据"; return; }
  local d; d=$(awk -v a="$a" -v b="$b" 'BEGIN{printf "%+.1f", (a / b - 1) * 100}')
  if awk -v a="$a" -v b="$b" 'BEGIN{exit !(a >= 0.95 * b)}'; then ok "$1 自适应 $a vs $3 $b tok/s（$d%）"
  else bad "$1 自适应 $a 比 $3 $b 慢 >5%（$d%）"; fi
}

echo "-- greedy"
for G in 4 7 0; do run g128_greedy_g$G $G; done
chk greedy g128_greedy_g0 g128_greedy_g4
for s in $SEEDS; do
  echo "-- 采样 seed $s（$SMP）"
  for G in 3 4 7 0; do run g128_smp${s}_g$G $G "QWENOX_SPEC_SAMPLE=$SMP,$s"; done
  chk "采样 s$s" g128_smp${s}_g0 g128_smp${s}_g3
done

echo "== 结束 $(date '+%T')"
(( fail )) && { echo FAIL; exit 1; }
echo PASS
