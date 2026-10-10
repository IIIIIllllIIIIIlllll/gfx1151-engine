#!/usr/bin/env bash
# epi_verify.sh — 一键验证 prefill 小 kernel 的 epilogue 融合（逐 bit 等价）
#   QWENOX_MOE_SG_FUSE   共享专家加法 k_axpy_sg 折进路由 reduce（k_moe_reduce_pw_sg）
#   QWENOX_QSA_GATE_BF16 QSA 输出门控 + o_proj 输入 bf16 转换合一（k_sigmoid_gate_bf16_v4，
#                      门控直接读 d_qgb，qsplit 不再拷 gs）
#   1. 2051 --ppl：全关 vs 每个开关单开 vs 全开，ppl_token + summary 必须逐字节一致
#   2. 32K --ppl：全关 vs 全开逐字节一致
#   3. 32K prefill 速度（pp.sh 同款 env，KVSNAP 关）：新旧交替各 2 次取最快，需快 ≥0.3%
# 用法: bash tools/epi_verify.sh        （约 6 分钟，结尾输出 PASS / FAIL）
#   BIN=build/qwenox-epi（默认）  SKIP_PPL=1 只测速度
# 日志: logs/epv_*.log（stderr）/ *.stdout（ppl 行单独存），汇总 logs/epi_verify.out
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
BIN="${BIN:-build/qwenox-epi}"
CHUNK=8192
OUT=logs/epi_verify.out
mkdir -p logs
exec > >(tee "$OUT") 2>&1
fail=0
bad() { echo "  FAIL: $*"; fail=1; }
ok() { echo "  OK: $*"; }
FLAGS=(QWENOX_MOE_SG_FUSE QWENOX_QSA_GATE_BF16)
OLD_ENV=()
for f in "${FLAGS[@]}"; do OLD_ENV+=("$f=0"); done

for f in "$BIN" data/qsa-oracle/2051.tokens data/qsa-oracle/32768.tokens \
         models/qwen38-flash-next-w4b.hgn models/qwen38-flash-next-w4b.overlay.hgn; do
  [[ -e "$f" ]] || { echo "缺文件: $f"; echo FAIL; exit 1; }
done
echo "== epi_verify  $(date '+%F %T')  BIN=$BIN"

run() {  # label len mode(ppl|pp) extra-env...
  local label=$1 len=$2 mode=$3; shift 3
  local args=(--gen 1)
  [[ $mode == ppl ]] && args=(--ppl)
  env QWENOX_QSA_KV_BF16=1 QWENOX_QSA_WMMA=1 QWENOX_QSA_WMMA_BTV=1 \
      QWENOX_MOE_LT=1 QWENOX_MOE_LT_BF16=1 QWENOX_GR_BF16=1 \
      QWENOX_GDN_STREAM=1 QWENOX_GDN_WAVE=1 \
      QWENOX_PREFILL_CHUNK=$CHUNK QWENOX_GEMM_WMMA=1 QWENOX_GDN_FUSED=1 \
      QWENOX_INDEX_FUSED2=1 QWENOX_PP_MOE_OUT=1 QWENOX_INDEX_STREAM_SELECT=1 \
      QWENOX_KVSNAP=0 QWENOX_PROF=1 QWENOX_PHASE=1 "$@" \
      "$BIN" models/qwen38-flash-next-w4b.hgn models/qwen38-flash-next-w4b.overlay.hgn \
      --tokens-file "data/qsa-oracle/$len.tokens" "${args[@]}" --maxctx $((len + 8192)) \
      >"logs/epv_$label.stdout" 2>"logs/epv_$label.log"
  echo $?
}

pplcmp() {  # len label-a label-b
  local len=$1 a=$2 b=$3
  local sa sb
  sa=$(grep '^ppl_summary' "logs/epv_$a.txt"); sb=$(grep '^ppl_summary' "logs/epv_$b.txt")
  echo "  $a: $sa"
  echo "  $b: $sb"
  n=$(grep -c $'^ppl_token\t[0-9]' "logs/epv_$a.txt")
  if [[ -z "$sa" || -z "$sb" ]]; then bad "ppl 运行失败（见 logs/epv_$a.log / $b.log）"
  elif [[ $n -lt $((len - 2)) ]]; then bad "ppl_token 行数不对: $n"
  elif cmp -s "logs/epv_$a.txt" "logs/epv_$b.txt"; then ok "$b vs $a: $n 个 ppl_token + summary 逐字节一致"
  else bad "$b vs $a 不一致，第一处: $(diff "logs/epv_$a.txt" "logs/epv_$b.txt" | sed -n 2p)"; fi
}
pplrun() {  # label len env...
  local label=$1 len=$2; shift 2
  run "$label" "$len" ppl "$@" >/dev/null
  grep '^ppl_' "logs/epv_$label.stdout" >"logs/epv_$label.txt"
}
if [[ -z "${SKIP_PPL:-}" ]]; then
  echo "== 1. 2051 --ppl：全关 / 单开 / 全开"
  pplrun p2051_off 2051 "${OLD_ENV[@]}"
  for f in "${FLAGS[@]}"; do
    e=()
    for g in "${FLAGS[@]}"; do [[ $g == "$f" ]] || e+=("$g=0"); done
    pplrun "p2051_$f" 2051 "${e[@]}"
    pplcmp 2051 p2051_off "p2051_$f"
  done
  pplrun p2051_on 2051
  pplcmp 2051 p2051_off p2051_on
  echo "== 2. 32K --ppl：全关 vs 全开"
  pplrun p32k_off 32768 "${OLD_ENV[@]}"
  pplrun p32k_on 32768
  pplcmp 32768 p32k_off p32k_on
fi

LEN=32768
NCHUNK=$(( (LEN + CHUNK - 1) / CHUNK ))
echo "== 3. 32K prefill 速度（pp.sh 同款 env，新旧交替各 2 次，每次约 40 秒）"
rates() { grep -E 'prefill: ' "logs/epv_$1.log" | tail -n "$NCHUNK" | sed 's/.* = \([0-9.]*\) tok\/s.*/\1/'; }
secs() { grep -E 'prefill: ' "logs/epv_$1.log" | tail -n "$NCHUNK" | sed 's/.* in \([0-9.]*\) s = .*/\1/' | awk '{s+=$1} END{printf "%.3f",s}'; }
okrun=1
for r in 1 2; do
  rc_o=$(run pp_old$r $LEN pp "${OLD_ENV[@]}")
  rc_n=$(run pp_new$r $LEN pp)
  for l in pp_old$r pp_new$r; do
    [[ $(rates $l | wc -l) == "$NCHUNK" ]] || okrun=0
  done
  [[ $rc_o == 0 && $rc_n == 0 ]] || okrun=0
done
if [[ $okrun != 1 ]]; then
  bad "速度测试运行失败（见 logs/epv_pp_*.log）"
else
  echo "  chunk  旧1 tok/s  旧2 tok/s  新1 tok/s  新2 tok/s"
  paste <(seq 1 "$NCHUNK") <(rates pp_old1) <(rates pp_old2) <(rates pp_new1) <(rates pp_new2) |
    awk '{printf "  %5d  %9.1f  %9.1f  %9.1f  %9.1f\n",$1,$2,$3,$4,$5}'
  t_o=$(printf '%s\n' "$(secs pp_old1)" "$(secs pp_old2)" | sort -n | head -1)
  t_n=$(printf '%s\n' "$(secs pp_new1)" "$(secs pp_new2)" | sort -n | head -1)
  a_o=$(awk -v t="$t_o" -v n="$LEN" 'BEGIN{printf "%.1f",n/t}')
  a_n=$(awk -v t="$t_n" -v n="$LEN" 'BEGIN{printf "%.1f",n/t}')
  echo "  整体（各取最快）: 旧 ${t_o} s = ${a_o} tok/s   新 ${t_n} s = ${a_n} tok/s  ($(awk -v a="$a_o" -v b="$a_n" 'BEGIN{printf "%+.2f%%",(b/a-1)*100}'))"
  if awk -v a="$a_o" -v b="$a_n" 'BEGIN{exit !(b >= a*1.003)}'; then
    ok "整体提升 ≥0.3%"
  else
    bad "速度没达标（整体需 ≥ 旧 ×1.003）"
  fi
fi

echo "== 结束 $(date '+%F %T')"
[[ $fail == 0 ]] && echo PASS || echo FAIL
exit $fail
