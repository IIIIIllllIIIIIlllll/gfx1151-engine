#!/usr/bin/env bash
# index_fast_verify.sh — 一键验证 indexer 快速打分 + 两遍精确选块（09_kernels_index.inc）
#   1. 单元测试 build/index_fast_test：新旧 kernel 逐 bit 对比（打分全矩阵 memcmp，
#      选块 vs stream/rs，含平分极多的合成行和 fallback 路径）+ 128K/256K 速度
#   2. 128K 端到端 --ppl：同一个二进制，旧路径 (QWENOX_INDEX_SCORE64=0 QWENOX_INDEX_SEL2P=0)
#      vs 新路径（默认），131071 个 ppl_token 行 + ppl_summary 必须逐字节一致
#   3. 128K prefill 速度（pp.sh 同款 env，KVSNAP 关）：新旧各跑一次，比最后一个 chunk
#      和整体平均 tok/s
# 用法: bash tools/index_fast_verify.sh        （约 12 分钟，结尾输出 PASS / FAIL）
#   BIN=build/qwenox-idx（默认）  SKIP_PPL=1 只测速度
# 日志: logs/idxv_*.log（stderr）/ *.stdout（ppl 行单独存，避免与 stderr 交错），汇总 logs/index_fast_verify.out
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
BIN="${BIN:-build/qwenox-idx}"
LEN=131072
TOK="data/qsa-oracle/${LEN}.tokens"
MAXCTX=$((LEN + 8192))
CHUNK=8192
NCHUNK=$(( (LEN + CHUNK - 1) / CHUNK ))
OUT=logs/index_fast_verify.out
mkdir -p logs
exec > >(tee "$OUT") 2>&1
fail=0
bad() { echo "  FAIL: $*"; fail=1; }
ok() { echo "  OK: $*"; }
OLD_ENV="QWENOX_INDEX_SCORE64=0 QWENOX_INDEX_SEL2P=0"

for f in "$BIN" "$TOK" models/qwen38-flash-next-w4b.hgn models/qwen38-flash-next-w4b.overlay.hgn; do
  [[ -e "$f" ]] || { echo "缺文件: $f"; echo FAIL; exit 1; }
done
echo "== index_fast_verify  $(date '+%F %T')  BIN=$BIN"

run() {  # label mode(ppl|pp) extra-env...
  local label=$1 mode=$2; shift 2
  local args=(--gen 1)
  [[ $mode == ppl ]] && args=(--ppl)
  env QWENOX_QSA_KV_BF16=1 QWENOX_QSA_WMMA=1 QWENOX_QSA_WMMA_BTV=1 \
      QWENOX_MOE_LT=1 QWENOX_MOE_LT_BF16=1 QWENOX_GR_BF16=1 \
      QWENOX_GDN_STREAM=1 QWENOX_GDN_WAVE=1 \
      QWENOX_PREFILL_CHUNK=$CHUNK QWENOX_GEMM_WMMA=1 QWENOX_GDN_FUSED=1 \
      QWENOX_INDEX_FUSED2=1 QWENOX_PP_MOE_OUT=1 QWENOX_INDEX_STREAM_SELECT=1 \
      QWENOX_KVSNAP=0 QWENOX_PROF=1 QWENOX_PHASE=1 "$@" \
      "$BIN" models/qwen38-flash-next-w4b.hgn models/qwen38-flash-next-w4b.overlay.hgn \
      --tokens-file "$TOK" "${args[@]}" --maxctx "$MAXCTX" \
      >"logs/idxv_$label.stdout" 2>"logs/idxv_$label.log"
  echo $?
}

echo "== 1. 单元测试（逐 bit 对比 + kernel 速度）"
if /opt/rocm/bin/hipcc -O3 --offload-arch=gfx1151 -I src/gpu tools/index_fast_test.cu \
     -o build/index_fast_test >logs/idxv_unit_build.log 2>&1; then
  build/index_fast_test >logs/idxv_unit.log 2>&1
  grep -E "DIFF|TFLOPS|select old" logs/idxv_unit.log | sed 's/^/  /'
  if [[ "$(tail -n1 logs/idxv_unit.log)" == PASS ]]; then ok "单元测试全部一致"
  else bad "单元测试（见 logs/idxv_unit.log）"; fi
else
  bad "单元测试编译失败（见 logs/idxv_unit_build.log）"
fi

if [[ -z "${SKIP_PPL:-}" ]]; then
  echo "== 2. 128K --ppl 端到端逐字节对比（每次约 3 分钟）"
  rc_o=$(run ppl_old ppl $OLD_ENV)
  rc_n=$(run ppl_new ppl)
  grep '^ppl_' logs/idxv_ppl_old.stdout >logs/idxv_ppl_old.txt
  grep '^ppl_' logs/idxv_ppl_new.stdout >logs/idxv_ppl_new.txt
  n_o=$(grep -c $'^ppl_token\t[0-9]' logs/idxv_ppl_old.txt)
  s_o=$(grep '^ppl_summary' logs/idxv_ppl_old.txt)
  s_n=$(grep '^ppl_summary' logs/idxv_ppl_new.txt)
  echo "  旧: rc=$rc_o  $s_o"
  echo "  新: rc=$rc_n  $s_n"
  if [[ $rc_o != 0 || $rc_n != 0 || -z "$s_o" || -z "$s_n" ]]; then
    bad "ppl 运行失败（见 logs/idxv_ppl_*.log）"
  elif [[ $n_o -lt $((LEN - 2)) ]]; then
    bad "ppl_token 行数不对: $n_o"
  elif cmp -s logs/idxv_ppl_old.txt logs/idxv_ppl_new.txt; then
    ok "$n_o 个 ppl_token + summary 逐字节一致"
  else
    d=$(diff logs/idxv_ppl_old.txt logs/idxv_ppl_new.txt | grep -c '^<')
    bad "ppl 输出不一致: $d 行不同，第一处: $(diff logs/idxv_ppl_old.txt logs/idxv_ppl_new.txt | sed -n 2p)"
  fi
fi

echo "== 3. 128K prefill 速度（pp.sh 同款 env，每次约 2.5 分钟）"
rates() { grep -E 'prefill: ' "logs/idxv_$1.log" | tail -n "$NCHUNK" | sed 's/.* = \([0-9.]*\) tok\/s.*/\1/'; }
secs() { grep -E 'prefill: ' "logs/idxv_$1.log" | tail -n "$NCHUNK" | sed 's/.* in \([0-9.]*\) s = .*/\1/'; }
rc_o=$(run pp_old pp $OLD_ENV)
rc_n=$(run pp_new pp)
if [[ $rc_o != 0 || $rc_n != 0 || $(rates pp_old | wc -l) != "$NCHUNK" || $(rates pp_new | wc -l) != "$NCHUNK" ]]; then
  bad "速度测试运行失败（rc $rc_o/$rc_n，见 logs/idxv_pp_*.log）"
else
  echo "  chunk  旧 tok/s  新 tok/s"
  paste <(seq 1 "$NCHUNK") <(rates pp_old) <(rates pp_new) | awk '{printf "  %5d  %8.1f  %8.1f\n",$1,$2,$3}'
  t_o=$(secs pp_old | awk '{s+=$1} END{printf "%.2f",s}')
  t_n=$(secs pp_new | awk '{s+=$1} END{printf "%.2f",s}')
  l_o=$(rates pp_old | tail -n1); l_n=$(rates pp_new | tail -n1)
  a_o=$(awk -v t="$t_o" -v n="$LEN" 'BEGIN{printf "%.1f",n/t}')
  a_n=$(awk -v t="$t_n" -v n="$LEN" 'BEGIN{printf "%.1f",n/t}')
  echo "  整体: 旧 ${t_o} s = ${a_o} tok/s   新 ${t_n} s = ${a_n} tok/s"
  echo "  最后一个 chunk: 旧 $l_o   新 $l_n tok/s"
  if awk -v a="$l_o" -v b="$l_n" 'BEGIN{exit !(b >= a*1.03)}' &&
     awk -v a="$a_o" -v b="$a_n" 'BEGIN{exit !(b >= a)}'; then
    ok "最后 chunk 提升 ≥3%，整体不慢"
  else
    bad "速度没达标（最后 chunk 需 ≥ 旧 ×1.03，整体需 ≥ 旧）"
  fi
fi

echo "== 结束 $(date '+%F %T')"
[[ $fail == 0 ]] && echo PASS || echo FAIL
exit $fail
