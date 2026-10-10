#!/usr/bin/env bash
# hcmix_verify.sh — 一键验证 #51 HC 门控融合（gr_mix_b: up GEMM + k_gr_combine_b_hc_bf16
# 折进 k_gemm_wmma Epi=1，40_model.inc hc_up_fused）
#   1. 原型 tools/hcmix_proto.cu（kernel 从当前 22_kernels_prefill.inc 现抽）：
#      融合 vs 生产两 kernel，x fp32 + bf16 逐 bit 一致（含 P 尾块）+ kernel 速度
#   2. 端到端 --ppl：同一个二进制，旧路径 (QWENOX_HC_FUSE=0) vs 新路径（默认），
#      2051（P 尾块）和 32K 两个长度的 ppl_token 行 + ppl_summary 必须逐字节一致
#   3. 32K prefill 速度（pp.sh 同款 env，KVSNAP 关）：新旧各跑一次，整体需快 ≥2%
# 用法: bash tools/hcmix_verify.sh        （约 6 分钟，结尾输出 PASS / FAIL）
#   BIN=build/qwenox-hc（默认）  SKIP_PPL=1 只测速度
# 日志: logs/hcv_*.log（stderr）/ *.stdout（ppl 行单独存），汇总 logs/hcmix_verify.out
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
BIN="${BIN:-build/qwenox-hc}"
CHUNK=8192
OUT=logs/hcmix_verify.out
mkdir -p logs
exec > >(tee "$OUT") 2>&1
fail=0
bad() { echo "  FAIL: $*"; fail=1; }
ok() { echo "  OK: $*"; }
OLD_ENV="QWENOX_HC_FUSE=0"

for f in "$BIN" data/qsa-oracle/2051.tokens data/qsa-oracle/32768.tokens \
         models/qwen38-flash-next-w4b.hgn models/qwen38-flash-next-w4b.overlay.hgn; do
  [[ -e "$f" ]] || { echo "缺文件: $f"; echo FAIL; exit 1; }
done
echo "== hcmix_verify  $(date '+%F %T')  BIN=$BIN"

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
      >"logs/hcv_$label.stdout" 2>"logs/hcv_$label.log"
  echo $?
}

echo "== 1. 原型（融合 vs 生产 kernel 逐 bit 对比 + 速度）"
sed -n '/\[gemm-wmma-begin\]/,/\[gemm-wmma-end\]/p' src/gpu/parts/22_kernels_prefill.inc \
  > tools/gemm_wmma_kernel.inc
if /opt/rocm/bin/hipcc -O3 --offload-arch=gfx1151 tools/hcmix_proto.cu \
     -o build/hcmix_proto >logs/hcv_proto_build.log 2>&1; then
  build/hcmix_proto >logs/hcv_proto.log 2>&1
  grep -E "P=|ms|DIFF|mismatch" logs/hcv_proto.log | head -20 | sed 's/^/  /'
  if tail -n1 logs/hcv_proto.log | grep -q "HCMIX PROTO: PASS"; then ok "原型逐 bit 一致"
  else bad "原型（见 logs/hcv_proto.log）"; fi
else
  bad "原型编译失败（见 logs/hcv_proto_build.log）"
fi

if [[ -z "${SKIP_PPL:-}" ]]; then
  for len in 2051 32768; do
    echo "== 2. ${len} --ppl 端到端逐字节对比"
    rc_o=$(run ppl${len}_old "$len" ppl $OLD_ENV)
    rc_n=$(run ppl${len}_new "$len" ppl)
    grep '^ppl_' "logs/hcv_ppl${len}_old.stdout" >"logs/hcv_ppl${len}_old.txt"
    grep '^ppl_' "logs/hcv_ppl${len}_new.stdout" >"logs/hcv_ppl${len}_new.txt"
    n_o=$(grep -c $'^ppl_token\t[0-9]' "logs/hcv_ppl${len}_old.txt")
    s_o=$(grep '^ppl_summary' "logs/hcv_ppl${len}_old.txt")
    s_n=$(grep '^ppl_summary' "logs/hcv_ppl${len}_new.txt")
    echo "  旧: rc=$rc_o  $s_o"
    echo "  新: rc=$rc_n  $s_n"
    if [[ $rc_o != 0 || $rc_n != 0 || -z "$s_o" || -z "$s_n" ]]; then
      bad "ppl 运行失败（见 logs/hcv_ppl${len}_*.log）"
    elif [[ $n_o -lt $((len - 2)) ]]; then
      bad "ppl_token 行数不对: $n_o"
    elif cmp -s "logs/hcv_ppl${len}_old.txt" "logs/hcv_ppl${len}_new.txt"; then
      ok "$n_o 个 ppl_token + summary 逐字节一致"
    else
      d=$(diff "logs/hcv_ppl${len}_old.txt" "logs/hcv_ppl${len}_new.txt" | grep -c '^<')
      bad "ppl 输出不一致: $d 行不同，第一处: $(diff "logs/hcv_ppl${len}_old.txt" "logs/hcv_ppl${len}_new.txt" | sed -n 2p)"
    fi
  done
fi

LEN=32768
NCHUNK=$(( (LEN + CHUNK - 1) / CHUNK ))
echo "== 3. 32K prefill 速度（pp.sh 同款 env，每次约 40 秒）"
rates() { grep -E 'prefill: ' "logs/hcv_$1.log" | tail -n "$NCHUNK" | sed 's/.* = \([0-9.]*\) tok\/s.*/\1/'; }
secs() { grep -E 'prefill: ' "logs/hcv_$1.log" | tail -n "$NCHUNK" | sed 's/.* in \([0-9.]*\) s = .*/\1/'; }
rc_o=$(run pp_old $LEN pp $OLD_ENV)
rc_n=$(run pp_new $LEN pp)
if [[ $rc_o != 0 || $rc_n != 0 || $(rates pp_old | wc -l) != "$NCHUNK" || $(rates pp_new | wc -l) != "$NCHUNK" ]]; then
  bad "速度测试运行失败（rc $rc_o/$rc_n，见 logs/hcv_pp_*.log）"
else
  echo "  chunk  旧 tok/s  新 tok/s"
  paste <(seq 1 "$NCHUNK") <(rates pp_old) <(rates pp_new) | awk '{printf "  %5d  %8.1f  %8.1f\n",$1,$2,$3}'
  t_o=$(secs pp_old | awk '{s+=$1} END{printf "%.3f",s}')
  t_n=$(secs pp_new | awk '{s+=$1} END{printf "%.3f",s}')
  a_o=$(awk -v t="$t_o" -v n="$LEN" 'BEGIN{printf "%.1f",n/t}')
  a_n=$(awk -v t="$t_n" -v n="$LEN" 'BEGIN{printf "%.1f",n/t}')
  echo "  整体: 旧 ${t_o} s = ${a_o} tok/s   新 ${t_n} s = ${a_n} tok/s"
  if awk -v a="$a_o" -v b="$a_n" 'BEGIN{exit !(b >= a*1.02)}'; then
    ok "整体提升 ≥2%"
  else
    bad "速度没达标（整体需 ≥ 旧 ×1.02）"
  fi
fi

echo "== 结束 $(date '+%F %T')"
[[ $fail == 0 ]] && echo PASS || echo FAIL
exit $fail
