#!/usr/bin/env bash
# convl2_verify.sh — 一键验证 GDN conv+silu 与 q/k L2 norm 融合（k_gdn_conv_l2n_b，
# 25_kernels_gdn.inc；40_model.inc gdn_b 调用处，替掉 k_gdn_conv_b + k_l2norm_qk_b）
#   1. 原型 tools/convl2_proto.cu（kernel 从当前 25_kernels_gdn.inc 现抽）：
#      融合 vs 生产两 kernel，P x 10240 输出逐 bit 一致（含 P=1/3 尾块、极端行）+ 速度
#   2. 端到端 --ppl：同一个二进制，旧路径 (QWENOX_GDN_CONVL2=0) vs 新路径（默认），
#      2051 和 32K 两个长度的 ppl_token 行 + ppl_summary 必须逐字节一致
#   3. 32K prefill 速度（pp.sh 同款 env，KVSNAP 关）：新旧交替各 2 次取最快，
#      整体需快 ≥0.3%（预期 ~0.8%：每 8K chunk 省 ~45 ms）
# 用法: bash tools/convl2_verify.sh        （约 7 分钟，结尾输出 PASS / FAIL）
#   BIN=build/qwenox-cl2（默认）  SKIP_PPL=1 只测速度
# 日志: logs/clv_*.log（stderr）/ *.stdout（ppl 行单独存），汇总 logs/convl2_verify.out
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
BIN="${BIN:-build/qwenox-cl2}"
CHUNK=8192
OUT=logs/convl2_verify.out
mkdir -p logs
exec > >(tee "$OUT") 2>&1
fail=0
bad() { echo "  FAIL: $*"; fail=1; }
ok() { echo "  OK: $*"; }
OLD_ENV="QWENOX_GDN_CONVL2=0"

for f in "$BIN" data/qsa-oracle/2051.tokens data/qsa-oracle/32768.tokens \
         models/qwen38-flash-next-w4b.hgn models/qwen38-flash-next-w4b.overlay.hgn; do
  [[ -e "$f" ]] || { echo "缺文件: $f"; echo FAIL; exit 1; }
done
echo "== convl2_verify  $(date '+%F %T')  BIN=$BIN"

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
      >"logs/clv_$label.stdout" 2>"logs/clv_$label.log"
  echo $?
}

echo "== 1. 原型（融合 vs 生产 kernel 逐 bit 对比 + 速度）"
sed -n '/\[gdn-convl2-begin\]/,/\[gdn-convl2-end\]/p' src/gpu/parts/25_kernels_gdn.inc \
  > tools/gdn_convl2_kernel.inc
if /opt/rocm/bin/hipcc -O3 --offload-arch=gfx1151 tools/convl2_proto.cu \
     -o build/convl2_proto >logs/clv_proto_build.log 2>&1; then
  build/convl2_proto 50 >logs/clv_proto.log 2>&1
  grep -E "P=" logs/clv_proto.log | head -20 | sed 's/^/  /'
  if tail -n1 logs/clv_proto.log | grep -q "CONVL2 PROTO: PASS"; then ok "原型逐 bit 一致"
  else bad "原型（见 logs/clv_proto.log）"; fi
else
  bad "原型编译失败（见 logs/clv_proto_build.log）"
fi

if [[ -z "${SKIP_PPL:-}" ]]; then
  for len in 2051 32768; do
    echo "== 2. ${len} --ppl 端到端逐字节对比"
    rc_o=$(run ppl${len}_old "$len" ppl $OLD_ENV)
    rc_n=$(run ppl${len}_new "$len" ppl)
    grep '^ppl_' "logs/clv_ppl${len}_old.stdout" >"logs/clv_ppl${len}_old.txt"
    grep '^ppl_' "logs/clv_ppl${len}_new.stdout" >"logs/clv_ppl${len}_new.txt"
    n_o=$(grep -c $'^ppl_token\t[0-9]' "logs/clv_ppl${len}_old.txt")
    s_o=$(grep '^ppl_summary' "logs/clv_ppl${len}_old.txt")
    s_n=$(grep '^ppl_summary' "logs/clv_ppl${len}_new.txt")
    echo "  旧: rc=$rc_o  $s_o"
    echo "  新: rc=$rc_n  $s_n"
    if [[ $rc_o != 0 || $rc_n != 0 || -z "$s_o" || -z "$s_n" ]]; then
      bad "ppl 运行失败（见 logs/clv_ppl${len}_*.log）"
    elif [[ $n_o -lt $((len - 2)) ]]; then
      bad "ppl_token 行数不对: $n_o"
    elif cmp -s "logs/clv_ppl${len}_old.txt" "logs/clv_ppl${len}_new.txt"; then
      ok "$n_o 个 ppl_token + summary 逐字节一致"
    else
      d=$(diff "logs/clv_ppl${len}_old.txt" "logs/clv_ppl${len}_new.txt" | grep -c '^<')
      bad "ppl 输出不一致: $d 行不同，第一处: $(diff "logs/clv_ppl${len}_old.txt" "logs/clv_ppl${len}_new.txt" | sed -n 2p)"
    fi
  done
fi

LEN=32768
NCHUNK=$(( (LEN + CHUNK - 1) / CHUNK ))
echo "== 3. 32K prefill 速度（pp.sh 同款 env，新旧交替各 2 次，每次约 40 秒）"
rates() { grep -E 'prefill: ' "logs/clv_$1.log" | tail -n "$NCHUNK" | sed 's/.* = \([0-9.]*\) tok\/s.*/\1/'; }
secs() { grep -E 'prefill: ' "logs/clv_$1.log" | tail -n "$NCHUNK" | sed 's/.* in \([0-9.]*\) s = .*/\1/' | awk '{s+=$1} END{printf "%.3f",s}'; }
okrun=1
for r in 1 2; do
  rc_o=$(run pp_old$r $LEN pp $OLD_ENV)
  rc_n=$(run pp_new$r $LEN pp)
  for l in pp_old$r pp_new$r; do
    [[ $(rates $l | wc -l) == "$NCHUNK" ]] || okrun=0
  done
  [[ $rc_o == 0 && $rc_n == 0 ]] || okrun=0
done
if [[ $okrun != 1 ]]; then
  bad "速度测试运行失败（见 logs/clv_pp_*.log）"
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
