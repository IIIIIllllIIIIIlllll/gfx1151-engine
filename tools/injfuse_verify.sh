#!/usr/bin/env bash
# injfuse_verify.sh — 一键验证 HC inject 融合（QWENOX_HC_INJ_FUSE，非逐 bit 等价，KLD 把关）
#   把 N=4 的 inject GEMM（Rhat · Winj）折进上一步的 scatter_norm：kernel 写 Rhat 的同时
#   累加 4 个 branch 的部分和，下一层 inject 只需 k_inj_psum 把 4 份部分和加起来。
#   R / Rhat 逐 bit 不变，只有 w4 的求和顺序变了（fp32，误差 ~1e-8 × sum|terms|）。
#   1. 单元原型 tools/injfuse_proto.cu：R/Rhat 逐 bit、w4 误差、单 kernel 计时 → PASS
#   2. KLD（CHUNKS=8 MAXCTX=8192，基准 data/kld/bf16_c8192.kld，生产 env）：
#      关 vs 开，开 - 关 ≤ 0.0003（噪声 ±0.0001），same_top（百分数）降幅 ≤ 0.3 个百分点
#   3. 32K prefill 速度（pp.sh 同款 env，KVSNAP 关）：新旧交替各 2 次取最快，需快 ≥0.5%
# 用法: bash tools/injfuse_verify.sh     （约 12 分钟，结尾输出 PASS / FAIL）
#   BIN=build/qwenox-inj（默认）  SKIP_KLD=1 只测速度  SKIP_PP=1 只测 KLD
# 日志: logs/ijv_*.log、logs/kld_ijv_*.log，汇总 logs/injfuse_verify.out
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
BIN="${BIN:-build/qwenox-inj}"
CHUNK=8192
OUT=logs/injfuse_verify.out
mkdir -p logs build
exec > >(tee "$OUT") 2>&1
fail=0
bad() { echo "  FAIL: $*"; fail=1; }
ok() { echo "  OK: $*"; }
KREF=data/kld/bf16_c8192.kld
for f in "$BIN" "$KREF" data/qsa-oracle/32768.tokens \
         models/qwen38-flash-next-w4b.hgn models/qwen38-flash-next-w4b.overlay.hgn; do
  [[ -e "$f" ]] || { echo "缺文件: $f"; echo FAIL; exit 1; }
done
echo "== injfuse_verify  $(date '+%F %T')  BIN=$BIN"

echo "== 1. 单元原型（R/Rhat 逐 bit + w4 误差 + 计时）"
sed -n '/\[gr-injfuse-begin\]/,/\[gr-injfuse-end\]/p' src/gpu/parts/22_kernels_prefill.inc > tools/gr_injfuse_kernel.inc
if hipcc -O3 --offload-arch=gfx1151 -o build/injfuse_proto tools/injfuse_proto.cu >logs/ijv_proto_build.log 2>&1; then
  build/injfuse_proto 20 >logs/ijv_proto.log 2>&1
  grep -E 'FAIL|timing|fused T=|worst' logs/ijv_proto.log | sed 's/^/  /'
  if tail -1 logs/ijv_proto.log | grep -q 'INJFUSE PROTO: PASS'; then ok "原型 PASS"; else bad "原型 FAIL（见 logs/ijv_proto.log）"; fi
else
  bad "原型编译失败（见 logs/ijv_proto_build.log）"
fi

if [[ -z "${SKIP_KLD:-}" ]]; then
  echo "== 2. KLD：关 vs 开（每次约 2 分钟）"
  kv() { sed -n 's/.*mean_kld=\([0-9.]*\).*/\1/p' "logs/kld_ijv_$1.log" | tail -1; }
  st() { sed -n 's/.*same_top=\([0-9.]*\).*/\1/p' "logs/kld_ijv_$1.log" | tail -1; }
  CHUNKS=8 MAXCTX=8192 BIN="$BIN" bash tools/kld_engine.sh "$KREF" ijv_off QWENOX_HC_INJ_FUSE=0 | grep -E 'kld_summary|rc=' | sed 's/^/  /'
  CHUNKS=8 MAXCTX=8192 BIN="$BIN" bash tools/kld_engine.sh "$KREF" ijv_on | grep -E 'kld_summary|rc=' | sed 's/^/  /'
  k0=$(kv off); k1=$(kv on); s0=$(st off); s1=$(st on)
  if grep -q 'hc-inj-fuse\] on' logs/kld_ijv_off.log; then bad "关的那次也走了融合路径"; fi
  if ! grep -q 'hc-inj-fuse\] on' logs/kld_ijv_on.log; then bad "开的那次没走融合路径（生产 env 缺 QWENOX_GR_BF16 / 条件不满足？）"; fi
  if [[ -z "$k0" || -z "$k1" ]]; then
    bad "KLD 运行失败（见 logs/kld_ijv_off.log / kld_ijv_on.log）"
  else
    echo "  mean KLD: 关 $k0  开 $k1  (差 $(awk -v a="$k0" -v b="$k1" 'BEGIN{printf "%+.6f",b-a}'))   same_top: 关 $s0  开 $s1"
    if awk -v a="$k0" -v b="$k1" 'BEGIN{exit !(b <= a + 0.0003)}'; then ok "KLD 差 ≤ 0.0003"; else bad "KLD 变差超过 0.0003"; fi
    if awk -v a="$s0" -v b="$s1" 'BEGIN{exit !(b >= a - 0.3)}'; then ok "same_top 降幅 ≤ 0.3 个百分点"; else bad "same_top 掉了 >0.3 个百分点"; fi
  fi
fi

if [[ -z "${SKIP_PP:-}" ]]; then
  run() {  # label len extra-env...
    local label=$1 len=$2; shift 2
    env QWENOX_QSA_KV_BF16=1 QWENOX_QSA_WMMA=1 QWENOX_QSA_WMMA_BTV=1 \
        QWENOX_MOE_LT=1 QWENOX_MOE_LT_BF16=1 QWENOX_GR_BF16=1 \
        QWENOX_GDN_STREAM=1 QWENOX_GDN_WAVE=1 \
        QWENOX_PREFILL_CHUNK=$CHUNK QWENOX_GEMM_WMMA=1 QWENOX_GDN_FUSED=1 \
        QWENOX_INDEX_FUSED2=1 QWENOX_PP_MOE_OUT=1 QWENOX_INDEX_STREAM_SELECT=1 \
        QWENOX_KVSNAP=0 QWENOX_PROF=1 QWENOX_PHASE=1 "$@" \
        "$BIN" models/qwen38-flash-next-w4b.hgn models/qwen38-flash-next-w4b.overlay.hgn \
        --tokens-file "data/qsa-oracle/$len.tokens" --gen 1 --maxctx $((len + 8192)) \
        >"logs/ijv_$label.stdout" 2>"logs/ijv_$label.log"
    echo $?
  }
  LEN=32768
  NCHUNK=$(( (LEN + CHUNK - 1) / CHUNK ))
  echo "== 3. 32K prefill 速度（新旧交替各 2 次，每次约 40 秒）"
  rates() { grep -E 'prefill: ' "logs/ijv_$1.log" | tail -n "$NCHUNK" | sed 's/.* = \([0-9.]*\) tok\/s.*/\1/'; }
  secs() { grep -E 'prefill: ' "logs/ijv_$1.log" | tail -n "$NCHUNK" | sed 's/.* in \([0-9.]*\) s = .*/\1/' | awk '{s+=$1} END{printf "%.3f",s}'; }
  okrun=1
  for r in 1 2; do
    rc_o=$(run pp_old$r $LEN QWENOX_HC_INJ_FUSE=0)
    rc_n=$(run pp_new$r $LEN)
    for l in pp_old$r pp_new$r; do
      [[ $(rates $l | wc -l) == "$NCHUNK" ]] || okrun=0
    done
    [[ $rc_o == 0 && $rc_n == 0 ]] || okrun=0
  done
  grep -q 'hc-inj-fuse\] on' logs/ijv_pp_new1.log || bad "速度测试的新版没走融合路径"
  if [[ $okrun != 1 ]]; then
    bad "速度测试运行失败（见 logs/ijv_pp_*.log）"
  else
    echo "  chunk  旧1 tok/s  旧2 tok/s  新1 tok/s  新2 tok/s"
    paste <(seq 1 "$NCHUNK") <(rates pp_old1) <(rates pp_old2) <(rates pp_new1) <(rates pp_new2) |
      awk '{printf "  %5d  %9.1f  %9.1f  %9.1f  %9.1f\n",$1,$2,$3,$4,$5}'
    t_o=$(printf '%s\n' "$(secs pp_old1)" "$(secs pp_old2)" | sort -n | head -1)
    t_n=$(printf '%s\n' "$(secs pp_new1)" "$(secs pp_new2)" | sort -n | head -1)
    a_o=$(awk -v t="$t_o" -v n="$LEN" 'BEGIN{printf "%.1f",n/t}')
    a_n=$(awk -v t="$t_n" -v n="$LEN" 'BEGIN{printf "%.1f",n/t}')
    echo "  整体（各取最快）: 旧 ${t_o} s = ${a_o} tok/s   新 ${t_n} s = ${a_n} tok/s  ($(awk -v a="$a_o" -v b="$a_n" 'BEGIN{printf "%+.2f%%",(b/a-1)*100}'))"
    if awk -v a="$a_o" -v b="$a_n" 'BEGIN{exit !(b >= a*1.005)}'; then
      ok "整体提升 ≥0.5%"
    else
      bad "速度没达标（整体需 ≥ 旧 ×1.005）"
    fi
  fi
fi

echo "== 结束 $(date '+%F %T')"
[[ $fail == 0 ]] && echo PASS || echo FAIL
exit $fail
