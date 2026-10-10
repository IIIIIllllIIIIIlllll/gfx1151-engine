#!/usr/bin/env bash
# halogen 0.15 hgn v2 原生加载一键验证（前台跑，最后一行 PASS/FAIL）。
#   bash tools/v2_verify.sh
# 五步，每步一次模型加载（加载前先 fadvise 丢掉模型/基准文件的 page cache，
# 避开 09-25 满 page cache 下加载的 kfd svm 死锁）：
#   1 v2 prefill KLD（LUT WMMA 专家路径）对 BF16 基准；权重参数取自 start_hgn.sh
#     --check，同时验证 NGRAM_FILE 拼参（v2 = 主体 + 独立 n-gram 文件）
#   2 v2 小 batch 路径（QWENOX_MOE_NAIVE_MAX 拉满 = MTP verify 用的 naive gemv）
#   3 v2 decode 路径（QWENOX_NOPREFILLBATCH 逐 token forward：ht/q6g64 gemv、
#     moe_v2_1）与 prefill 的 --ppl 对比，两者应基本一致
#   4 v1 回归：w4b + overlay KLD 必须等于 09-25 基线
#   5 汇总各步 GTT / 进程 RSS 峰值（显存占用对比）
# 可覆盖：REF= V2= NG= W4B= OVL= V1_KLD= MAX_KLD= CHUNKS= BIN=
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
REF=${REF:-$HOME/Workspace/gfx-1151-kvsnap/data/kld/bf16_c512.kld}
V2=${V2:-models/qwen38-flash-next-v2.hgn}
NG=${NG:-models/qwen38-flash-next-ngram.hgn}
W4B=${W4B:-models/old/qwen38-flash-next-w4b.hgn}
OVL=${OVL:-models/old/qwen38-flash-next-w4b.overlay.hgn}
V1_KLD=${V1_KLD:-0.163078}  # 09-25 bench_hgn：w4b+overlay，64 chunk x 512
MAX_KLD=${MAX_KLD:-0.5}     # 只判"算对没算错"（坏 kernel 是 KLD>>1），不评质量
CHUNKS=${CHUNKS:-64}
BIN=${BIN:-build/qwenox-engine}
export BIN
GTT=$(ls /sys/class/drm/card*/device/mem_info_gtt_used 2>/dev/null | head -1)

FAILS=()
fail() { FAILS+=("$1"); echo "  FAIL: $1"; }
for f in "$REF" "$V2" "$NG" "$W4B" "$OVL" "$BIN"; do
  [[ -f $f ]] || { echo "缺文件：$f"; echo FAIL; exit 1; }
done
pgrep -x qwenox >/dev/null && { echo "已有 qwenox 在跑，先停掉"; echo FAIL; exit 1; }
mkdir -p logs

drop_cache() {
  python3 - "$REF" models/*.hgn models/old/*.hgn <<'PY'
import os, sys
for f in sys.argv[1:]:
    try:
        fd = os.open(f, os.O_RDONLY); os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED); os.close(fd)
    except OSError:
        pass
PY
}

# 后台采样：GTT 已用量与 qwenox 进程 RSS 的峰值（GiB），写到 $1
sampler() {
  local out=$1 g0 g=0 r=0 v pid
  g0=$(cat "$GTT" 2>/dev/null || echo 0)
  while :; do
    v=$(cat "$GTT" 2>/dev/null || echo 0); (( v > g )) && g=$v
    for pid in $(pgrep -x qwenox); do
      v=$(awk '/^VmRSS/{print $2*1024}' /proc/$pid/status 2>/dev/null || echo 0)
      (( ${v:-0} > r )) && r=$v
    done
    awk -v g=$g -v g0=$g0 -v r=$r 'BEGIN{printf "gtt_peak=%.1f gtt_delta=%.1f rss_peak=%.1f\n",g/2^30,(g-g0)/2^30,r/2^30}' >"$out"
    sleep 1
  done
}
SPID=
start_sampler() { sampler "logs/v2v_mem_$1.txt" & SPID=$!; }
stop_sampler() { kill "$SPID" 2>/dev/null; wait "$SPID" 2>/dev/null; }

kv() { sed -n "s/.*$1=\([0-9.eE+-]*\).*/\1/p" <<<"$2" | tail -1; }

run_kld() {  # 标签 chunks [K=V ...]；env 已设好 MODEL_FILE 等
  local label=$1 chunks=$2; shift 2
  drop_cache
  start_sampler "$label"
  OUT=$(CHUNKS=$chunks bash tools/kld_engine.sh "$REF" "v2v_$label" "$@" 2>&1)
  local rc=$?
  stop_sampler
  echo "$OUT" | grep -E '^\[|kld_summary|hipError|FATAL|Segmentation|Aborted' | cut -c1-220
  echo "  mem: $(cat logs/v2v_mem_$label.txt 2>/dev/null)"
  echo "  $(grep -m1 -E 'weight arena' logs/kld_v2v_$label.log | cut -c1-160)"
  return $rc
}

v2_env() { export MODEL_FILE=$V2 NGRAM_FILE=$NG OVERLAY_FILE= MTP_FILE= VISION_FILE=; }
t0=$SECONDS

echo "== 1/4 v2 prefill KLD（$CHUNKS chunk）=="
v2_env
CMD=$(bash start_hgn.sh --check 2>&1 | sed -n 's/^CMD //p')
eval "C=($CMD)"
if [[ ${C[1]:-} -ef $V2 && ${C[2]:-} -ef $NG ]]; then echo "  launcher args OK: ${C[1]} ${C[2]}"
else fail "start_hgn.sh 拼参不对：${C[*]:0:4}"; fi
run_kld prefill "$CHUNKS" || fail "step1 rc!=0"
K1=$(kv mean_kld "$OUT"); T1=$(kv same_top "$OUT")
awk -v k="${K1:-99}" -v t="${T1:-0}" -v m="$MAX_KLD" 'BEGIN{exit !(k<m && t>70)}' ||
  fail "v2 prefill KLD=${K1:-?} same_top=${T1:-?}（要求 <$MAX_KLD 且 >70）"

echo "== 2/4 v2 小 batch naive 路径 KLD（8 chunk，QWENOX_MOE_NAIVE_MAX=1000000）=="
run_kld naive 8 QWENOX_MOE_NAIVE_MAX=1000000 || fail "step2 rc!=0"
K2=$(kv mean_kld "$OUT"); T2=$(kv same_top "$OUT")
awk -v k="${K2:-99}" -v t="${T2:-0}" -v m="$MAX_KLD" 'BEGIN{exit !(k<m && t>70)}' ||
  fail "v2 naive KLD=${K2:-?} same_top=${T2:-?}"

echo "== 3/4 v2 decode vs prefill（--ppl，512 token）=="
python3 - "$REF" logs/v2v_tok512.txt <<'PY'
import struct, sys
with open(sys.argv[1], 'rb') as f:
    assert f.read(8) == b'_logits_'
    n_ctx, nv, nch = struct.unpack('<3i', f.read(12))
    toks = struct.unpack('<512i', f.read(2048))
open(sys.argv[2], 'w').write(' '.join(map(str, toks)) + '\n')
PY
chk="$(bash start_hgn.sh --check 2>&1)"
mapfile -t PENV < <(sed -n 's/^ENV //p' <<<"$chk")
ppl_run() {  # 标签 [K=V ...]
  local label=$1; shift
  drop_cache
  start_sampler "$label"
  ( for e in $(compgen -e | grep '^QWENOX_'); do unset "$e"; done
    for e in "${PENV[@]}" QWENOX_KVSNAP=0 "$@"; do export "$e"; done
    bash tools/run_capped.sh 86 -- "$BIN" "$V2" "$NG" --tokens-file logs/v2v_tok512.txt --ppl \
      --maxctx 2048 >"logs/v2v_$label.log" 2>&1 )
  local rc=$?
  stop_sampler
  PPL=$(grep ppl_summary "logs/v2v_$label.log" | tail -1)
  echo "  $label rc=$rc ${PPL:-（无 ppl_summary，见 logs/v2v_$label.log）}"
  echo "  mem: $(cat logs/v2v_mem_$label.txt 2>/dev/null)"
  return $rc
}
ppl_run ppl_prefill || fail "ppl prefill rc!=0"; NP=$(kv mean_nll "$PPL")
ppl_run ppl_decode QWENOX_NOPREFILLBATCH=1 || fail "ppl decode rc!=0"; ND=$(kv mean_nll "$PPL")
awk -v a="${NP:-0}" -v b="${ND:-99}" 'BEGIN{d=a-b; if(d<0)d=-d; exit !(a>0 && a<3 && d<0.03)}' ||
  fail "decode/prefill mean_nll 不一致：prefill=${NP:-?} decode=${ND:-?}（|Δ| 要求 <0.03）"

echo "== 4/4 v1 回归：w4b + overlay KLD（$CHUNKS chunk，基线 $V1_KLD）=="
export MODEL_FILE=$W4B NGRAM_FILE= OVERLAY_FILE=$OVL MTP_FILE= VISION_FILE=
unset NGRAM_FILE
CMD=$(bash start_hgn.sh --check 2>&1 | sed -n 's/^CMD //p')
eval "C=($CMD)"
[[ ${C[2]:-} -ef $OVL && ${C[3]:-} == --* ]] || fail "w4b 拼参不对（n-gram 不应重复传）：${C[*]:0:4}"
run_kld v1 "$CHUNKS" || fail "step4 rc!=0"
K4=$(kv mean_kld "$OUT")
if [[ $CHUNKS == 64 ]]; then
  awk -v k="${K4:-99}" -v b="$V1_KLD" 'BEGIN{d=k-b; if(d<0)d=-d; exit !(d<0.003)}' ||
    fail "v1 KLD=${K4:-?} 偏离基线 $V1_KLD"
fi

echo "== 汇总（$((SECONDS - t0))s）=="
printf '  v2 prefill KLD %s same_top %s | naive KLD %s | ppl nll prefill %s decode %s | v1 KLD %s\n' \
  "${K1:-?}" "${T1:-?}" "${K2:-?}" "${NP:-?}" "${ND:-?}" "${K4:-?}"
for l in prefill v1; do printf '  %-8s %s\n' "$l" "$(cat logs/v2v_mem_$l.txt 2>/dev/null)"; done
if (( ${#FAILS[@]} )); then printf '  - %s\n' "${FAILS[@]}"; echo FAIL; exit 1; fi
echo PASS
