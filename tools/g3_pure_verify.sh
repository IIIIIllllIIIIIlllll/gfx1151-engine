#!/usr/bin/env bash
# G3.3 验证：纯 GGUF 启动（不需要任何 .hgn）——
#   gdec <第一个分片>.gguf ...  = GDEC_GGUF + GDEC_GGUF_DENSE=1 + 同目录 mtp-*.gguf，
#   PLE n-gram 表直接用 GGUF 的 IQ4_NL（90 B/行，GPU/CPU 解量化）。
# 一键跑完，最后打印 PASS / FAIL。
#   bash tools/g3_pure_verify.sh
#     GGDIR=<GGUF 目录>       默认 ~/App/llama.cpp/models/Qwen3.8-Flash-Next-UD-Q4_K_XL
#     BIN=build/gdec-gguf     REF=build/gdec-head（hgn 生产路径的参照二进制，HEAD 编出）
#     QUICK=1                 KLD 只跑 16 chunk，跳过速度
# 检查项：
#   0. start.sh --check：MODEL_FILE=*.gguf 时命令行无 overlay、MTP 走 GDEC_GGUF_MTP、视觉塔用 mmproj
#   1. KLD（BF16 基准 64×512），纯 GGUF < KMAX（默认 0.055）。纯 GGUF 与 llama.cpp 用的是
#      同一份权重（llama.cpp 0.049），差别只剩计算路径（bf16 KV 等）；G3.2 的 0.0453 用的是
#      hgn 的 fp8 PLE 表（比 GGUF 的 IQ4_NL 准），纯 GGUF 拿不到
#   2. 纯 GGUF 与混合启动（hgn 基座 + GGUF 全覆盖 + GDEC_GGUF_PLE=1）prefill PPL 逐位相同
#      ——证明没有任何张量还在从 hgn 读
#   3. 纯 GGUF：逐 token decode PPL vs batched prefill PPL 相对差 < 1%
#   4. 不设 GGUF 变量时 BIN 与 REF 的 prefill/decode PPL 逐位相同（hgn 路径没被改动）
#   5. MTP 投机（8K prompt，256 token，gamma 3）纯 GGUF commit/round >= hgn 的 0.9 倍
#   6. 速度（只报数）：pp 8K@2048 / 32K@16384 + decode@32K，G3.2（fp8 PLE）vs 纯 GGUF
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
D=${GGDIR:-$HOME/App/llama.cpp/models/Qwen3.8-Flash-Next-UD-Q4_K_XL}
GG=$D/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf
MTP=$D/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf
MMPROJ=$D/mmproj-BF16.gguf
BIN=${BIN:-build/gdec-gguf}
REF=${REF:-build/gdec-head}
KMAX=${KMAX:-0.055}
export PROBE_CAP_GB=${PROBE_CAP_GB:-100}
KREF=data/kld/bf16_c512.kld
TOKSRC=$HOME/ppbench/tok8192.txt
# 纯 GGUF 的 service.conf 覆盖（start.sh --check / pp_prod.sh / kld_engine.sh 都读它）
PURE=(MODEL_FILE="$GG" MTP_FILE="$MTP" VISION_FILE="$MMPROJ")
# G3.2 的混合启动：hgn 基座 + overlay，GGUF 覆盖全部张量但 PLE 表仍是 hgn fp8
G32=(GDEC_GGUF="$GG" GDEC_GGUF_DENSE=1 GDEC_GGUF_MTP="$MTP")
fail=0
note() { echo "== $*"; }
bad() { echo "FAIL: $*" >&2; fail=1; }

for f in "$GG" "$MTP" "$MMPROJ" "$BIN" "$REF" "$KREF" "$TOKSRC" "$HOME/ppbench/tok32768.txt"; do
  [[ -e "$f" ]] || { echo "缺少 $f"; echo FAIL; exit 1; }
done
if pgrep -af '(^|/)(gdec[^/[:space:]]*|flash_serve|serve_api\.py|llama-server|llama-perplexity|llama-cli)([[:space:]]|$)' >/dev/null; then
  echo "GPU 上已有引擎/llama.cpp 在跑，请先停掉"; echo FAIL; exit 1
fi
mkdir -p logs

# ---- 0. start.sh --check ---------------------------------------------------
note "0. start.sh --check（MODEL_FILE=*.gguf）"
pchk="$(env "${PURE[@]}" bash start.sh --check 2>&1)" || { echo "$pchk"; bad "start.sh --check 失败"; }
pcmd=$(sed -n 's/^CMD //p' <<<"$pchk")
eval "PC=($pcmd)"
echo "CMD $pcmd" | cut -c1-240
[[ "${PC[1]:-}" == "$GG" && "${PC[2]:-}" == --serve ]] || bad "命令行应为 <gdec> $GG --serve ...（不带 overlay/MTP 位置参数）"
grep -qx "ENV GDEC_GGUF_MTP=$MTP" <<<"$pchk" || bad "ENV 里没有 GDEC_GGUF_MTP=$MTP"
[[ "$pcmd" == *"--vision-tower $MMPROJ"* ]] || bad "命令行没有 --vision-tower $MMPROJ"
(( fail )) || echo OK

# ---- 1. KLD ----------------------------------------------------------------
NCH=$([[ -n ${QUICK:-} ]] && echo 16 || echo 64)
note "1. KLD vs BF16（$NCH chunk × 512），纯 GGUF"
out=$(env "${PURE[@]}" BIN=$BIN CHUNKS=$NCH bash tools/kld_engine.sh "$KREF" g33_kld GDEC_KVSNAP=0 2>&1)
echo "$out" | grep -E '^\[|KLD|top p|PPL' | head -8
L=logs/kld_g33_kld.log
grep -q '^gguf: pure GGUF start' $L || bad "日志里没有 'gguf: pure GGUF start'"
grep -q '^gguf experts: 48 layers.*MTP sidecar' $L || bad "日志里没有 'gguf experts: 48 layers ... + MTP sidecar'"
grep -q '^gguf dense: .*MTP sidecar.*PLE table IQ4_NL' $L || bad "日志里没有 'gguf dense: ... MTP sidecar ... PLE table IQ4_NL'"
grep -E '^gguf' $L | cut -c1-200
kld=$(sed -n 's/^Mean    KLD: *\([0-9.]*\).*/\1/p' <<<"$out" | head -1)
top=$(sed -n 's/^Same top p: *\([0-9.]*\).*/\1/p' <<<"$out" | head -1)
if [[ -z "$kld" ]]; then bad "KLD 没有输出"
elif awk -v k="$kld" -v m="$KMAX" 'BEGIN{exit !(k < m)}'; then
  echo "KLD=$kld top1=$top%  (G3.2 fp8 PLE 0.0453 / 93.25%，llama.cpp 同权重 0.049，hgn 0.1631)  OK"
else bad "KLD=$kld 不低于 $KMAX"; fi

# ---- 2/3/4. PPL ------------------------------------------------------------
TOK=logs/g3_tok1024.txt
tr -s ' ,\t' '\n' <"$TOKSRC" | grep -E '^[0-9]+$' | head -1024 >"$TOK"
chk="$(bash start.sh --check 2>&1)" || { echo "$chk"; echo FAIL; exit 1; }
mapfile -t PENV < <(sed -n 's/^ENV //p' <<<"$chk")
eval "C=($(sed -n 's/^CMD //p' <<<"$chk"))"
M=${C[1]}; O=${C[2]}; [[ "$O" == --* ]] && O=""
run_ppl() {  # label bin model overlay extra-env...  -> "ppl mean_nll"
  local label=$1 bin=$2 m=$3 o=$4; shift 4
  ( for e in $(compgen -e | grep '^GDEC_'); do unset "$e"; done
    for e in "${PENV[@]}" GDEC_KVSNAP=0 "$@"; do export "$e"; done
    bash tools/run_capped.sh "$PROBE_CAP_GB" -- "$bin" "$m" ${o:+"$o"} --tokens-file "$TOK" \
      --ppl --maxctx 4096 >"logs/$label.log" 2>&1 )
  grep -qE 'hipError|Segmentation|Aborted|FATAL|what\(\)' "logs/$label.log" && bad "$label 崩溃（logs/$label.log）"
  sed -n 's/^ppl_summary.*mean_nll=\([0-9.]*\).*ppl=\([0-9.]*\).*/\2 \1/p' "logs/$label.log"
}
note "2. 纯 GGUF vs 混合启动（hgn 基座 + GGUF 全覆盖 + IQ4_NL PLE），1024 token prefill"
read -r pp_p np < <(run_ppl g33_pure_prefill "$BIN" "$GG" "" GDEC_GGUF_MTP="$MTP")
read -r pp_h nh < <(run_ppl g33_hyb_prefill "$BIN" "$M" "$O" GDEC_GGUF="$GG" GDEC_GGUF_DENSE=1 GDEC_GGUF_MTP="$MTP" GDEC_GGUF_PLE=1)
echo "mean_nll pure=${np:-?} hybrid=${nh:-?}  (PPL $pp_p / $pp_h)"
grep -q '^loaded 0 tensors' logs/g33_pure_prefill.log || bad "纯 GGUF 启动时基座不是空的"
if [[ -n "${np:-}" && "${np:-}" == "${nh:-}" ]]; then echo "逐位相同  OK"
else bad "纯 GGUF 与混合启动结果不同"; fi

note "3. 纯 GGUF：逐 token decode vs batched prefill（1024 token）"
read -r pp_d nd < <(run_ppl g33_pure_decode "$BIN" "$GG" "" GDEC_GGUF_MTP="$MTP" GDEC_NOPREFILLBATCH=1)
echo "PPL prefill=${pp_p:-?} decode=${pp_d:-?}  (G3.1 6.814 / 6.760 = 0.8%，hgn 自身 0.6%)"
if [[ -z "${pp_p:-}" || -z "${pp_d:-}" ]]; then bad "PPL 没有输出（见 logs/g33_pure_*.log）"
elif awk -v a="$pp_p" -v b="$pp_d" 'BEGIN{d=(a-b)/a; if(d<0)d=-d; exit !(d < 0.01)}'; then echo "相对差 < 1%  OK"
else bad "decode 与 prefill PPL 相差 >= 1%"; fi

note "4. hgn 路径不变：$BIN vs $REF（不设 GGUF 变量）"
read -r _ rnb < <(run_ppl g33_hgn_ref_prefill "$REF" "$M" "$O")
read -r _ nnb < <(run_ppl g33_hgn_new_prefill "$BIN" "$M" "$O")
read -r _ rnd < <(run_ppl g33_hgn_ref_decode "$REF" "$M" "$O" GDEC_NOPREFILLBATCH=1)
read -r _ nnd < <(run_ppl g33_hgn_new_decode "$BIN" "$M" "$O" GDEC_NOPREFILLBATCH=1)
echo "prefill mean_nll ref=${rnb:-?} new=${nnb:-?}   decode ref=${rnd:-?} new=${nnd:-?}"
if [[ -n "${rnb:-}" && "${rnb:-}" == "${nnb:-}" && -n "${rnd:-}" && "${rnd:-}" == "${nnd:-}" ]]; then
  echo "逐位相同  OK"
else bad "hgn 路径结果与 $REF 不同"; fi

# ---- 5. MTP 投机 -------------------------------------------------------------
spec() {  # label bin [env overrides for start.sh] -- extra...
  local label=$1 bin=$2; shift 2
  local pre=()
  while (($#)) && [[ $1 != -- ]]; do pre+=("$1"); shift; done
  (($#)) && shift
  env "${pre[@]}" BIN=$bin SPEC=256 GAMMA=3 bash tools/pp_prod.sh 8k "$label" "$@" >/dev/null 2>&1
  grep -qE 'hipError|Segmentation|Aborted|FATAL' "logs/$label.log" && bad "$label 崩溃（logs/$label.log）"
  tr '\r' '\n' <"logs/$label.log" | grep -ao 'spec: [0-9]* tokens in [0-9.]* s = [0-9.]* tok/s | rounds=[0-9]* commit/round=[0-9.]*' |
    tail -1 | sed 's/.*= \([0-9.]*\) tok\/s.*commit\/round=\([0-9.]*\).*/\2 \1/'
}
note "5. MTP 投机：8K prompt，256 token，gamma 3"
read -r ch sh < <(spec g33_spec_hgn "$REF" --)
read -r cg sg < <(spec g33_spec_pure "$BIN" "${PURE[@]}" --)
echo "commit/round hgn=${ch:-?} pure=${cg:-?}   tok/s hgn=${sh:-?} pure=${sg:-?}  (G3.2 3.82 / 46)"
if [[ -z "${ch:-}" || -z "${cg:-}" ]]; then bad "spec 没有输出（见 logs/g33_spec_*.log）"
elif awk -v a="$cg" -v b="$ch" 'BEGIN{exit !(a >= 0.9*b)}'; then echo "OK"
else bad "纯 GGUF MTP commit/round 低于 hgn 的 0.9 倍"; fi

# ---- 6. 速度 -----------------------------------------------------------------
speed() {  # label bin tok gen [start.sh overrides] -- extra...
  local label=$1 bin=$2 tok=$3 gen=$4; shift 4
  local pre=()
  while (($#)) && [[ $1 != -- ]]; do pre+=("$1"); shift; done
  (($#)) && shift
  env "${pre[@]}" BIN=$bin GEN=$gen bash tools/pp_prod.sh "$tok" "$label" "$@" >/dev/null 2>&1
  local p d
  p=$(tr '\r' '\n' <"logs/$label.log" | sed -n 's/.*prefill: \([0-9]*\) tokens in \([0-9.]*\) s = \([0-9.]*\) tok\/s.*/\3/p' | tail -1)
  d=$(grep -ao 'decode: [0-9]* tokens in [0-9]* ms = [0-9.]* tok/s' "logs/$label.log" | tail -1 | sed 's/.*= \([0-9.]*\) tok.*/\1/')
  grep -qE 'hipError|Segmentation|Aborted|FATAL' "logs/$label.log" && bad "$label 崩溃（logs/$label.log）"
  echo "${p:-?} ${d:--}"
}
if [[ -z ${QUICK:-} ]]; then
  note "6a. pp 8K @ chunk 2048（最后一个 chunk 的 tok/s）"
  read -r a _ < <(speed g33_pp8k_g32 "$BIN" 8k 1 -- GDEC_PREFILL_CHUNK=2048 "${G32[@]}")
  read -r b _ < <(speed g33_pp8k_pure "$BIN" 8k 1 "${PURE[@]}" -- GDEC_PREFILL_CHUNK=2048)
  echo "G3.2(fp8 PLE)=$a  pure(IQ4_NL PLE)=$b tok/s"
  note "6b. pp 32K @ chunk 16384 + decode 128"
  read -r a ad < <(speed g33_pp32k_g32 "$BIN" 32k 128 -- "${G32[@]}")
  read -r b bd < <(speed g33_pp32k_pure "$BIN" 32k 128 "${PURE[@]}" --)
  echo "pp     G3.2=$a  pure=$b tok/s"
  echo "decode G3.2=$ad pure=$bd tok/s"
fi

echo
if (( fail )); then echo FAIL; exit 1; else echo PASS; fi
