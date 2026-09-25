#!/usr/bin/env bash
# G2 验证：trunk MoE 专家权重改由 llama.cpp GGUF（UD-Q4_K_XL）提供（GDEC_GGUF），
# 其余权重仍来自 hgn。一键跑完，最后打印 PASS / FAIL。
#   bash tools/g2_verify.sh
#     GG=<第一个分片>     默认 ~/App/llama.cpp/models/Qwen3.8-Flash-Next-UD-Q4_K_XL/…-00001-of-00004.gguf
#     BIN=build/gdec-gguf  REF=build/gdec（hgn 生产二进制，做速度对照）
#     QUICK=1              KLD 只跑 16 chunk
# 检查项：
#   1. KLD（BF16 基准 data/kld/bf16_c512.kld，64×512，prefill 路径）< 0.20；生产 hgn = 0.1631
#   2. 同一 1024 token 文本：逐 token decode 的 PPL 与 batched prefill 的 PPL 相对差 < 1%
#      （decode P=1 路由核 vs prefill 排序路由 + WMMA，两条路互证）
#   3. pp：8K@chunk2048、32K@chunk16384（生产），GGUF vs hgn（只报数，不判）
#   4. decode：32K 后生成 128 token 的 tok/s，GGUF vs hgn（只报数，不判）
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
GG=${GG:-$HOME/App/llama.cpp/models/Qwen3.8-Flash-Next-UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf}
BIN=${BIN:-build/gdec-gguf}
REF=${REF:-build/gdec}
export PROBE_CAP_GB=${PROBE_CAP_GB:-100}   # GGUF 专家 ~70 GiB 锁页 + hgn 其余权重
KREF=data/kld/bf16_c512.kld
TOKSRC=$HOME/ppbench/tok8192.txt
fail=0
note() { echo "== $*"; }
bad() { echo "FAIL: $*"; fail=1; }

for f in "$GG" "$BIN" "$REF" "$KREF" "$TOKSRC" "$HOME/ppbench/tok32768.txt"; do
  [[ -e "$f" ]] || { echo "缺少 $f"; echo FAIL; exit 1; }
done
if pgrep -af '(^|/)(gdec[^/[:space:]]*|flash_serve|serve_api\.py|llama-server|llama-perplexity|llama-cli)([[:space:]]|$)' >/dev/null; then
  echo "GPU 上已有引擎/llama.cpp 在跑，请先停掉"; echo FAIL; exit 1
fi
mkdir -p logs

# ---- 1. KLD ----------------------------------------------------------------
note "1. KLD vs BF16（${QUICK:+16}${QUICK:-64} chunk × 512）"
out=$(BIN=$BIN CHUNKS=$([[ -n ${QUICK:-} ]] && echo 16 || echo 64) \
      bash tools/kld_engine.sh "$KREF" g2_kld GDEC_GGUF="$GG" 2>&1)
echo "$out"
grep -q '^gguf experts: 48 layers' logs/kld_g2_kld.log || bad "日志里没有 'gguf experts: 48 layers'（GGUF 没生效？）"
grep -m1 '^gguf experts' logs/kld_g2_kld.log
kld=$(sed -n 's/^Mean    KLD: *\([0-9.]*\).*/\1/p' <<<"$out" | head -1)
top=$(sed -n 's/^Same top p: *\([0-9.]*\).*/\1/p' <<<"$out" | head -1)
if [[ -z "$kld" ]]; then bad "KLD 没有输出"
elif awk -v k="$kld" 'BEGIN{exit !(k < 0.20)}'; then echo "KLD=$kld top1=$top%  (hgn 生产 0.1631 / 86.61%)  OK"
else bad "KLD=$kld >= 0.20"; fi

# ---- 2. decode vs prefill PPL ------------------------------------------------
note "2. PPL：逐 token decode vs batched prefill（1024 token）"
TOK=logs/g2_tok1024.txt
tr -s ' ,\t' '\n' <"$TOKSRC" | grep -E '^[0-9]+$' | head -1024 >"$TOK"
chk="$(bash start_hgn.sh --check 2>&1)" || { echo "$chk"; echo FAIL; exit 1; }
mapfile -t PENV < <(sed -n 's/^ENV //p' <<<"$chk")
eval "C=($(sed -n 's/^CMD //p' <<<"$chk"))"
M=${C[1]}; O=${C[2]}; [[ "$O" == --* ]] && O=""
run_ppl() {  # label extra-env...
  local label=$1; shift
  ( for e in $(compgen -e | grep '^GDEC_'); do unset "$e"; done
    for e in "${PENV[@]}" GDEC_KVSNAP=0 GDEC_GGUF="$GG" "$@"; do export "$e"; done
    bash tools/run_capped.sh "$PROBE_CAP_GB" -- "$BIN" "$M" ${O:+"$O"} --tokens-file "$TOK" \
      --ppl --maxctx 4096 >"logs/$label.log" 2>&1 )
  sed -n 's/^ppl_summary.*ppl=\([0-9.]*\).*/\1/p' "logs/$label.log"
}
pp_b=$(run_ppl g2_ppl_prefill)
pp_d=$(run_ppl g2_ppl_decode GDEC_NOPREFILLBATCH=1)
echo "PPL prefill=$pp_b decode=$pp_d"
if [[ -z "$pp_b" || -z "$pp_d" ]]; then bad "PPL 没有输出（见 logs/g2_ppl_*.log）"
elif awk -v a="$pp_b" -v b="$pp_d" 'BEGIN{d=(a-b)/a; if(d<0)d=-d; exit !(d < 0.01)}'; then echo "相对差 < 1%  OK"
else bad "decode 与 prefill PPL 相差 >= 1%"; fi

# ---- 3/4. pp + decode 速度 -----------------------------------------------------
speed() {  # label bin tok gen extra...
  local label=$1 bin=$2 tok=$3 gen=$4; shift 4
  BIN=$bin GEN=$gen bash tools/pp_prod.sh "$tok" "$label" "$@" >/dev/null 2>&1
  local p d
  p=$(tr '\r' '\n' <"logs/$label.log" | sed -n 's/.*prefill: \([0-9]*\) tokens in \([0-9.]*\) s = \([0-9.]*\) tok\/s.*/\3/p' | tail -1)
  # stdout/stderr interleave: "decode:" is not always at line start
  d=$(grep -ao 'decode: [0-9]* tokens in [0-9]* ms = [0-9.]* tok/s' "logs/$label.log" | tail -1 | sed 's/.*= \([0-9.]*\) tok.*/\1/')
  grep -qE 'hipError|Segmentation|Aborted|FATAL' "logs/$label.log" && bad "$label 崩溃（logs/$label.log）"
  echo "${p:-?} ${d:--}"
}
note "3. pp 8K @ chunk 2048"
read -r a _ < <(speed g2_pp8k_hgn "$REF" 8k 1 GDEC_PREFILL_CHUNK=2048)
read -r b _ < <(speed g2_pp8k_gguf "$BIN" 8k 1 GDEC_PREFILL_CHUNK=2048 GDEC_GGUF="$GG")
echo "pp(最后一个 2048 chunk)  hgn=$a  gguf=$b tok/s"
note "3/4. pp 32K @ chunk 16384 + decode 128"
read -r a ad < <(speed g2_pp32k_hgn "$REF" 32k 128)
read -r b bd < <(speed g2_pp32k_gguf "$BIN" 32k 128 GDEC_GGUF="$GG")
echo "pp(最后一个 16384 chunk) hgn=$a  gguf=$b tok/s"
echo "decode@32K               hgn=$ad gguf=$bd tok/s"

echo
if (( fail )); then echo FAIL; exit 1; else echo PASS; fi
