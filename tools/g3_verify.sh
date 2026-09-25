#!/usr/bin/env bash
# G3.1 验证：除 PLE n-gram 表和 MTP 路由专家外，全部权重来自 llama.cpp GGUF
# （UD-Q4_K_XL 主体 + MTP Q8_0 sidecar；GDEC_GGUF + GDEC_GGUF_DENSE=1 + GDEC_GGUF_MTP）。
# 一键跑完，最后打印 PASS / FAIL。
#   bash tools/g3_verify.sh
#     GGDIR=<GGUF 目录>    默认 ~/App/llama.cpp/models/Qwen3.8-Flash-Next-UD-Q4_K_XL
#     BIN=build/gdec-gguf  REF=build/gdec（hgn 生产二进制）
#       REF 必须是改动前那个提交编出来的二进制：09-24 的旧 build/gdec 逐 token decode
#       本身不确定（同一输入 3 次 3 个 NLL），第 3 项会误报。例如 REF=build/gdec-head
#     QUICK=1              KLD 只跑 16 chunk，跳过速度
# 检查项：
#   1. KLD（BF16 基准 data/kld/bf16_c512.kld，64×512）< G2 的 0.1494（dense 从 q4cp 换成 Q8_0 应更好）
#   2. 同一 1024 token：逐 token decode PPL vs batched prefill PPL 相对差 < 1%
#      （decode 走 q8g32 GEMV，prefill 走 q8g32→bf16 GEMM，两条路互证）
#   3. 不设 GGUF 变量时 BIN 与 REF 的 prefill/decode PPL 逐位相同（hgn 路径没被改动）
#   4. MTP 投机（8K prompt，256 token，gamma 3）commit/round >= hgn 的 0.9 倍
#   5. pp 8K@2048 / 32K@16384 + decode@32K：hgn vs G2 vs G3（只报数，不判）
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
D=${GGDIR:-$HOME/App/llama.cpp/models/Qwen3.8-Flash-Next-UD-Q4_K_XL}
GG=$D/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf
MTP=$D/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf
BIN=${BIN:-build/gdec-gguf}
REF=${REF:-build/gdec}
export PROBE_CAP_GB=${PROBE_CAP_GB:-100}
KREF=data/kld/bf16_c512.kld
TOKSRC=$HOME/ppbench/tok8192.txt
G3=(GDEC_GGUF="$GG" GDEC_GGUF_DENSE=1 GDEC_GGUF_MTP="$MTP")
fail=0
note() { echo "== $*"; }
bad() { echo "FAIL: $*" >&2; fail=1; }  # stderr: also visible from inside < <(...)

for f in "$GG" "$MTP" "$BIN" "$REF" "$KREF" "$TOKSRC" "$HOME/ppbench/tok32768.txt"; do
  [[ -e "$f" ]] || { echo "缺少 $f"; echo FAIL; exit 1; }
done
if pgrep -af '(^|/)(gdec[^/[:space:]]*|flash_serve|serve_api\.py|llama-server|llama-perplexity|llama-cli)([[:space:]]|$)' >/dev/null; then
  echo "GPU 上已有引擎/llama.cpp 在跑，请先停掉"; echo FAIL; exit 1
fi
mkdir -p logs

# ---- 1. KLD ----------------------------------------------------------------
note "1. KLD vs BF16（$([[ -n ${QUICK:-} ]] && echo 16 || echo 64) chunk × 512）"
out=$(BIN=$BIN CHUNKS=$([[ -n ${QUICK:-} ]] && echo 16 || echo 64) \
      bash tools/kld_engine.sh "$KREF" g3_kld "${G3[@]}" 2>&1)
echo "$out"
L=logs/kld_g3_kld.log
grep -q '^gguf experts: 48 layers' $L || bad "日志里没有 'gguf experts: 48 layers'"
grep -q '^gguf dense: .* MTP sidecar' $L || bad "日志里没有 'gguf dense: ... MTP sidecar'"
grep -E '^gguf (dense|experts)' $L | cut -c1-160
kld=$(sed -n 's/^Mean    KLD: *\([0-9.]*\).*/\1/p' <<<"$out" | head -1)
top=$(sed -n 's/^Same top p: *\([0-9.]*\).*/\1/p' <<<"$out" | head -1)
if [[ -z "$kld" ]]; then bad "KLD 没有输出"
elif awk -v k="$kld" 'BEGIN{exit !(k < 0.1494)}'; then
  echo "KLD=$kld top1=$top%  (G2 0.1494 / 87.20%，hgn 0.1631 / 86.61%，llama Q4_K_XL 0.049)  OK"
else bad "KLD=$kld 不低于 G2 的 0.1494"; fi

# ---- 2/3. PPL --------------------------------------------------------------
TOK=logs/g3_tok1024.txt
tr -s ' ,\t' '\n' <"$TOKSRC" | grep -E '^[0-9]+$' | head -1024 >"$TOK"
chk="$(bash start.sh --check 2>&1)" || { echo "$chk"; echo FAIL; exit 1; }
mapfile -t PENV < <(sed -n 's/^ENV //p' <<<"$chk")
eval "C=($(sed -n 's/^CMD //p' <<<"$chk"))"
M=${C[1]}; O=${C[2]}; [[ "$O" == --* ]] && O=""
run_ppl() {  # label bin extra-env...  -> "ppl mean_nll"
  local label=$1 bin=$2; shift 2
  ( for e in $(compgen -e | grep '^GDEC_'); do unset "$e"; done
    for e in "${PENV[@]}" GDEC_KVSNAP=0 "$@"; do export "$e"; done
    bash tools/run_capped.sh "$PROBE_CAP_GB" -- "$bin" "$M" ${O:+"$O"} --tokens-file "$TOK" \
      --ppl --maxctx 4096 >"logs/$label.log" 2>&1 )
  grep -qE 'hipError|Segmentation|Aborted|FATAL|what\(\)' "logs/$label.log" && bad "$label 崩溃（logs/$label.log）"
  sed -n 's/^ppl_summary.*mean_nll=\([0-9.]*\).*ppl=\([0-9.]*\).*/\2 \1/p' "logs/$label.log"
}
note "2. PPL：逐 token decode vs batched prefill（1024 token，G3）"
read -r pp_b nb < <(run_ppl g3_ppl_prefill "$BIN" "${G3[@]}")
read -r pp_d nd < <(run_ppl g3_ppl_decode "$BIN" "${G3[@]}" GDEC_NOPREFILLBATCH=1)
echo "PPL prefill=$pp_b decode=$pp_d  (G2 6.941 / 6.929)"
if [[ -z "${pp_b:-}" || -z "${pp_d:-}" ]]; then bad "PPL 没有输出（见 logs/g3_ppl_*.log）"
elif awk -v a="$pp_b" -v b="$pp_d" 'BEGIN{d=(a-b)/a; if(d<0)d=-d; exit !(d < 0.01)}'; then echo "相对差 < 1%  OK"
else bad "decode 与 prefill PPL 相差 >= 1%"; fi

note "3. hgn 路径不变：$BIN vs $REF（不设 GGUF 变量）"
read -r r_b rnb < <(run_ppl g3_hgn_ref_prefill "$REF")
read -r n_b nnb < <(run_ppl g3_hgn_new_prefill "$BIN")
read -r r_d rnd < <(run_ppl g3_hgn_ref_decode "$REF" GDEC_NOPREFILLBATCH=1)
read -r n_d nnd < <(run_ppl g3_hgn_new_decode "$BIN" GDEC_NOPREFILLBATCH=1)
echo "prefill mean_nll ref=${rnb:-?} new=${nnb:-?}   decode ref=${rnd:-?} new=${nnd:-?}"
if [[ -n "${rnb:-}" && "${rnb:-}" == "${nnb:-}" && -n "${rnd:-}" && "${rnd:-}" == "${nnd:-}" ]]; then
  echo "逐位相同  OK"
else bad "hgn 路径结果与 $REF 不同"; fi

# ---- 4. MTP 投机 -------------------------------------------------------------
spec() {  # label bin extra... -> "commit/round tok/s"
  local label=$1 bin=$2; shift 2
  BIN=$bin SPEC=256 GAMMA=3 bash tools/pp_prod.sh 8k "$label" "$@" >/dev/null 2>&1
  grep -qE 'hipError|Segmentation|Aborted|FATAL' "logs/$label.log" && bad "$label 崩溃（logs/$label.log）"
  tr '\r' '\n' <"logs/$label.log" | grep -ao 'spec: [0-9]* tokens in [0-9.]* s = [0-9.]* tok/s | rounds=[0-9]* commit/round=[0-9.]*' |
    tail -1 | sed 's/.*= \([0-9.]*\) tok\/s.*commit\/round=\([0-9.]*\).*/\2 \1/'
}
note "4. MTP 投机：8K prompt，256 token，gamma 3"
read -r ch sh < <(spec g3_spec_hgn "$REF")
read -r cg sg < <(spec g3_spec_gguf "$BIN" "${G3[@]}")
echo "commit/round hgn=${ch:-?} gguf=${cg:-?}   tok/s hgn=${sh:-?} gguf=${sg:-?}"
grep -h '^spec depth acc' logs/g3_spec_hgn.log logs/g3_spec_gguf.log 2>/dev/null
if [[ -z "${ch:-}" || -z "${cg:-}" ]]; then bad "spec 没有输出（见 logs/g3_spec_*.log）"
elif awk -v a="$cg" -v b="$ch" 'BEGIN{exit !(a >= 0.9*b)}'; then echo "OK"
else bad "GGUF MTP commit/round 低于 hgn 的 0.9 倍"; fi

# ---- 5. 速度 -----------------------------------------------------------------
speed() {  # label bin tok gen extra...
  local label=$1 bin=$2 tok=$3 gen=$4; shift 4
  BIN=$bin GEN=$gen bash tools/pp_prod.sh "$tok" "$label" "$@" >/dev/null 2>&1
  local p d
  p=$(tr '\r' '\n' <"logs/$label.log" | sed -n 's/.*prefill: \([0-9]*\) tokens in \([0-9.]*\) s = \([0-9.]*\) tok\/s.*/\3/p' | tail -1)
  d=$(grep -ao 'decode: [0-9]* tokens in [0-9]* ms = [0-9.]* tok/s' "logs/$label.log" | tail -1 | sed 's/.*= \([0-9.]*\) tok.*/\1/')
  grep -qE 'hipError|Segmentation|Aborted|FATAL' "logs/$label.log" && bad "$label 崩溃（logs/$label.log）"
  echo "${p:-?} ${d:--}"
}
if [[ -z ${QUICK:-} ]]; then
  note "5a. pp 8K @ chunk 2048（最后一个 chunk 的 tok/s）"
  read -r a _ < <(speed g3_pp8k_hgn "$REF" 8k 1 GDEC_PREFILL_CHUNK=2048)
  read -r b _ < <(speed g3_pp8k_g2 "$BIN" 8k 1 GDEC_PREFILL_CHUNK=2048 GDEC_GGUF="$GG")
  read -r c _ < <(speed g3_pp8k_g3 "$BIN" 8k 1 GDEC_PREFILL_CHUNK=2048 "${G3[@]}")
  echo "hgn=$a  G2=$b  G3=$c tok/s"
  note "5b. pp 32K @ chunk 16384 + decode 128"
  read -r a ad < <(speed g3_pp32k_hgn "$REF" 32k 128)
  read -r b bd < <(speed g3_pp32k_g2 "$BIN" 32k 128 GDEC_GGUF="$GG")
  read -r c cd < <(speed g3_pp32k_g3 "$BIN" 32k 128 "${G3[@]}")
  echo "pp     hgn=$a  G2=$b  G3=$c tok/s"
  echo "decode hgn=$ad G2=$bd G3=$cd tok/s"
fi

echo
if (( fail )); then echo FAIL; exit 1; else echo PASS; fi
