#!/usr/bin/env bash
# #78 验证：8-bit dense overlay（tools/hgn_hq.py 从 BF16 safetensors 生成）替换生产
# w4b.overlay.hgn；可选 NEWBASE 同时替换基座（tools/hgn_q4i.py 的 imatrix 路由专家版）。
# mtp.hgn 不变。一键跑完打印 PASS / FAIL。
#   bash tools/hq_verify.sh
#     NEW=~/Models/hq/qwen38-flash-next-w4b.overlay-q8.hgn   新 overlay
#     NEWBASE=~/Models/hq/qwen38-flash-next-w4b-imat.hgn     新基座（默认不换）
#     KMAX=0.075           KLD 上限（设了 NEWBASE 时默认 0.060）
#     BIN=build/gdec-hq2   新二进制      REF=build/gdec-q4w  改动前提交（fdf805a）编出的二进制
#     QUICK=1              KLD 只跑 16 chunk，跳过速度
# 检查项：
#   1. 旧 overlay：BIN 与 REF 逐位相同（1024 token prefill / 逐 token decode mean_nll）
#   2. 新 overlay：1024 token prefill 与 decode PPL 相对差 < 1%；日志有 few-row bf16 gemv 行
#   3. KLD（BF16 基准 64×512）：新 <= KMAX 且比旧低 >= 0.08
#   4. MTP 投机（8K prompt，256 token，gamma 3）：commit/round 新 >= 0.95 × 旧
#   5. 速度：pp 8K @ 2048 / pp 32K @ 16384 + decode 128（只报数；decode 新 >= 0.7 × 旧）
#   6. weight arena：新 - 旧 <= 2.6 GiB（Windows 95 GiB 上限的余量）
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
NEW=${NEW:-$HOME/Models/hq/qwen38-flash-next-w4b.overlay-q8.hgn}
NEWBASE=${NEWBASE:-}
KMAX=${KMAX:-$([[ -n $NEWBASE ]] && echo 0.060 || echo 0.075)}
BIN=${BIN:-build/gdec-hq2}
REF=${REF:-build/gdec-q4w}
export PROBE_CAP_GB=${PROBE_CAP_GB:-100}
KREF=data/kld/bf16_c512.kld
TOKSRC=$HOME/ppbench/tok8192.txt
fail=0
note() { echo; echo "== $*"; }
bad() { echo "FAIL: $*" >&2; fail=1; }

for f in "$NEW" ${NEWBASE:+"$NEWBASE"} "$BIN" "$REF" "$KREF" "$TOKSRC" "$HOME/ppbench/tok32768.txt"; do
  [[ -e "$f" ]] || { echo "缺少 $f"; echo FAIL; exit 1; }
done
if pgrep -af '(^|/)(gdec[^/[:space:]]*|flash_serve|serve_api\.py|llama-server|llama-perplexity|llama-cli)([[:space:]]|$)' >/dev/null; then
  echo "GPU 上已有引擎/llama.cpp 在跑，请先停掉"; echo FAIL; exit 1
fi
mkdir -p logs
svm0=$(journalctl -k -b 2>/dev/null | grep -c svm_range_cpu_invalidate_pagetables)
svm_check() {
  local n
  n=$(journalctl -k -b 2>/dev/null | grep -c svm_range_cpu_invalidate_pagetables)
  if (( n > svm0 )); then
    echo "内核日志出现 svm_range_cpu_invalidate_pagetables（$svm0 -> $n），停止" >&2
    echo FAIL; exit 1
  fi
  free -g | awk 'NR==2{print "   [mem] used " $3 " GiB, available " $7 " GiB"}'
}

TOK=logs/hq_tok1024.txt
tr -s ' ,\t' '\n' <"$TOKSRC" | grep -E '^[0-9]+$' | head -1024 >"$TOK"
chk="$(bash start_hgn.sh --check 2>&1)" || { echo "$chk"; echo FAIL; exit 1; }
mapfile -t PENV < <(sed -n 's/^ENV //p' <<<"$chk")
eval "C=($(sed -n 's/^CMD //p' <<<"$chk"))"
WOLD=()
for a in "${C[@]:1}"; do [[ $a == --* ]] && break; WOLD+=("$a"); done
(( ${#WOLD[@]} >= 2 )) || { echo "生产命令里没有 overlay: ${WOLD[*]}"; echo FAIL; exit 1; }
WNEW=("${WOLD[@]}"); WNEW[1]=$NEW
[[ -n $NEWBASE ]] && WNEW[0]=$NEWBASE
# pp_prod.sh / kld_engine.sh 用的覆盖变量
wenv() { [[ $1 == new ]] && echo "OVERLAY=$NEW MODEL=${WNEW[0]}" || echo "OVERLAY=${WOLD[1]} MODEL=${WOLD[0]}"; }
echo "旧权重: ${WOLD[*]}"
echo "新权重: ${WNEW[*]}"
run_ppl() {  # label bin old|new extra-env...  -> "ppl mean_nll"
  local label=$1 bin=$2 which=$3; shift 3
  local -a W=("${WOLD[@]}"); [[ $which == new ]] && W=("${WNEW[@]}")
  ( for e in $(compgen -e | grep '^GDEC_'); do unset "$e"; done
    for e in "${PENV[@]}" GDEC_KVSNAP=0 "$@"; do export "$e"; done
    bash tools/run_capped.sh "$PROBE_CAP_GB" -- "$bin" "${W[@]}" --tokens-file "$TOK" \
      --ppl --maxctx 4096 >"logs/$label.log" 2>&1 )
  grep -qE 'hipError|Segmentation|Aborted|FATAL|what\(\)' "logs/$label.log" && bad "$label 崩溃（logs/$label.log）"
  sed -n 's/^ppl_summary.*mean_nll=\([0-9.]*\).*ppl=\([0-9.]*\).*/\2 \1/p' "logs/$label.log"
}
arena() { sed -n 's/^weight arena: \([0-9.]*\) GiB.*/\1/p' "$1" | head -1; }

# ---- 1. 旧 overlay 逐位 -------------------------------------------------------
note "1. 旧 overlay：$BIN 与 $REF 逐位相同"
read -r _ rnb < <(run_ppl hq_ref_prefill "$REF" old); svm_check
read -r _ onb < <(run_ppl hq_old_prefill "$BIN" old); svm_check
read -r _ rnd < <(run_ppl hq_ref_decode "$REF" old GDEC_NOPREFILLBATCH=1); svm_check
read -r _ ond < <(run_ppl hq_old_decode "$BIN" old GDEC_NOPREFILLBATCH=1); svm_check
echo "prefill mean_nll ref=${rnb:-?} bin=${onb:-?}   decode ref=${rnd:-?} bin=${ond:-?}"
if [[ -n "${rnb:-}" && "${rnb:-}" == "${onb:-}" && -n "${rnd:-}" && "${rnd:-}" == "${ond:-}" ]]; then
  echo "逐位相同  OK"
else bad "旧 overlay 结果与 $REF 不同"; fi
grep -q 'few-row bf16 gemv on' logs/hq_old_prefill.log && bad "旧 overlay 不应打开 few-row bf16 gemv"

# ---- 2. 新 overlay prefill vs decode -----------------------------------------
note "2. 新 overlay：1024 token prefill vs decode PPL"
read -r np nnb < <(run_ppl hq_new_prefill "$BIN" new); svm_check
read -r nd nnd < <(run_ppl hq_new_decode "$BIN" new GDEC_NOPREFILLBATCH=1); svm_check
read -r op _ < <(sed -n 's/^ppl_summary.*mean_nll=\([0-9.]*\).*ppl=\([0-9.]*\).*/\2 \1/p' logs/hq_old_prefill.log)
echo "PPL 新 prefill=${np:-?} decode=${nd:-?}   旧 prefill=${op:-?}"
if [[ -z "${np:-}" || -z "${nd:-}" ]]; then bad "PPL 没有输出"
elif awk -v a="$np" -v b="$nd" 'BEGIN{d=(a-b)/a; if(d<0)d=-d; exit !(d < 0.01)}'; then echo "相对差 < 1%  OK"
else bad "prefill 与 decode PPL 相差 >= 1%"; fi
grep -q 'few-row bf16 gemv on' logs/hq_new_prefill.log && echo "日志：few-row bf16 gemv on  OK" || bad "新 overlay 没打开 few-row bf16 gemv"

# ---- 3. KLD ------------------------------------------------------------------
NCH=$([[ -n ${QUICK:-} ]] && echo 16 || echo 64)
note "3. KLD vs BF16（$NCH chunk × 512）"
kld_run() {  # label old|new -> "kld top1 ppl"
  local label=$1 out
  out=$(env BIN=$BIN CHUNKS=$NCH $(wenv "$2") bash tools/kld_engine.sh "$KREF" "$label" 2>&1)
  grep -qE 'hipError|Segmentation|Aborted|FATAL' "logs/kld_$label.log" && bad "$label 崩溃（logs/kld_$label.log）"
  echo "$(sed -n 's/^Mean    KLD: *\([0-9.]*\).*/\1/p' <<<"$out" | head -1)" \
       "$(sed -n 's/^Same top p: *\([0-9.]*\).*/\1/p' <<<"$out" | head -1)" \
       "$(sed -n 's/^Mean PPL(Q) *: *\([0-9.]*\).*/\1/p' <<<"$out" | head -1)"
}
read -r k_old t_old p_old < <(kld_run hq_kld_old old); svm_check
read -r k_new t_new p_new < <(kld_run hq_kld_new new); svm_check
echo "KLD 旧=${k_old:-?} (top1 ${t_old:-?}%, PPL ${p_old:-?})  新=${k_new:-?} (top1 ${t_new:-?}%, PPL ${p_new:-?})"
echo "     [参考：GGUF Q4_K_XL 纯 0.0511，llama.cpp 0.049]"
if [[ -z "${k_old:-}" || -z "${k_new:-}" ]]; then bad "KLD 没有输出"
elif awk -v a="$k_new" -v b="$k_old" -v m="$KMAX" 'BEGIN{exit !(a <= m && a <= b - 0.08)}'; then echo "OK"
else bad "新权重 KLD 不达标（要求 <= $KMAX 且比旧低 >= 0.08）"; fi

# ---- 4. MTP 投机 ---------------------------------------------------------------
spec() {  # label old|new -> "commit/round tok/s"
  local label=$1
  env BIN=$BIN $(wenv "$2") SPEC=256 GAMMA=3 bash tools/pp_prod.sh 8k "$label" >/dev/null 2>&1
  grep -qE 'hipError|Segmentation|Aborted|FATAL' "logs/$label.log" && bad "$label 崩溃（logs/$label.log）"
  tr '\r' '\n' <"logs/$label.log" | grep -ao 'spec: [0-9]* tokens in [0-9.]* s = [0-9.]* tok/s | rounds=[0-9]* commit/round=[0-9.]*' |
    tail -1 | sed 's/.*= \([0-9.]*\) tok\/s.*commit\/round=\([0-9.]*\).*/\2 \1/'
}
note "4. MTP 投机：8K prompt，256 token，gamma 3"
read -r c_old s_old < <(spec hq_spec_old old); svm_check
read -r c_new s_new < <(spec hq_spec_new new); svm_check
echo "commit/round 旧=${c_old:-?} 新=${c_new:-?}   tok/s 旧=${s_old:-?} 新=${s_new:-?}"
if [[ -z "${c_old:-}" || -z "${c_new:-}" ]]; then bad "spec 没有输出"
elif awk -v a="$c_new" -v b="$c_old" 'BEGIN{exit !(a >= 0.95*b)}'; then echo "OK"
else bad "commit/round 低于旧的 0.95 倍"; fi

# ---- 5. 速度 -------------------------------------------------------------------
speed() {  # label old|new tok gen extra... -> "prefill_tok/s decode_tok/s"
  local label=$1 which=$2 tok=$3 gen=$4; shift 4
  env BIN=$BIN $(wenv "$which") GEN=$gen bash tools/pp_prod.sh "$tok" "$label" "$@" >/dev/null 2>&1
  local p d
  p=$(tr '\r' '\n' <"logs/$label.log" | sed -n 's/.*prefill: \([0-9]*\) tokens in \([0-9.]*\) s = \([0-9.]*\) tok\/s.*/\3/p' | tail -1)
  d=$(grep -ao 'decode: [0-9]* tokens in [0-9]* ms = [0-9.]* tok/s' "logs/$label.log" | tail -1 | sed 's/.*= \([0-9.]*\) tok.*/\1/')
  grep -qE 'hipError|Segmentation|Aborted|FATAL' "logs/$label.log" && bad "$label 崩溃（logs/$label.log）"
  echo "${p:-?} ${d:--}"
}
if [[ -z ${QUICK:-} ]]; then
  note "5a. pp 8K @ chunk 2048"
  read -r a _ < <(speed hq_pp8k_old old 8k 1 GDEC_PREFILL_CHUNK=2048); svm_check
  read -r b _ < <(speed hq_pp8k_new new 8k 1 GDEC_PREFILL_CHUNK=2048); svm_check
  echo "旧=$a  新=$b tok/s"
  note "5b. pp 32K @ chunk 16384 + decode 128"
  read -r a ad < <(speed hq_pp32k_old old 32k 128 GDEC_PREFILL_CHUNK=16384); svm_check
  read -r b bd < <(speed hq_pp32k_new new 32k 128 GDEC_PREFILL_CHUNK=16384); svm_check
  echo "pp     旧=$a  新=$b tok/s"
  echo "decode 旧=$ad 新=$bd tok/s   [GGUF 纯 ~25]"
  if [[ "$ad" == "-" || "$bd" == "-" ]]; then bad "decode 没有输出"
  elif awk -v a="$bd" -v b="$ad" 'BEGIN{exit !(a >= 0.7*b)}'; then echo "OK"
  else bad "decode 低于旧的 0.7 倍"; fi
fi

# ---- 6. 显存 -------------------------------------------------------------------
note "6. weight arena"
ao=$(arena logs/hq_old_prefill.log); an=$(arena logs/hq_new_prefill.log)
echo "旧=${ao:-?} GiB  新=${an:-?} GiB   [Windows hgn 256K 估算 91.03 GiB，上限 95]"
if [[ -z "${ao:-}" || -z "${an:-}" ]]; then bad "没有 weight arena 日志行"
elif awk -v a="$an" -v b="$ao" 'BEGIN{exit !(a - b <= 2.6)}'; then
  awk -v a="$an" -v b="$ao" 'BEGIN{printf "增加 %.1f GiB → Windows 估算 ~%.1f GiB  OK\n", a-b, 91.03+a-b}'
else bad "weight arena 增加超过 2.6 GiB"; fi

echo
if (( fail )); then echo FAIL; exit 1; else echo PASS; fi
