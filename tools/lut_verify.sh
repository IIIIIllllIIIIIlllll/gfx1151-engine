#!/usr/bin/env bash
# L2 验证：hgn q4cp 路由专家的 prefill 走 LUT 解码 F16 WMMA kernel（默认开启，
# GDEC_MOE_Q4W=0 关闭；src/gpu/parts/27_kernels_moe_lut.inc）。只用 hgn 权重
# （start_hgn.sh 生产配置），chunk <= 16384。一键跑完，最后打印 PASS / FAIL。
#   bash tools/lut_verify.sh
#     BIN=build/gdec-q4w  新二进制      REF=build/gdec  改动前提交编出的 hgn 二进制
#     ON=""               开启新路径的 env（默认开启；旧的 opt-in 版本用 ON="GDEC_MOE_Q4W=1" OFF=""）
#     OFF="GDEC_MOE_Q4W=0" 关闭新路径的 env
#     QUICK=1             KLD 只跑 16 chunk，跳过速度
# 检查项：
#   1. 开关关闭时 BIN 与 REF 逐位相同：1024 token prefill / 逐 token decode 的 mean_nll
#   2. 开关打开时 decode 路径不变：逐 token decode（GDEC_NOPREFILLBATCH=1）mean_nll 与 REF 逐位相同
#   3. KLD（BF16 基准 data/kld/bf16_c512.kld，64×512）：on 与 off 都测，on <= off + 0.003
#   4. 开关打开：1024 token prefill 与 decode PPL 相对差 < 1%
#   5. MTP 投机（8K prompt，256 token，gamma 3）commit/round on >= 0.95 × off
#   6. 速度：pp 8K @ chunk 2048 / pp 8K 单 chunk / pp 32K @ 16384 + decode 128；
#      pp32K on 必须 >= off × 1.05（其余只报数）
# 每一步之后检查内核日志的 svm_range_cpu_invalidate_pagetables（出现即停止，FAIL）。
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
BIN=${BIN:-build/gdec-q4w}
REF=${REF:-build/gdec}
read -r -a ONV <<<"${ON-}"
read -r -a OFFV <<<"${OFF-GDEC_MOE_Q4W=0}"
export PROBE_CAP_GB=${PROBE_CAP_GB:-100}
KREF=data/kld/bf16_c512.kld
TOKSRC=$HOME/ppbench/tok8192.txt
fail=0
note() { echo; echo "== $*"; }
bad() { echo "FAIL: $*" >&2; fail=1; }

for f in "$BIN" "$REF" "$KREF" "$TOKSRC" "$HOME/ppbench/tok32768.txt"; do
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

# ---- 1/2. 逐位检查 -----------------------------------------------------------
TOK=logs/lut_tok1024.txt
tr -s ' ,\t' '\n' <"$TOKSRC" | grep -E '^[0-9]+$' | head -1024 >"$TOK"
chk="$(bash start_hgn.sh --check 2>&1)" || { echo "$chk"; echo FAIL; exit 1; }
mapfile -t PENV < <(sed -n 's/^ENV //p' <<<"$chk")
eval "C=($(sed -n 's/^CMD //p' <<<"$chk"))"
WARGS=()
for a in "${C[@]:1}"; do [[ $a == --* ]] && break; WARGS+=("$a"); done
echo "生产 env: ${PENV[*]}"
echo "权重: ${WARGS[*]}"
run_ppl() {  # label bin extra-env...  -> "ppl mean_nll"
  local label=$1 bin=$2; shift 2
  ( for e in $(compgen -e | grep '^GDEC_'); do unset "$e"; done
    for e in "${PENV[@]}" GDEC_KVSNAP=0 "$@"; do export "$e"; done
    bash tools/run_capped.sh "$PROBE_CAP_GB" -- "$bin" "${WARGS[@]}" --tokens-file "$TOK" \
      --ppl --maxctx 4096 >"logs/$label.log" 2>&1 )
  grep -qE 'hipError|Segmentation|Aborted|FATAL|what\(\)' "logs/$label.log" && bad "$label 崩溃（logs/$label.log）"
  sed -n 's/^ppl_summary.*mean_nll=\([0-9.]*\).*ppl=\([0-9.]*\).*/\2 \1/p' "logs/$label.log"
}
note "1. 开关关闭：$BIN 与 $REF 逐位相同"
read -r _ rnb < <(run_ppl lut_ref_prefill "$REF"); svm_check
read -r _ nnb < <(run_ppl lut_off_prefill "$BIN" "${OFFV[@]}"); svm_check
read -r _ rnd < <(run_ppl lut_ref_decode "$REF" GDEC_NOPREFILLBATCH=1); svm_check
read -r _ nnd < <(run_ppl lut_off_decode "$BIN" "${OFFV[@]}" GDEC_NOPREFILLBATCH=1); svm_check
echo "prefill mean_nll ref=${rnb:-?} off=${nnb:-?}   decode ref=${rnd:-?} off=${nnd:-?}"
if [[ -n "${rnb:-}" && "${rnb:-}" == "${nnb:-}" && -n "${rnd:-}" && "${rnd:-}" == "${nnd:-}" ]]; then
  echo "逐位相同  OK"
else bad "开关关闭时结果与 $REF 不同"; fi

note "2. 开关打开：逐 token decode 不变"
read -r pp_d ond < <(run_ppl lut_on_decode "$BIN" "${ONV[@]}" GDEC_NOPREFILLBATCH=1); svm_check
echo "decode mean_nll ref=${rnd:-?} on=${ond:-?}"
if [[ -n "${ond:-}" && "${ond:-}" == "${rnd:-}" ]]; then echo "逐位相同  OK"
else bad "开关打开后 decode 结果变了"; fi

# ---- 3. KLD ------------------------------------------------------------------
NCH=$([[ -n ${QUICK:-} ]] && echo 16 || echo 64)
note "3. KLD vs BF16（$NCH chunk × 512）"
kld_run() {  # label extra... -> "kld top1"
  local label=$1; shift
  local out
  out=$(BIN=$BIN CHUNKS=$NCH bash tools/kld_engine.sh "$KREF" "$label" "$@" 2>&1)
  grep -qE 'hipError|Segmentation|Aborted|FATAL' "logs/kld_$label.log" && bad "$label 崩溃（logs/kld_$label.log）"
  echo "$(sed -n 's/^Mean    KLD: *\([0-9.]*\).*/\1/p' <<<"$out" | head -1) $(sed -n 's/^Same top p: *\([0-9.]*\).*/\1/p' <<<"$out" | head -1)"
}
read -r k_off t_off < <(kld_run lut_off "${OFFV[@]}"); svm_check
read -r k_on t_on < <(kld_run lut_on "${ONV[@]}"); svm_check
echo "KLD off=${k_off:-?} (top1 ${t_off:-?}%)  on=${k_on:-?} (top1 ${t_on:-?}%)   [hgn 历史 0.1631 / 86.61%]"
if [[ -z "${k_off:-}" || -z "${k_on:-}" ]]; then bad "KLD 没有输出"
elif awk -v a="$k_on" -v b="$k_off" 'BEGIN{exit !(a <= b + 0.003)}'; then echo "OK"
else bad "开关打开后 KLD 变差超过 0.003"; fi

# ---- 4. prefill vs decode PPL --------------------------------------------------
note "4. 开关打开：1024 token prefill vs decode PPL"
read -r pp_b onb < <(run_ppl lut_on_prefill "$BIN" "${ONV[@]}"); svm_check
echo "PPL prefill(on)=${pp_b:-?} decode=${pp_d:-?}   prefill mean_nll off=${nnb:-?} on=${onb:-?}"
if [[ -z "${pp_b:-}" || -z "${pp_d:-}" ]]; then bad "PPL 没有输出"
elif awk -v a="$pp_b" -v b="$pp_d" 'BEGIN{d=(a-b)/a; if(d<0)d=-d; exit !(d < 0.01)}'; then echo "相对差 < 1%  OK"
else bad "prefill 与 decode PPL 相差 >= 1%"; fi
if grep -q 'q4cp LUT WMMA' logs/lut_on_prefill.log && ! grep -q 'q4cp LUT WMMA' logs/lut_off_prefill.log; then
  echo "日志：on 走 q4cp LUT WMMA，off 不走  OK"
elif ! grep -q 'q4cp LUT WMMA' logs/lut_off_prefill.log && [[ "${ON-}" == *GDEC_MOE_Q4W=1* ]]; then
  echo "日志：旧 opt-in 版本（没有日志行），跳过"
else bad "q4cp LUT WMMA 日志行与开关不符"; fi

# ---- 5. MTP 投机 ---------------------------------------------------------------
spec() {  # label extra... -> "commit/round tok/s"
  local label=$1; shift
  BIN=$BIN SPEC=256 GAMMA=3 bash tools/pp_prod.sh 8k "$label" "$@" >/dev/null 2>&1
  grep -qE 'hipError|Segmentation|Aborted|FATAL' "logs/$label.log" && bad "$label 崩溃（logs/$label.log）"
  tr '\r' '\n' <"logs/$label.log" | grep -ao 'spec: [0-9]* tokens in [0-9.]* s = [0-9.]* tok/s | rounds=[0-9]* commit/round=[0-9.]*' |
    tail -1 | sed 's/.*= \([0-9.]*\) tok\/s.*commit\/round=\([0-9.]*\).*/\2 \1/'
}
note "5. MTP 投机：8K prompt，256 token，gamma 3"
read -r c_off s_off < <(spec lut_spec_off "${OFFV[@]}"); svm_check
read -r c_on s_on < <(spec lut_spec_on "${ONV[@]}"); svm_check
echo "commit/round off=${c_off:-?} on=${c_on:-?}   tok/s off=${s_off:-?} on=${s_on:-?}"
if [[ -z "${c_off:-}" || -z "${c_on:-}" ]]; then bad "spec 没有输出"
elif awk -v a="$c_on" -v b="$c_off" 'BEGIN{exit !(a >= 0.95*b)}'; then echo "OK"
else bad "commit/round 低于 off 的 0.95 倍"; fi

# ---- 6. 速度 -------------------------------------------------------------------
speed() {  # label tok gen extra... -> "prefill_tok/s decode_tok/s"
  local label=$1 tok=$2 gen=$3; shift 3
  BIN=$BIN GEN=$gen bash tools/pp_prod.sh "$tok" "$label" "$@" >/dev/null 2>&1
  local p d
  p=$(tr '\r' '\n' <"logs/$label.log" | sed -n 's/.*prefill: \([0-9]*\) tokens in \([0-9.]*\) s = \([0-9.]*\) tok\/s.*/\3/p' | tail -1)
  d=$(grep -ao 'decode: [0-9]* tokens in [0-9]* ms = [0-9.]* tok/s' "logs/$label.log" | tail -1 | sed 's/.*= \([0-9.]*\) tok.*/\1/')
  grep -qE 'hipError|Segmentation|Aborted|FATAL' "logs/$label.log" && bad "$label 崩溃（logs/$label.log）"
  echo "${p:-?} ${d:--}"
}
if [[ -z ${QUICK:-} ]]; then
  note "6a. pp 8K @ chunk 2048"
  read -r a _ < <(speed lut_pp8k2k_off 8k 1 GDEC_PREFILL_CHUNK=2048 "${OFFV[@]}"); svm_check
  read -r b _ < <(speed lut_pp8k2k_on 8k 1 GDEC_PREFILL_CHUNK=2048 "${ONV[@]}"); svm_check
  echo "off=$a  on=$b tok/s   [hgn 历史 786，GGUF 1206]"
  note "6b. pp 8K 单 chunk"
  read -r a _ < <(speed lut_pp8k_off 8k 1 GDEC_PREFILL_CHUNK=8192 "${OFFV[@]}"); svm_check
  read -r b _ < <(speed lut_pp8k_on 8k 1 GDEC_PREFILL_CHUNK=8192 "${ONV[@]}"); svm_check
  echo "off=$a  on=$b tok/s   [hgn 历史 1056，GGUF 1370]"
  note "6c. pp 32K @ chunk 16384 + decode 128"
  read -r a ad < <(speed lut_pp32k_off 32k 128 GDEC_PREFILL_CHUNK=16384 "${OFFV[@]}"); svm_check
  read -r b bd < <(speed lut_pp32k_on 32k 128 GDEC_PREFILL_CHUNK=16384 "${ONV[@]}"); svm_check
  echo "pp     off=$a  on=$b tok/s   [hgn 历史 1250，GGUF 1435]"
  echo "decode off=$ad on=$bd tok/s"
  if [[ "$a" == "?" || "$b" == "?" ]]; then bad "pp32K 没有输出"
  elif awk -v a="$b" -v b="$a" 'BEGIN{exit !(a >= 1.05*b)}'; then echo "OK"
  else bad "pp32K 提速不足 5%"; fi
fi

echo
if (( fail )); then echo FAIL; exit 1; else echo PASS; fi
