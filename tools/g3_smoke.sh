#!/usr/bin/env bash
# G3.1 冒烟：512 token prefill PPL，G2（专家来自 GGUF）vs G3（+ dense/norm/embed 来自 GGUF）。
# 额外参数原样作为 G3 的 env，例如二分：
#   SKIP_G2=1 bash tools/g3_smoke.sh GDEC_GGUF_DENSE_FILTER=lm_head,embed GDEC_NOPREFILLBATCH=1
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
D=${GGDIR:-$HOME/App/llama.cpp/models/Qwen3.8-Flash-Next-UD-Q4_K_XL}
GG=$D/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf
MTP=$D/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf
BIN=${BIN:-build/gdec-gguf}
export PROBE_CAP_GB=${PROBE_CAP_GB:-100}
N=${N:-512}
TOK=logs/g3_tok$N.txt
tr -s ' ,\t' '\n' <"$HOME/ppbench/tok8192.txt" | grep -E '^[0-9]+$' | head -$N >"$TOK"
chk="$(bash start.sh --check 2>&1)" || { echo "$chk"; exit 1; }
mapfile -t PENV < <(sed -n 's/^ENV //p' <<<"$chk")
eval "C=($(sed -n 's/^CMD //p' <<<"$chk"))"
M=${C[1]}; O=${C[2]}; [[ "$O" == --* ]] && O=""
run_ppl() {
  local label=$1; shift
  ( for e in $(compgen -e | grep '^GDEC_'); do unset "$e"; done
    for e in "${PENV[@]}" GDEC_KVSNAP=0 GDEC_GGUF="$GG" "$@"; do export "$e"; done
    bash tools/run_capped.sh "$PROBE_CAP_GB" -- "$BIN" "$M" ${O:+"$O"} --tokens-file "$TOK" \
      --ppl --maxctx 4096 >"logs/$label.log" 2>&1 )
  echo "$label: $(grep -m1 '^ppl_summary' logs/$label.log)"
  grep -E '^gguf (dense|experts)' "logs/$label.log"
  grep -E 'hipError|Segmentation|Aborted|FATAL|what\(\)' "logs/$label.log" | head -5
}
[[ ${SKIP_G2:-} ]] || run_ppl g3s_g2
run_ppl g3s_g3 GDEC_GGUF_DENSE=1 GDEC_GGUF_MTP="$MTP" "$@"
echo SMOKE_DONE
