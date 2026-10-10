#!/usr/bin/env bash
# Long-context PPL triple: same 2048-token target text scored
#  (a) factor2 deep (position ~350K)   (b) factor2 shallow   (c) factor1 deep (expect blow-up)
# Engine exits(1) on any non-finite NLL, so a completed run means finite losses.
set -uo pipefail
cd /home/mark/Workspace/gfx1151-engine-yarn
chk="$(bash start_hgn.sh --check 2>&1)" || { echo "$chk"; exit 1; }
mapfile -t PENV < <(sed -n 's/^ENV //p' <<<"$chk")
CMDLINE="$(sed -n 's/^CMD //p' <<<"$chk")"
eval "C=($CMDLINE)"
M=${C[1]}; O=${C[2]}

run_ppl() { # label extra-env... ids maxctx
  local label=$1 ids=$2 maxctx=$3; shift 3
  for e in $(compgen -e | grep '^QWENOX_'); do unset "$e"; done
  for e in "${PENV[@]}" "$@"; do export "$e"; done
  export QWENOX_KVSNAP=0
  echo "[$label] ids=$ids maxctx=$maxctx extra=$*"
  bash tools/run_capped.sh 86 -- build/qwenox-engine "$M" "$O" \
    --tokens-file "$ids" --ppl --maxctx "$maxctx" > "logs/ppl_${label}.raw" 2> "logs/ppl_${label}.err"
  echo "[$label] rc=$?"
}

run_ppl f2_long  /tmp/ppl_long_ids.txt  524288 QWENOX_ROPE_FACTOR=2 QWENOX_ROPE_ORIGINAL_CTX=262144 QWENOX_ROPE_BETA_FAST=32 QWENOX_ROPE_BETA_SLOW=1
run_ppl f2_short /tmp/ppl_short_ids.txt 8192   QWENOX_ROPE_FACTOR=2 QWENOX_ROPE_ORIGINAL_CTX=262144 QWENOX_ROPE_BETA_FAST=32 QWENOX_ROPE_BETA_SLOW=1
run_ppl f1_long  /tmp/ppl_long_ids.txt  524288 QWENOX_ROPE_FACTOR=1
echo PPL_TRIPLE_DONE
