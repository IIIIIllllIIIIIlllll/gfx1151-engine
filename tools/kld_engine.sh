#!/usr/bin/env bash
# 引擎侧 KLD：生产环境变量（start.sh --check）下读 llama.cpp 的 logits 基准文件。
#   bash tools/kld_engine.sh <ref.kld> [标签] [K=V ...]
#     CHUNKS=N            只算前 N 个 chunk（默认文件里全部）
#     SAVE=out.kld        同时把引擎自己的 logits 写成 llama.cpp 格式（可给 llama 读）
#     BIN=build/gdec      换二进制
#     MODEL= OVERLAY=     覆盖 start.sh 的权重；OVERLAY=none 去掉 overlay
#     UNSET="GDEC_A ..."  从生产 env 去掉这些变量
#   bash tools/kld_engine.sh --tokens IDS.txt <out.kld> [标签]   无基准：只写引擎 logits 文件
#     CTX=512 CHUNKS=N
# 输出：logs/kld_<标签>.log（stdout 为 llama.cpp 同格式摘要 + kld_summary 行）。
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
[[ $# -ge 1 ]] || { sed -n '2,12p' "$0"; exit 1; }
KARGS=()
if [[ $1 == --tokens ]]; then
  [[ $# -ge 3 ]] || { echo "用法: --tokens IDS.txt out.kld [标签]"; exit 1; }
  KARGS=(--tokens-file "$2" --kld-save "$3" --kld-ctx "${CTX:-512}")
  LABEL=${4:-eng_save_$(date +%H%M%S)}
  shift $(( $# >= 4 ? 4 : 3 ))
else
  [[ -f "$1" ]] || { echo "找不到基准文件: $1"; exit 1; }
  KARGS=(--kld-base "$1")
  [[ -n "${SAVE:-}" ]] && KARGS+=(--kld-save "$SAVE")
  LABEL=${2:-eng_$(basename "$1" .kld)_$(date +%H%M%S)}
  shift $(( $# >= 2 ? 2 : 1 ))
fi
[[ -n "${CHUNKS:-}" ]] && KARGS+=(--kld-chunks "$CHUNKS")
BIN=${BIN:-build/gdec}

for f in start.sh service.conf; do grep -q $'\r' "$f" && sed -i 's/\r$//' "$f"; done
chk="$(bash start.sh --check 2>&1)" || { echo "$chk"; echo "start.sh --check 失败"; exit 1; }
mapfile -t PENV < <(sed -n 's/^ENV //p' <<<"$chk")
CMDLINE="$(sed -n 's/^CMD //p' <<<"$chk")"
eval "C=($CMDLINE)"
M=${MODEL:-${C[1]}}
O=${OVERLAY-${C[2]}}
[[ "$O" == --* || "$O" == none ]] && O=""

for e in $(compgen -e | grep '^GDEC_'); do unset "$e"; done
for e in "${PENV[@]}" "$@"; do export "$e"; done
export GDEC_KVSNAP=0
for e in ${UNSET:-}; do unset "$e"; done
mkdir -p logs
LOG="logs/kld_$LABEL.log"
echo "[$LABEL] model=$M overlay=${O:-无} bin=$BIN args=${KARGS[*]} extra=${*:-无}"
t0=$SECONDS
bash tools/run_capped.sh "${PROBE_CAP_GB:-86}" -- "$BIN" "$M" ${O:+"$O"} "${KARGS[@]}" \
  --maxctx "${MAXCTX:-4096}" >"$LOG" 2>&1
rc=$?
tr '\r' '\n' <"$LOG" | grep -E '^(Mean PPL|Cor\(|Mean ln|Mean    KLD|Maximum KLD|99\.9%   KLD|Median  KLD|RMS Δp|Same top p|kld_summary|kld_saved)|^kld: |hipError|Segmentation|Aborted|FATAL' | cut -c1-200
echo "[$LABEL] rc=$rc wall=$((SECONDS - t0))s log=$LOG"
exit $rc
