#!/usr/bin/env bash
# 生产环境变量下的离线 prefill 基准（env 取自 start_hgn.sh --check，含 KV_PAGED 等）。
#     LAUNCHER=start_gguf.sh  改取 GGUF 启动器的权重与环境
#   bash tools/pp_prod.sh <tokfile|32k|64k|...> [标签] [K=V ...]
#     BIN=build/gdec.base     换二进制做 A/B（默认 build/gdec）
#     KTRACE=1                包 rocprofv3 --kernel-trace --stats，打印 kernel 耗时前 25
#     MAXCTX=N                默认 token 数 + 8192
#     UNSET="GDEC_A GDEC_B"   从生产 env 中去掉这些变量（开关类变量只看存在性）
#     GEN=N                   生成 token 数（默认 1；日志末尾 ids: 行可做 A/B 比对）
#     SPEC=N [GAMMA=3]        改用 MTP 投机生成 --spec-gen N（看 MTP 接受率）
# 长度简写从 data/ppbench/tok<N>.txt 或 ~/ppbench/tok<N>.txt 找。
# 输出：logs/<标签>.log；打印 phase / prefill / prof 行与总耗时。
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
[[ $# -ge 1 ]] || { sed -n '2,11p' "$0"; exit 1; }
arg=$1; shift
case "$arg" in
  *[kK]) n=$(( ${arg%[kK]} * 1024 ))
         TOK=""
         for d in data/ppbench "$HOME/ppbench"; do [[ -f "$d/tok$n.txt" ]] && TOK="$d/tok$n.txt" && break; done
         [[ -n "$TOK" ]] || { echo "找不到 tok$n.txt"; exit 1; } ;;
  *) TOK=$arg ;;
esac
LABEL="${1:-pp_$(basename "$TOK" .txt)_$(date +%H%M%S)}"; [[ $# -ge 1 ]] && shift
NTOK=$(wc -w <"$TOK")
MAXCTX=${MAXCTX:-$(( (NTOK + 8192 + 255) / 256 * 256 ))}
BIN=${BIN:-build/gdec}

for f in start_hgn.sh start_gguf.sh tools/serve_common.sh service.conf; do grep -q $'\r' "$f" && sed -i 's/\r$//' "$f"; done
LAUNCHER=${LAUNCHER:-start_hgn.sh}
chk="$(bash "$LAUNCHER" --check 2>&1)" || { echo "$chk"; echo "$LAUNCHER --check 失败"; exit 1; }
mapfile -t PENV < <(sed -n 's/^ENV //p' <<<"$chk")
CMDLINE="$(sed -n 's/^CMD //p' <<<"$chk")"
eval "C=($CMDLINE)"
MODEL=${C[1]}; OVERLAY=${C[2]}
[[ "$OVERLAY" == --* ]] && OVERLAY=""

for e in $(compgen -e | grep '^GDEC_'); do unset "$e"; done
for e in "${PENV[@]}" GDEC_PROF=1 GDEC_PHASE=1 "$@"; do export "$e"; done
export GDEC_KVSNAP=0
for e in ${UNSET:-}; do unset "$e"; done
mkdir -p logs
LOG="logs/$LABEL.log"
GENARG=(--gen "${GEN:-1}")
[[ -n "${SPEC:-}" ]] && GENARG=(--spec-gen "$SPEC" --gamma "${GAMMA:-3}")
RUN=("$BIN" "$MODEL" ${OVERLAY:+"$OVERLAY"} --tokens-file "$TOK" "${GENARG[@]}" --maxctx "$MAXCTX")
if [[ "${KTRACE:-0}" == 1 ]]; then
  KD="logs/$LABEL.ktrace"; rm -rf "$KD"
  RUN=(rocprofv3 --kernel-trace --stats -d "$KD" -o run -- "${RUN[@]}")
fi
echo "[$LABEL] tok=$TOK ($NTOK) maxctx=$MAXCTX bin=$BIN extra=${*:-无} unset=${UNSET:-无}"
t0=$SECONDS
bash tools/run_capped.sh "${PROBE_CAP_GB:-86}" -- "${RUN[@]}" >"$LOG" 2>&1
rc=$?
tr '\r' '\n' <"$LOG" | grep -E '^phase |prefill: |^prof: |hipError|Segmentation|Aborted|FATAL' | cut -c1-200
echo "[$LABEL] rc=$rc wall=$((SECONDS - t0))s log=$LOG"
[[ "${KTRACE:-0}" == 1 ]] && python3 tools/ktrace_top.py "$KD" 30 --group
exit $rc
