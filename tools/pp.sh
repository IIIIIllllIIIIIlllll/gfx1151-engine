#!/usr/bin/env bash
# pp.sh — PP 性能测试（直跑，验证过的满速命令）
# 用法: ./tools/pp.sh <长度> [标签]
#   ./tools/pp.sh 32k            # 32K prefill
#   ./tools/pp.sh 128k my128     # 128K，自定义日志名
# 长度支持 2k / 32k / 128k（也可写 2048 / 32768 / 131072），
# 自动匹配 data/ppbench/tok<N>.txt 并计算 maxctx（长度 + 8K 余量）。
# 输出全部进日志，跑完自动打印 prefill 行。终端无洪水。
set -u

if [[ $# -lt 1 || "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    sed -n '2,8p' "$0"
    exit 1
fi

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

case "$1" in
    *[!0-9kK]*) echo "长度格式不对: $1（支持 2k / 32k / 128k 或纯数字）" >&2; exit 1 ;;
    *[kK])     LEN=$(( ${1%[kK]} * 1024 )) ;;
    *)         LEN=$1 ;;
esac
LABEL="${2:-pp_${LEN}_$(date +%H%M%S)}"

TOK="data/qsa-oracle/${LEN}.tokens"

if [[ ! -f "$TOK" ]]; then
    echo "没有对应长度的 token 文件: $TOK" >&2
    echo "可用的长度: $(ls data/ppbench/tok*.txt 2>/dev/null | sed 's|.*tok||;s|\.txt||' | tr '\n' ' ')" >&2
    exit 1
fi

MAXCTX=$((LEN + 8192))

LOG="$ROOT/logs/${LABEL}.log"
env GDEC_QSA_KV_BF16=1 GDEC_QSA_WMMA=1 GDEC_QSA_WMMA_BTV=1 \
    GDEC_MOE_LT=1 GDEC_MOE_LT_BF16=1 GDEC_GR_BF16=1 \
    GDEC_GDN_STREAM=1 GDEC_GDN_WAVE=1 GDEC_NOWARMUP=1 \
    GDEC_PREFILL_CHUNK=16384 GDEC_GEMM_WMMA=1 GDEC_GDN_FUSED=1 \
    GDEC_INDEX_FUSED2=1 GDEC_PP_MOE_OUT=1 GDEC_INDEX_STREAM_SELECT=1 \
    GDEC_KVSNAP=1 GDEC_KVSNAP_MAX_GB=20 GDEC_PROF=1 GDEC_PHASE=1 \
    build/gdec models/qwen38-flash-next-w4b.hgn models/qwen38-flash-next-w4b.overlay.hgn \
    --tokens-file "$TOK" --gen 1 --maxctx "$MAXCTX" >"$LOG" 2>&1
rc=$?
echo "rc=$rc 日志: $LOG"
grep -E 'prefill: ' "$LOG"
