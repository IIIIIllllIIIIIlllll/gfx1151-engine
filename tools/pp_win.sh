#!/usr/bin/env bash
# pp_win.sh — PP 性能测试（Windows / Git Bash 版，对应 tools/pp.sh）
# 用法: ./tools/pp_win.sh <长度> [标签]
#   ./tools/pp_win.sh 32k            # 32K prefill
#   ./tools/pp_win.sh 128k my128     # 128K，自定义日志名
# 长度支持 2k / 32k / 128k（也可写 2048 / 32768 / 131072），
# 自动匹配 data/qsa-oracle/<N>.tokens 并计算 maxctx（长度 + 8K 余量）。
# 输出全部进日志，跑完自动汇总（预热/首块/稳态分开）。终端无洪水。
# 开关：PP_NOWARMUP=1 跳过引擎预热；PP_PHASE=1 开启同步 profiling（诊断用）。
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
    echo "可用的长度: $(ls data/qsa-oracle/*.tokens 2>/dev/null | sed 's|.*qsa-oracle/||;s|\.tokens||' | tr '\n' ' ')" >&2
    exit 1
fi

MAXCTX=$((LEN + 8192))

# chunk 大小与下方 QWENOX_PREFILL_CHUNK 保持一致；NCHUNK 是真实 chunk 数
# （向上取整，tail slack 的余量并入上一个 chunk，不单独成行）。
# 引擎默认先用全零 dummy 跑一轮预热（52_main.inc:460），打印一行 prefill。
# 注意：QWENOX_NOWARMUP / QWENOX_PHASE 引擎判断的是变量“是否存在”，设 0 也生效，
# 所以这里必须显式 unset，并从脚本自己的开关重新导出。
CHUNK=${PREFILL_CHUNK:-8192}
NCHUNK=$(( (LEN + CHUNK - 1) / CHUNK ))

unset QWENOX_NOWARMUP QWENOX_PHASE
EXTRA_ENV=()
if [[ "${PP_NOWARMUP:-}" == "1" ]]; then
    EXTRA_ENV+=(QWENOX_NOWARMUP=1); WARMUP="关（PP_NOWARMUP=1）"
else
    WARMUP="开（全零 dummy 一轮，不含真实 PLE 行）"
fi
if [[ "${PP_PHASE:-}" == "1" ]]; then
    EXTRA_ENV+=(QWENOX_PHASE=1); PHASE="开（同步 profiling，会打乱流水重叠，仅诊断用）"
else
    PHASE="关"
fi
echo "预热: $WARMUP | QWENOX_PHASE: $PHASE"

LOG="$ROOT/logs/${LABEL}.log"
# 默认 w4b + overlay；MODEL_FILE 覆盖底模（如 v2），此时用 NGRAM_FILE 作 overlay
MODEL_ARGS=("${MODEL_FILE:-models/qwen38-flash-next-w4b.hgn}")
if [[ -z "${MODEL_FILE:-}" ]]; then
    MODEL_ARGS+=("models/qwen38-flash-next-w4b.overlay.hgn")
elif [[ -n "${NGRAM_FILE:-}" ]]; then
    MODEL_ARGS+=("$NGRAM_FILE")
fi
env "${EXTRA_ENV[@]}" \
    QWENOX_QSA_KV_BF16=1 QWENOX_QSA_WMMA=1 QWENOX_QSA_WMMA_BTV=1 \
    QWENOX_MOE_LT=1 QWENOX_MOE_LT_BF16=1 QWENOX_GR_BF16=1 \
    QWENOX_GDN_STREAM=1 QWENOX_GDN_WAVE=1 \
    QWENOX_PREFILL_CHUNK=$CHUNK QWENOX_GEMM_WMMA=1 QWENOX_GDN_FUSED=1 \
    QWENOX_INDEX_FUSED2=1 QWENOX_PP_MOE_OUT=1 QWENOX_INDEX_STREAM_SELECT=1 \
    QWENOX_KVSNAP=1 QWENOX_KVSNAP_MAX_GB=20 QWENOX_PROF=1 \
    build/qwenox-engine-win "${MODEL_ARGS[@]}" \
    --tokens-file "$TOK" --gen 1 --maxctx "$MAXCTX" >"$LOG" 2>&1
rc=$?
echo "rc=$rc 日志: $LOG"

# 汇总：区分预热行 / 首块（冷 PLE，ple_wait 集中在这一块）/ 稳态块
grep -E 'prefill: [0-9]+ tokens in' "$LOG" | \
    sed -E 's/.*in ([0-9.]+) s = ([0-9.]+) tok\/s/\1 \2/' | \
    awk -v warm="$WARMUP" -v nchunk="$NCHUNK" '{
        t[NR]=$1; v[NR]=$2
    } END {
        i = 1
        if (NR == nchunk + 1) { printf "预热(dummy): %.1f tok/s\n", v[1]; i = 2 }
        else if (NR != nchunk) { printf "(行数 %d != 预期 %d，全量列出)\n", NR, nchunk }
        printf "chunk 1 (冷): %.1f tok/s (%.2f s)\n", v[i], t[i]
        n = 0; s = 0; mn = 1e9; mx = 0
        for (j = i + 1; j <= NR; j++) { s += v[j]; n++; if (v[j]<mn) mn=v[j]; if (v[j]>mx) mx=v[j] }
        if (n > 0) printf "chunk 2+ 稳态: avg %.1f tok/s (min %.1f / max %.1f, n=%d)\n", s/n, mn, mx, n
    }'
