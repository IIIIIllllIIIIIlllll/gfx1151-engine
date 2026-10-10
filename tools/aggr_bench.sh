#!/usr/bin/env bash
# aggr_bench.sh — prefill 基线/AB 一键脚本（llm_speedtest 口径，对标 halogen V2 表）
# 用生产启动器 start_hgn.sh 拉起服务（v2 权重；PARALLEL=1 单路、无视觉塔），
# 跑 tools/speedtest_cli.py，打印各长度 prefill 与 halogen 比值，结束自动停服务。
# 用法: bash tools/aggr_bench.sh [TAG] [长度列表，默认 4096,8192,16384,32768]
# 环境: KPROF=1 打开 QWENOX_KPROF（段级计时，有同步开销，只看分段不看总速）
#       其余 QWENOX_* / service.conf 变量照常透传（如 PREFILL_CHUNK、PARALLEL）。
set -u
cd "$(dirname "$0")/.."
TAG="${1:-base}"
LENS="${2:-4096,8192,16384,32768}"
mkdir -p logs
OUT="logs/aggr_${TAG}.txt"
export MODEL_FILE="${MODEL_FILE:-./models/qwen38-flash-next-v2.hgn}"
export NGRAM_FILE="${NGRAM_FILE:-./models/qwen38-flash-next-ngram.hgn}"
export PARALLEL="${PARALLEL:-1}" VISION_FILE="${VISION_FILE-}" OVERLAY_FILE="${OVERLAY_FILE-}"
export API_PORT="${API_PORT:-8731}" ENGINE_PORT="${ENGINE_PORT:-8730}"
[[ "${KPROF:-0}" == 1 ]] && export QWENOX_KPROF=1 QWENOX_PROF=1
setsid bash start_hgn.sh > "logs/aggr_${TAG}.serve.log" 2>&1 < /dev/null &
SPID=$!
stop() { kill -TERM -- "-$SPID" 2>/dev/null; for _ in $(seq 1 30); do kill -0 $SPID 2>/dev/null || break; sleep 1; done; kill -KILL -- "-$SPID" 2>/dev/null; }
trap stop EXIT
begin=$SECONDS
until grep -q '服务已就绪' "logs/aggr_${TAG}.serve.log" 2>/dev/null; do
  kill -0 $SPID 2>/dev/null || { tail -20 "logs/aggr_${TAG}.serve.log"; echo FAIL; exit 1; }
  (( SECONDS - begin < 400 )) || { echo "启动超时"; echo FAIL; exit 1; }
  sleep 2
done
echo "服务就绪 $((SECONDS - begin)) s" | tee "$OUT"
PYTHONPATH="$HOME/Workspace/pylib:$HOME/Workspace/llm_speedtest" python3 tools/speedtest_cli.py \
  --url "http://127.0.0.1:${API_PORT}/v1/chat/completions" --model qwenox \
  --lengths "$LENS" --out 32 --json "logs/aggr_${TAG}.json" 2>&1 \
  | grep -vE '^\[(Request|Response|Stream|FirstChunk|FirstToken|Usage|Debug|Token|TimeSource|Timing|ITL|Result|Latency|Timeout|Warmup|PromptEstimate)\]' | tee -a "$OUT"
RUNDIR=$(grep -oE 'logs/[0-9]{8}-[0-9]{6}-[0-9]+' "logs/aggr_${TAG}.serve.log" | head -1)
echo "engine log: $RUNDIR/engine.log" | tee -a "$OUT"
grep -E 'prefill: ' "$RUNDIR/engine.log" | tail -20 | tee -a "$OUT"
echo DONE
