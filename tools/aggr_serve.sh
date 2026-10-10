#!/usr/bin/env bash
# aggr_serve.sh start|stop|status — 常驻 v2 服务（PARALLEL=1，无视觉），供手工实验
cd "$(dirname "$0")/.."
case "${1:-status}" in
start)
  export MODEL_FILE="${MODEL_FILE:-./models/qwen38-flash-next-v2.hgn}" NGRAM_FILE="${NGRAM_FILE:-./models/qwen38-flash-next-ngram.hgn}"
  export PARALLEL="${PARALLEL:-1}" VISION_FILE="${VISION_FILE-}" OVERLAY_FILE="${OVERLAY_FILE-}"
  setsid bash start_hgn.sh > logs/aggr_serve.log 2>&1 < /dev/null &
  echo $! > logs/aggr_serve.pid
  for i in $(seq 1 200); do
    grep -q '服务已就绪' logs/aggr_serve.log && { echo READY; exit 0; }
    kill -0 "$(cat logs/aggr_serve.pid)" 2>/dev/null || { tail logs/aggr_serve.log; exit 1; }
    sleep 2
  done
  echo TIMEOUT; exit 1 ;;
stop)
  p=$(cat logs/aggr_serve.pid 2>/dev/null)
  [ -n "$p" ] && kill -TERM -- "-$p" 2>/dev/null
  sleep 5; pgrep -af 'build/qwenox-engine' || echo STOPPED ;;
status) pgrep -af 'build/qwenox-engine' || echo none ;;
esac
