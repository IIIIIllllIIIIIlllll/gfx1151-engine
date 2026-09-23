# probe_lib.sh — 前台启动/等待/停止一个测试引擎（端口 8732），供一键验证脚本 source。
#   probe_precheck                 # 没有其它引擎/API、8732 空闲
#   probe_start <tag> <cmd...>     # 后台起引擎，前台等 "serve: listening"，每 10s 打印进度
#   probe_stop                     # 停掉本次引擎（含 systemd scope），等进程退净
# 调用方负责先 export 好 GDEC_* 环境变量，并在 repo 根目录运行。
PROBE_PID=''
PROBE_SCOPE=''
PROBE_LOG=''
PROBE_PAT='(^|/)(gdec[^/[:space:]]*|flash_serve|serve_api\.py)([[:space:]]|$)'

probe_precheck() {
  local p
  if p="$(pgrep -af "$PROBE_PAT")"; then
    echo "已有引擎或 API 在运行，请先停掉（生产服务 start.sh 按 Ctrl+C）：" >&2
    echo "$p" >&2
    return 1
  fi
  if command -v ss >/dev/null && [[ -n "$(ss -H -ltn 'sport = :8732')" ]]; then
    echo "端口 8732 已被占用" >&2
    return 1
  fi
  [[ -x "${PROBE_BINARY:-build/gdec}" ]] || { echo "找不到 ${PROBE_BINARY:-build/gdec}，先 bash build.sh engine" >&2; return 1; }
  mkdir -p logs
}

probe_start() {
  local tag=$1; shift
  PROBE_LOG="logs/$tag.log"
  setsid bash tools/run_capped.sh "${PROBE_CAP_GB:-86}" -- "$@" >"$PROBE_LOG" 2>&1 </dev/null &
  PROBE_PID=$!
  PROBE_SCOPE="capped-${PROBE_PID}.scope"
  echo "[$tag] 引擎启动中（pid $PROBE_PID），日志 $PROBE_LOG"
  local t0=$SECONDS next=10
  until grep -q 'serve: listening' "$PROBE_LOG" 2>/dev/null; do
    if ! kill -0 "$PROBE_PID" 2>/dev/null; then
      echo "[$tag] 引擎提前退出，日志末尾：" >&2; tail -n 20 "$PROBE_LOG" >&2
      PROBE_PID=''; return 1
    fi
    if (( SECONDS - t0 >= ${PROBE_TIMEOUT:-600} )); then
      echo "[$tag] 启动超过 ${PROBE_TIMEOUT:-600}s，日志末尾：" >&2; tail -n 20 "$PROBE_LOG" >&2
      probe_stop; return 1
    fi
    if (( SECONDS - t0 >= next )); then
      echo "[$tag]   ... ${next}s: $(tail -n 1 "$PROBE_LOG" 2>/dev/null | cut -c1-120)"
      next=$((next + 10))
    fi
    sleep 1
  done
  echo "[$tag] 就绪，用时 $((SECONDS - t0))s"
}

probe_stop() {
  [[ -n "$PROBE_PID" ]] || return 0
  echo "[stop] 停止引擎 pid $PROBE_PID"
  kill -TERM -- "-$PROBE_PID" 2>/dev/null || true
  timeout -k 2 8 systemctl --user stop "$PROBE_SCOPE" >/dev/null 2>&1 || \
    systemctl --user kill --signal=KILL "$PROBE_SCOPE" >/dev/null 2>&1 || true
  kill -KILL -- "-$PROBE_PID" 2>/dev/null || true
  wait "$PROBE_PID" 2>/dev/null || true
  PROBE_PID=''
  local i
  for i in {1..30}; do pgrep -f "$PROBE_PAT" >/dev/null || break; sleep 1; done
  sleep 2   # 让 GTT/arena 释放干净再起下一个
}
