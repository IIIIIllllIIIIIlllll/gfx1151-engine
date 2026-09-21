#!/usr/bin/env bash
# Foreground supervisor: Ctrl+C stops this instance's API and engine scope.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
fail() { echo "错误：$*" >&2; exit 1; }
[[ $# -le 1 ]] || fail '用法：bash start.sh [--check]'
case "${1:-}" in
  -h|--help) echo '用法：bash start.sh [--check]；--check 只检查配置，Ctrl+C 停止服务。'; exit 0 ;;
  ''|--check) ;;
  *) fail '用法：bash start.sh [--check]' ;;
esac
source "$ROOT/service.conf"
for key in ENGINE_PORT API_PORT MAX_CONTEXT MTP_GAMMA MEMORY_CAP_GB MIN_AVAILABLE_GB START_TIMEOUT STALL_TIMEOUT; do
  value="${!key}"
  [[ "$value" =~ ^[1-9][0-9]*$ && ${#value} -le 8 ]] || fail "$key 必须为正整数"
done
(( ENGINE_PORT <= 65535 && API_PORT <= 65535 && ENGINE_PORT != API_PORT )) || fail '端口必须为不同的 1–65535 整数'
(( MTP_GAMMA <= 8 )) || fail 'MTP_GAMMA 范围为 1–8'
for cmd in flock ss systemctl stat awk pgrep setsid; do command -v "$cmd" >/dev/null || fail "缺少命令：$cmd"; done
systemctl --user show-environment >/dev/null || fail 'systemd 用户会话不可用，请通过普通用户 SSH 登录运行'
[[ -x build/gdec && -x build/gdec-api ]] || fail '缺少编译产物，请先运行 bash build.sh'
[[ -r "$MODEL_FILE" && -f "$MODEL_FILE" ]] || fail "找不到模型：$MODEL_FILE（修改 service.conf）"
[[ -z "$OVERLAY_FILE" || -f "$OVERLAY_FILE" && -r "$OVERLAY_FILE" ]] || fail "找不到 overlay：$OVERLAY_FILE"
[[ -z "$MTP_FILE" || -f "$MTP_FILE" && -r "$MTP_FILE" ]] || fail "找不到 MTP 权重：$MTP_FILE（可设 MTP_FILE=\"\" 退回 overlay 内置草稿头）"
[[ -z "$VISION_FILE" || -f "$VISION_FILE" && -r "$VISION_FILE" ]] || fail "找不到视觉塔：$VISION_FILE（纯文本可设 VISION_FILE=\"\"）"
[[ -r "$TOKENIZER_DIR/tokenizer.json" ]] || fail "找不到 tokenizer：$TOKENIZER_DIR/tokenizer.json"
for port in "$ENGINE_PORT" "$API_PORT"; do
  [[ -z "$(ss -H -ltn "sport = :$port")" ]] || fail "端口 $port 已被占用"
done
if processes="$(pgrep -af '(^|/)(gdec[^/[:space:]]*|flash_serve|serve_api\.py)([[:space:]]|$)')"; then
  fail "已有引擎或 API 进程，请先停止：$processes"
fi
available="$(awk '/MemAvailable:/{print int($2/1048576)}' /proc/meminfo)"
(( available >= MIN_AVAILABLE_GB )) || fail "可用内存 ${available} GiB，要求至少 ${MIN_AVAILABLE_GB} GiB"
echo "项目：$ROOT"
echo "模型：$MODEL_FILE"
echo "配置：${MAX_CONTEXT} 上下文，MTP gamma=${MTP_GAMMA}，API ${API_HOST}:${API_PORT}"
if [[ "${1:-}" == --check ]]; then
  echo '检查通过；没有启动引擎或 API。'
  exit 0
fi

mkdir -p logs
exec 9>logs/.service.lock
flock -n 9 || fail '本目录已有启动脚本在运行'
exec 8>"${XDG_RUNTIME_DIR:-/run/user/$UID}/hgn-work.lock"
flock -n 8 || fail '另一个 gfx1151-engine 编译或启动任务正在运行'
RUN_LOG_DIR="logs/$(date +%Y%m%d-%H%M%S)-$$"
mkdir "$RUN_LOG_DIR"
ENGINE_LOG="$RUN_LOG_DIR/engine.log"
API_LOG="$RUN_LOG_DIR/api.log"
engine_pid=''
api_pid=''
scope=''
cleanup() {
  local result=$?
  trap - EXIT INT TERM HUP
  # Stop only the scope created by this invocation, including GPU descendants.
  if [[ -n "$api_pid" ]]; then kill -TERM "$api_pid" 2>/dev/null || true; fi
  # setsid makes the wrapper and its children our private process group. This
  # also handles interruption before systemd has finished creating the scope.
  if [[ -n "$engine_pid" ]]; then kill -TERM -- "-$engine_pid" 2>/dev/null || true; fi
  if [[ -n "$scope" ]]; then
    timeout -k 2 8 systemctl --user stop "$scope" >/dev/null 2>&1 || \
      systemctl --user kill --signal=KILL "$scope" >/dev/null 2>&1 || true
  fi
  if [[ -n "$engine_pid" ]]; then kill -KILL -- "-$engine_pid" 2>/dev/null || true; fi
  if [[ -n "$api_pid" ]]; then
    for _ in {1..20}; do kill -0 "$api_pid" 2>/dev/null || break; sleep 0.1; done
    kill -KILL "$api_pid" 2>/dev/null || true
    wait "$api_pid" 2>/dev/null || true
  fi
  if [[ -n "$engine_pid" ]]; then wait "$engine_pid" 2>/dev/null || true; fi
  echo "本次服务已停止。日志：$RUN_LOG_DIR"
  exit "$result"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP

# Existing production options; no closed-source modules or experimental paths.
export GDEC_QSA_KV_BF16=1 GDEC_QSA_WMMA=1 GDEC_QSA_WMMA_BTV=1
# 自写 WMMA dense GEMM（Phase 3g）：8K +2%、32K +3%，对拍 8049 token 仅尾部分歧 4 个。
export GDEC_GEMM_WMMA=1
# GDN 融合持久化 kernel（Phase 3c）：intra+strip 全融合、ws 不落 DRAM，kernel 3.34×，
# 8K +5.7%、32K +4.4%，ids 对拍 8054 token 0 分歧。与 GDEC_GDN_PIPE2 互斥（fused 优先）。
export GDEC_GDN_FUSED=1
export GDEC_MOE_LT=1 GDEC_MOE_LT_BF16=1 GDEC_GR_BF16=1
export GDEC_GDN_STREAM=1 GDEC_GDN_WAVE=1 GDEC_NOWARMUP=1
# 32768: 32K prompt 单 chunk 实测 +7.4%（1155 vs 1076 tok/s）；65536 超内存 PSI 上限。
export GDEC_PREFILL_CHUNK=32768
export GDEC_INDEX_FUSED2=1 GDEC_PP_MOE_OUT=1 GDEC_INDEX_STREAM_SELECT=1
export GDEC_KVSNAP=1
export GDEC_KVSNAP_MAX_GB=20
# --serve reads GDEC_SPEC_GAMMA; --gamma is for offline --spec-gen.
export GDEC_SPEC_GAMMA="$MTP_GAMMA"
engine=("$ROOT/build/gdec" "$MODEL_FILE")
[[ -z "$OVERLAY_FILE" ]] || engine+=("$OVERLAY_FILE")
[[ -z "$MTP_FILE" ]] || engine+=("$MTP_FILE")
engine+=(--serve --port "$ENGINE_PORT" --maxctx "$MAX_CONTEXT")
[[ -z "$VISION_FILE" ]] || engine+=(--vision-tower "$VISION_FILE")
echo "加载模型中，日志：$ENGINE_LOG；Ctrl+C 停止。"
setsid bash tools/run_capped.sh "$MEMORY_CAP_GB" -- "${engine[@]}" >"$ENGINE_LOG" 2>&1 9>&- 8>&- &
engine_pid=$!
scope="capped-${engine_pid}.scope"

# Track actual engine I/O and GPU allocation as well as log output while loading.
progress_stamp() {
  local cg pid f
  stat -c %s "$ENGINE_LOG"
  cg="$(systemctl --user show "$scope" -p ControlGroup --value 2>/dev/null || true)"
  if [[ -n "$cg" && -r "/sys/fs/cgroup$cg/cgroup.procs" ]]; then
    while read -r pid; do
      awk '/^(rchar|wchar|read_bytes|write_bytes):/{print}' "/proc/$pid/io" 2>/dev/null || true
    done <"/sys/fs/cgroup$cg/cgroup.procs"
  fi
  for f in /sys/class/drm/card*/device/mem_info_gtt_used; do
    [[ ! -r "$f" ]] || cat "$f"
  done
}
begin=$SECONDS
last_progress=$SECONDS
previous=''
until grep -q 'serve: listening' "$ENGINE_LOG"; do
  kill -0 "$engine_pid" 2>/dev/null || { tail -n 15 "$ENGINE_LOG" >&2; fail '引擎提前退出'; }
  (( SECONDS - begin < START_TIMEOUT )) || fail "引擎启动超过 ${START_TIMEOUT} 秒，查看 $ENGINE_LOG"
  current="$(progress_stamp)"
  if [[ "$current" != "$previous" ]]; then previous="$current"; last_progress=$SECONDS; fi
  (( SECONDS - last_progress < STALL_TIMEOUT )) || fail "加载停滞 ${STALL_TIMEOUT} 秒，查看 $ENGINE_LOG"
  sleep 1
done
kill -0 "$engine_pid" 2>/dev/null || fail '引擎已退出'
"$ROOT/build/gdec-api" --tokenizer "$TOKENIZER_DIR" \
  --engine "127.0.0.1:$ENGINE_PORT" --host "$API_HOST" \
  --port "$API_PORT" --context "$MAX_CONTEXT" >"$API_LOG" 2>&1 9>&- 8>&- &
api_pid=$!
begin=$SECONDS
until [[ -n "$(ss -H -ltn "sport = :$API_PORT")" ]]; do
  kill -0 "$engine_pid" 2>/dev/null || fail '引擎已退出'
  kill -0 "$api_pid" 2>/dev/null || { tail -n 15 "$API_LOG" >&2; fail 'API 提前退出'; }
  (( SECONDS - begin < 15 )) || fail "API 启动超时，查看 $API_LOG"
  sleep 0.2
done
echo "服务已就绪：http://${API_HOST}:${API_PORT}/v1（0.0.0.0 表示监听所有网卡）"
echo "日志：$RUN_LOG_DIR；Ctrl+C 同时停止 API 和引擎。"
if wait -n "$engine_pid" "$api_pid"; then
  fail '服务进程意外结束'
else
  result=$?
  echo "服务进程退出（$result），查看 $RUN_LOG_DIR" >&2
  exit "$result"
fi
