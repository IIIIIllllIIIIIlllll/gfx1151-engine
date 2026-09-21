#!/usr/bin/env bash
# Windows (TheRock) 启动入口：与 Linux start.sh 并列。
# 起引擎（行协议 ENGINE_PORT，默认 8730）+ OpenAI HTTP API 前端
# （gdec-api-win，API_PORT，默认 8731）；Ctrl+C 同时停两者。
#
# 用法:
#   bash start_win.sh            # 前台运行，Ctrl+C 停止
#   bash start_win.sh --check    # 只检查配置，不启动
#
# 覆盖（service.conf 风格，环境变量优先）:
#   MODEL_FILE=../path/x.hgn MTP_FILE="" VISION_FILE="" ENGINE_PORT=8730 \
#   API_PORT=8731 MAX_CONTEXT=262144 MTP_GAMMA=3 bash start_win.sh
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
# 部署参数（模型/端口/上下文等）读根目录 service.conf，与 Linux start.sh、
# Windows start_win.exe 同一文件；环境变量优先。计时类平台项不取 conf
#（Windows 冷加载分钟级，START_TIMEOUT 内置 1800）。
if [[ -f service.conf ]]; then
  _env_start_timeout="${START_TIMEOUT-}"
  source service.conf
  START_TIMEOUT="${_env_start_timeout:-1800}"
fi
fail() { echo "错误：$*" >&2; exit 1; }
[[ $# -le 1 ]] || fail '用法：bash start_win.sh [--check]'
case "${1:-}" in
  -h|--help) sed -n '2,13p' "$0"; exit 0 ;;
  ''|--check) ;;
  *) fail '用法：bash start_win.sh [--check]' ;;
esac

# build/ 自包含：HIP/rocBLAS/hipBLASLt DLL 与 gfx1151 kernel db 都在 exe 旁，
# 无需 TheRock/ROCm 运行库，也不需要 HIP_PATH/ROCM_PATH（已断根实测）。

MODEL_DIR="${MODEL_DIR:-./models}"
MODEL_FILE="${MODEL_FILE:-$MODEL_DIR/heretic.hgn}"
MTP_FILE="${MTP_FILE-$MODEL_DIR/heretic-mtp.hgn}"
VISION_FILE="${VISION_FILE-$MODEL_DIR/heretic-vision.hgn}"
# 覆盖层（可选的高精度替换张量）：默认空=不叠加；非空但文件不存在则报错。
OVERLAY_FILE="${OVERLAY_FILE-}"
TOKENIZER_DIR="${TOKENIZER_DIR:-$MODEL_DIR/tokenizer}"
ENGINE_PORT="${ENGINE_PORT:-8730}"
API_HOST="${API_HOST:-0.0.0.0}"
API_PORT="${API_PORT:-8731}"
MAX_CONTEXT="${MAX_CONTEXT:-262144}"
MTP_GAMMA="${MTP_GAMMA:-3}"
KVSNAP_MAX_GB="${KVSNAP_MAX_GB:-20}"
# 本机 68 GiB 权重 cold-load 实测 ~9 分钟（NVMe 弱盘），超时给足。
START_TIMEOUT="${START_TIMEOUT:-1800}"

[[ "$ENGINE_PORT" =~ ^[1-9][0-9]*$ && "$ENGINE_PORT" -le 65535 ]] || fail 'ENGINE_PORT 必须为 1–65535'
[[ "$API_PORT" =~ ^[1-9][0-9]*$ && "$API_PORT" -le 65535 ]] || fail 'API_PORT 必须为 1–65535'
[[ "$ENGINE_PORT" != "$API_PORT" ]] || fail 'ENGINE_PORT 与 API_PORT 必须不同'
[[ "$MTP_GAMMA" =~ ^[1-8]$ ]] || fail 'MTP_GAMMA 范围为 1–8'
[[ "$KVSNAP_MAX_GB" =~ ^[1-9][0-9]*$ ]] || fail 'KVSNAP_MAX_GB 必须为正整数'
[[ -f build/gdec-win.exe ]] || fail '缺少 build/gdec-win.exe，请先运行 bash build_win.sh'
[[ -f build/gdec-api-win.exe ]] || fail '缺少 build/gdec-api-win.exe，请先运行 bash build_win.sh api'
[[ -r "$MODEL_FILE" ]] || fail "找不到模型：$MODEL_FILE"
[[ -z "$MTP_FILE" || -r "$MTP_FILE" ]] || fail "找不到 MTP 权重：$MTP_FILE"
[[ -z "$VISION_FILE" || -r "$VISION_FILE" ]] || fail "找不到视觉塔：$VISION_FILE（纯文本可设 VISION_FILE=\"\"）"
[[ -z "$OVERLAY_FILE" || -r "$OVERLAY_FILE" ]] || fail "找不到 overlay：$OVERLAY_FILE（无 overlay 可设 OVERLAY_FILE=\"\"）"
[[ -r "$TOKENIZER_DIR/tokenizer.json" ]] || fail "找不到 tokenizer：$TOKENIZER_DIR"
[[ -d build/rocblas/library && -d build/hipblaslt/library ]] || fail '缺少 rocBLAS/hipBLASLt kernel db（build/*/library），请先运行 bash build_win.sh'
for port in "$ENGINE_PORT" "$API_PORT"; do
  if netstat -an | grep -E "[:.]$port\s+.*LISTENING" >/dev/null 2>&1; then
    fail "端口 $port 已被占用"
  fi
done
echo "项目：$ROOT"
echo "模型：$MODEL_FILE"
echo "配置：${MAX_CONTEXT} 上下文，MTP gamma=${MTP_GAMMA}，API ${API_HOST}:${API_PORT}"
if [[ "${1:-}" == --check ]]; then
  echo '检查通过；没有启动引擎或 API。'
  exit 0
fi

# 生产选项与 Linux start.sh 一致，唯独不设 GDEC_PREFILL_CHUNK：
# Windows 默认 8192（256K 下 16384 会顶破 95 GiB arena 上限，实测见
# PORTING-WINDOWS.md；maxctx ≤ 40K 时可手动 GDEC_PREFILL_CHUNK=16384 换 ~6% PP）。
export GDEC_QSA_KV_BF16=1 GDEC_QSA_WMMA=1 GDEC_QSA_WMMA_BTV=1
export GDEC_MOE_LT=1 GDEC_MOE_LT_BF16=1 GDEC_GR_BF16=1
export GDEC_GDN_STREAM=1 GDEC_GDN_WAVE=1 GDEC_NOWARMUP=1
export GDEC_INDEX_FUSED2=1 GDEC_PP_MOE_OUT=1 GDEC_INDEX_STREAM_SELECT=1
export GDEC_KVSNAP=1 GDEC_KVSNAP_MAX_GB="$KVSNAP_MAX_GB"
export GDEC_SPEC_GAMMA="$MTP_GAMMA"

mkdir -p logs
ENGINE_LOG="logs/engine-win-$(date +%Y%m%d-%H%M%S).log"

engine=(build/gdec-win.exe "$MODEL_FILE")
[[ -z "$OVERLAY_FILE" ]] || engine+=("$OVERLAY_FILE")
[[ -z "$MTP_FILE" ]] || engine+=("$MTP_FILE")
engine+=(--serve --port "$ENGINE_PORT" --maxctx "$MAX_CONTEXT")
[[ -z "$VISION_FILE" ]] || engine+=(--vision-tower "$VISION_FILE")

echo "加载模型中，日志：$ENGINE_LOG；Ctrl+C 停止。"
"${engine[@]}" >"$ENGINE_LOG" 2>&1 &
engine_pid=$!
api_pid=''
API_LOG="logs/api-win-$(date +%Y%m%d-%H%M%S).log"
cleanup() {
  trap - EXIT INT TERM
  for pid in $api_pid $engine_pid; do
    [[ -n "$pid" ]] || continue
    kill "$pid" 2>/dev/null || true
    taskkill //PID "$pid" //F //T >/dev/null 2>&1 || true
  done
  echo "本次服务已停止。日志：$ENGINE_LOG $API_LOG"
}
trap cleanup EXIT INT TERM

begin=$SECONDS
until grep -q 'serve: listening' "$ENGINE_LOG" 2>/dev/null; do
  kill -0 "$engine_pid" 2>/dev/null || { tail -n 15 "$ENGINE_LOG" >&2; fail '引擎提前退出'; }
  (( SECONDS - begin < START_TIMEOUT )) || fail "引擎启动超过 ${START_TIMEOUT} 秒，查看 $ENGINE_LOG"
  sleep 2
done

build/gdec-api-win.exe --tokenizer "$TOKENIZER_DIR" \
  --engine "127.0.0.1:$ENGINE_PORT" --host "$API_HOST" \
  --port "$API_PORT" --context "$MAX_CONTEXT" >"$API_LOG" 2>&1 &
api_pid=$!
begin=$SECONDS
until netstat -an | grep -E "[:.]$API_PORT\s+.*LISTENING" >/dev/null 2>&1; do
  kill -0 "$engine_pid" 2>/dev/null || fail '引擎已退出'
  kill -0 "$api_pid" 2>/dev/null || { tail -n 15 "$API_LOG" >&2; fail 'API 提前退出'; }
  (( SECONDS - begin < 30 )) || fail "API 启动超时，查看 $API_LOG"
  sleep 1
done

echo "服务已就绪：http://${API_HOST}:${API_PORT}/v1（0.0.0.0 表示监听所有网卡）"
echo "日志：$ENGINE_LOG $API_LOG；Ctrl+C 同时停止 API 和引擎。"
if wait -n "$engine_pid" "$api_pid"; then
  fail '服务进程意外结束'
else
  result=$?
  echo "服务进程退出（$result），查看日志" >&2
  exit "$result"
fi
