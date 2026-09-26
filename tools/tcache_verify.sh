#!/usr/bin/env bash
# 一键验证多轮对话的 KV 前缀复用（API token 缓存 + 视觉 M-RoPE 门放宽 + 未命中诊断）。
#   bash tools/tcache_verify.sh                 # 用 build/ 下现有引擎和 API
#   ENGINE_BIN=build/x API_BIN=build/y bash tools/tcache_verify.sh
#   MAX_TOKENS=3000 ONLY=text bash tools/tcache_verify.sh
# 用生产环境变量（start_hgn.sh --check）起测试引擎 :8732，kvsnap 写临时目录；
# API 起两份：:8733 token 缓存打开，:8734 GDEC_API_TOKCACHE=0 作对照。
# 文本：采样生成长回复后追问，要求整段复用；视觉：文本历史后发图、再加第二张图都要复用，
# 换掉图 1 的像素则不能复用图片 KV。toolcall：float 参数的工具调用回填后追问，
# 要求命中 <tool_call> 处的 mid-decode 检查点而非退回上轮 prompt 末尾。
# persist：种子轮后重启缓存 ON 的 API（同一 GDEC_API_TOKCACHE_FILE），追问仍须整段复用。
# 耗时约 5–10 分钟（加载 ~1 分钟 + 长回复解码）。
# 最后一行：TCACHE VERIFY: PASS 或 FAIL。
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source tools/probe_lib.sh
ENGINE_BIN="${ENGINE_BIN:-build/gdec}"
API_BIN="${API_BIN:-build/gdec-api}"
ON_PORT=8733
OFF_PORT=8734

PROBE_BINARY="$ENGINE_BIN" probe_precheck || exit 1
[[ -x "$API_BIN" ]] || { echo "找不到 $API_BIN，先 bash build.sh api"; exit 1; }
for p in $ON_PORT $OFF_PORT; do
  [[ -z "$(ss -H -ltn "sport = :$p")" ]] || { echo "端口 $p 已被占用"; exit 1; }
done
for f in start_hgn.sh start_gguf.sh tools/serve_common.sh service.conf; do grep -q $'\r' "$f" && sed -i 's/\r$//' "$f"; done
chk="$(bash start_hgn.sh --check 2>&1)" || { echo "$chk"; echo "start_hgn.sh --check 失败"; exit 1; }
mapfile -t PENV < <(sed -n 's/^ENV //p' <<<"$chk")
CMDLINE="$(sed -n 's/^CMD //p' <<<"$chk")"
eval "ENGINE=($CMDLINE)"
ENGINE[0]="$PWD/$ENGINE_BIN"
for i in "${!ENGINE[@]}"; do [[ "${ENGINE[$i]}" == --port ]] && ENGINE[$((i + 1))]=8732; done
PROBE_CAP_GB="$(source service.conf; echo "${MEMORY_CAP_GB:-86}")"; export PROBE_CAP_GB
TOKENIZER_DIR="$(source service.conf; echo "$TOKENIZER_DIR")"
echo "引擎 $ENGINE_BIN，API $API_BIN"

TMPD="$(mktemp -d)"
API_PIDS=()
apis_stop() {
  local p
  for p in "${API_PIDS[@]}"; do kill "$p" 2>/dev/null; wait "$p" 2>/dev/null; done
  API_PIDS=()
}
api_start() {  # $1 端口 $2 日志 tag；其余为额外环境变量 K=V
  local port=$1 tag=$2; shift 2
  env "$@" "$API_BIN" --tokenizer "$TOKENIZER_DIR" --engine 127.0.0.1:8732 \
    --host 127.0.0.1 --port "$port" --overrides "" >"logs/$tag.log" 2>&1 </dev/null &
  local pid=$!
  API_PIDS+=("$pid")
  LAST_API_PID=$pid
  local t0=$SECONDS
  until [[ -n "$(ss -H -ltn "sport = :$port")" ]]; do
    kill -0 "$pid" 2>/dev/null || { echo "API 提前退出："; tail -n 15 "logs/$tag.log"; return 1; }
    (( SECONDS - t0 < 30 )) || { echo "API 启动超时"; tail -n 15 "logs/$tag.log"; return 1; }
    sleep 0.5
  done
}
trap 'apis_stop; probe_stop; rm -rf "$TMPD"' EXIT
trap 'exit 130' INT TERM

for e in $(compgen -e | grep '^GDEC_'); do unset "$e"; done
for e in "${PENV[@]}"; do export "$e"; done
export GDEC_KVSNAP_DIR="$TMPD/kvsnap"
# 降低 rckpt 最小 token 门槛（默认 4096），让 toolcall 场景不用堆 4K+ prompt。
export GDEC_RCKPT_MIN="${GDEC_RCKPT_MIN:-200}"
mkdir -p "$GDEC_KVSNAP_DIR" logs
probe_start tcache-engine "${ENGINE[@]}" || { echo "TCACHE VERIFY: FAIL（引擎启动失败）"; exit 1; }
api_start $ON_PORT tcache-api-on \
  GDEC_API_TOKCACHE_FILE="$TMPD/tcache.bin" GDEC_API_TOKCACHE_SAVE_S=0 \
  || { echo "TCACHE VERIFY: FAIL（API 启动失败）"; exit 1; }
ON_PID=$LAST_API_PID
api_start $OFF_PORT tcache-api-off GDEC_API_TOKCACHE=0 GDEC_API_TOKCACHE_FILE= \
  || { echo "TCACHE VERIFY: FAIL（API 启动失败）"; exit 1; }

args=(--on $ON_PORT --off $OFF_PORT --max-tokens "${MAX_TOKENS:-6000}")
[[ -n "${ONLY:-}" && "$ONLY" != persist ]] && args+=(--only "$ONLY")
rc=0
if [[ -z "${ONLY:-}" || "$ONLY" != persist ]]; then
  python3 tools/tcache_verify.py "${args[@]}" || rc=1
fi

# persist 场景：种子轮 → 杀 ON API 重启（同一份 tcache.bin）→ 追问仍复用。
if [[ -z "${ONLY:-}" || "$ONLY" == persist ]]; then
  python3 tools/tcache_verify.py --on $ON_PORT --off $OFF_PORT \
    --only persist1 --state "$TMPD/persist.json" || rc=1
  if [[ -s "$TMPD/tcache.bin" ]]; then
    echo "[persist] tcache.bin 已落盘（$(stat -c%s "$TMPD/tcache.bin") 字节），重启 API ..."
    kill "$ON_PID" 2>/dev/null; wait "$ON_PID" 2>/dev/null
    api_start $ON_PORT tcache-api-on2 \
      GDEC_API_TOKCACHE_FILE="$TMPD/tcache.bin" GDEC_API_TOKCACHE_SAVE_S=0 || rc=1
    if grep -q 'tcache: loaded' logs/tcache-api-on2.log; then
      echo "PASS persist-load: $(grep 'tcache: loaded' logs/tcache-api-on2.log | tail -1)"
    else
      echo "FAIL persist-load: 重启后日志没有 'tcache: loaded'"; rc=1
    fi
    python3 tools/tcache_verify.py --on $ON_PORT --off $OFF_PORT \
      --only persist2 --state "$TMPD/persist.json" || rc=1
  else
    echo "FAIL persist: 种子轮后 $TMPD/tcache.bin 不存在（SAVE_S=0 应立即落盘）"; rc=1
  fi
fi

apis_stop

echo
echo "==== 日志摘录 ===="
echo "API（缓存 ON）："; grep 'tcache: reused' logs/tcache-api-on.log | tail -n 5
n_off="$(grep -c 'tcache: reused' logs/tcache-api-off.log)"
echo "API（缓存 OFF）tcache 命中行数：$n_off（应为 0）"
echo "引擎 未命中诊断 / 视觉 cont 否决 / 恢复 / ckpt："
grep -E 'no live prefix|live prefix .* not reused|rckpt|kvsnap: restore|ckpt: saved' "$PROBE_LOG" | tail -n 12
crash="$(grep -E 'hipError|Segmentation|Aborted|FATAL|GUARD PAGE' "$PROBE_LOG" | head -3)"
[[ -z "$crash" ]] || { echo "引擎日志有错误："; echo "$crash"; rc=1; }
[[ "$n_off" == 0 ]] || { echo "FAIL GDEC_API_TOKCACHE=0 没有关掉缓存"; rc=1; }

echo
(( rc == 0 )) && echo "TCACHE VERIFY: PASS" || echo "TCACHE VERIFY: FAIL"
