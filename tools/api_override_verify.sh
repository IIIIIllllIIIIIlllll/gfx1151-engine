#!/usr/bin/env bash
# 一键验证 gdec-api 服务端参数覆盖（/admin/overrides）与控制台页面。
#   bash tools/api_override_verify.sh           # 先 bash build.sh api，再测
#   SKIP_BUILD=1 bash tools/api_override_verify.sh
# 用生产环境变量（start_hgn.sh --check）起测试引擎 :8732 + 测试 API :8733，覆盖表写到临时文件，
# 不碰生产的 data/api-overrides.json。三段：功能 / 重启后持久化 / 管理密钥鉴权。
# 最后一行：API OVERRIDE VERIFY: PASS 或 FAIL。
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source tools/probe_lib.sh
API_PORT=8733

probe_precheck || exit 1
[[ -z "$(ss -H -ltn "sport = :$API_PORT")" ]] || { echo "端口 $API_PORT 已被占用"; exit 1; }
if [[ "${SKIP_BUILD:-0}" != 1 ]]; then
  bash build.sh api >/dev/null 2>&1 || { echo "bash build.sh api 失败"; exit 1; }
fi
for f in start_hgn.sh start_gguf.sh tools/serve_common.sh service.conf; do grep -q $'\r' "$f" && sed -i 's/\r$//' "$f"; done
chk="$(bash start_hgn.sh --check 2>&1)" || { echo "$chk"; echo "start_hgn.sh --check 失败"; exit 1; }
mapfile -t PENV < <(sed -n 's/^ENV //p' <<<"$chk")
CMDLINE="$(sed -n 's/^CMD //p' <<<"$chk")"
eval "ENGINE=($CMDLINE)"
for i in "${!ENGINE[@]}"; do [[ "${ENGINE[$i]}" == --port ]] && ENGINE[$((i + 1))]=8732; done
PROBE_CAP_GB="$(source service.conf; echo "${MEMORY_CAP_GB:-86}")"; export PROBE_CAP_GB
TOKENIZER_DIR="$(source service.conf; echo "$TOKENIZER_DIR")"

TMPD="$(mktemp -d)"
OVR_FILE="$TMPD/api-overrides.json"
API_PID=''
api_stop() {
  [[ -n "$API_PID" ]] || return 0
  kill "$API_PID" 2>/dev/null; wait "$API_PID" 2>/dev/null; API_PID=''
}
api_start() {  # $1 = 日志 tag；其余为额外环境变量 K=V
  local tag=$1; shift
  env "$@" build/gdec-api --tokenizer "$TOKENIZER_DIR" --engine 127.0.0.1:8732 \
    --host 127.0.0.1 --port "$API_PORT" --overrides "$OVR_FILE" >"logs/$tag.log" 2>&1 </dev/null &
  API_PID=$!
  local t0=$SECONDS
  until [[ -n "$(ss -H -ltn "sport = :$API_PORT")" ]]; do
    kill -0 "$API_PID" 2>/dev/null || { echo "API 提前退出："; tail -n 15 "logs/$tag.log"; API_PID=''; return 1; }
    (( SECONDS - t0 < 30 )) || { echo "API 启动超时"; tail -n 15 "logs/$tag.log"; return 1; }
    sleep 0.5
  done
}
trap 'api_stop; probe_stop; rm -rf "$TMPD"' EXIT
trap 'exit 130' INT TERM

for e in $(compgen -e | grep '^GDEC_'); do unset "$e"; done
for e in "${PENV[@]}"; do export "$e"; done
export GDEC_KVSNAP_DIR="$TMPD/kvsnap"
mkdir -p "$GDEC_KVSNAP_DIR" logs
probe_start apiovr-engine "${ENGINE[@]}" || { echo "API OVERRIDE VERIFY: FAIL（引擎启动失败）"; exit 1; }

declare -A RES
api_start apiovr-api1 && python3 tools/api_override_verify.py main --port "$API_PORT" --file "$OVR_FILE"
RES[main]=$?
api_stop
api_start apiovr-api2 && python3 tools/api_override_verify.py persist --port "$API_PORT" --file "$OVR_FILE"
RES[persist]=$?
grep -q "overrides from $OVR_FILE" logs/apiovr-api2.log || { echo "FAIL API 启动日志没有加载覆盖表"; RES[persist]=1; }
api_stop
KEY="k$RANDOM$RANDOM"
api_start apiovr-api3 GDEC_API_ADMIN_KEY="$KEY" && python3 tools/api_override_verify.py auth --port "$API_PORT" --key "$KEY"
RES[auth]=$?
api_stop
grep -E 'hipError|Segmentation|Aborted|FATAL|GUARD PAGE' "$PROBE_LOG" | head -3

echo
echo "==== API 覆盖验证汇总 ===="
ok=1
for k in main persist auth; do
  [[ ${RES[$k]} == 0 ]] && s=PASS || { s=FAIL; ok=0; }
  printf '%-8s %s\n' "$k" "$s"
done
(( ok )) && echo "API OVERRIDE VERIFY: PASS" || echo "API OVERRIDE VERIFY: FAIL"
