#!/usr/bin/env bash
# 并发（GDEC_PARALLEL）一键验证：前台依次起 3~4 个测试引擎（端口 8732），最后一行 PASS/FAIL。
#   bash tools/conc_verify.sh
#   OLD_BIN=build/gdec.old bash tools/conc_verify.sh      # 另外对拍一个旧二进制的单路结果
#   NEW_BIN=build/gdec.conc OLD_BIN=build/gdec bash tools/conc_verify.sh
#   STAGES="ovf" CONC_ENV="GDEC_KV_ZERO=1" bash tools/conc_verify.sh   # 调试：只跑某几段 / 附加引擎环境
# 需要先停掉生产服务。生产环境变量取自 start.sh --check；kvsnap 关闭，ngram verify
# 分块固定为 16（自适应分块跨请求共享，会让结果依赖执行顺序）。
#   1. [OLD_BIN 单路] 与 新二进制单路：同一组请求串行跑，逐 token + spec 统计一致
#   2. 新二进制 4 路：同一组请求由 4 个客户端线程各开连接并发发送（对话流不被挤掉槽位），
#      必须和单路串行逐位一致；
#      再测 INFO kv_slots、负载下 PING 延迟、排队中取消、运行中取消、同连接重复 GEN、断连
#   3. 新二进制 2 路 + --maxctx 16384（共享池只够一条 16K 序列）：池溢出时后来的请求失败，
#      先来的请求结果与单独运行一致，之后池恢复可用
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source tools/probe_lib.sh
NEW_BIN="${NEW_BIN:-build/gdec}"
OLD_BIN="${OLD_BIN:-}"
STAGES=" ${STAGES:-seq par ovf} "
read -ra CONC_ENV <<<"${CONC_ENV:-}"
want() { [[ "$STAGES" == *" $1 "* ]]; }
export PROBE_BINARY="$NEW_BIN"
probe_precheck || exit 1
[[ -z "$OLD_BIN" || -x "$OLD_BIN" ]] || { echo "找不到 OLD_BIN=$OLD_BIN"; exit 1; }
for f in start.sh service.conf; do grep -q $'\r' "$f" && sed -i 's/\r$//' "$f"; done
chk="$(bash start.sh --check 2>&1)" || { echo "$chk"; echo "start.sh --check 失败"; exit 1; }
mapfile -t PENV < <(sed -n 's/^ENV //p' <<<"$chk")
CMDLINE="$(sed -n 's/^CMD //p' <<<"$chk")"
eval "BASE=($CMDLINE)"
PROBE_CAP_GB="$(source service.conf; echo "${MEMORY_CAP_GB:-86}")"; export PROBE_CAP_GB
trap 'probe_stop' EXIT
trap 'exit 130' INT TERM
mkdir -p logs/conc
want seq && rm -f logs/conc/*.json

declare -a NAMES RES
fails=0
note() { NAMES+=("$1"); RES+=("$2"); [[ $2 == PASS* ]] || fails=$((fails + 1)); echo ">>> $1: $2"; }

# start <tag> <binary> <parallel> <maxctx|""> [extra env...]
start() {
  local tag=$1 bin=$2 par=$3 mc=$4; shift 4
  for e in $(compgen -e | grep '^GDEC_'); do unset "$e"; done
  for e in "${PENV[@]}"; do export "$e"; done
  export GDEC_KVSNAP=0 GDEC_NGRAM_CHUNK=16 GDEC_PARALLEL="$par"
  for e in "${CONC_ENV[@]}" "$@"; do export "$e"; done
  local cmd=("${BASE[@]}") i
  cmd[0]="$bin"
  for i in "${!cmd[@]}"; do
    [[ "${cmd[$i]}" == --port ]] && cmd[$((i + 1))]=8732
    [[ -n "$mc" && "${cmd[$i]}" == --maxctx ]] && cmd[$((i + 1))]="$mc"
  done
  probe_start "conc-$tag" "${cmd[@]}"
}
crashed() {
  grep -E 'hipError|HIP error|Segmentation|Aborted|FATAL|GUARD PAGE|terminate called' "$PROBE_LOG" | head -3
}

if want seq && [[ -n "$OLD_BIN" ]]; then
  if start old1 "$OLD_BIN" 1 ""; then
    python3 tools/conc_verify.py seq --tag old1 || note "旧二进制单路" "FAIL（请求异常）"
    probe_stop
  else note "旧二进制单路" "FAIL（引擎启动失败）"; fi
fi

if ! want seq; then :
elif start new1 "$NEW_BIN" 1 ""; then
  if python3 tools/conc_verify.py seq --tag new1; then
    if [[ -n "$OLD_BIN" && -f logs/conc/old1.json ]]; then
      python3 tools/conc_verify.py cmp --a old1 --b new1 && note "单路回归（旧 == 新）" PASS \
        || note "单路回归（旧 == 新）" FAIL
    fi
  else note "新二进制单路" "FAIL（请求异常）"; fi
  c="$(crashed)"; [[ -z "$c" ]] || note "单路引擎日志" "FAIL: $c"
  probe_stop
else note "新二进制单路" "FAIL（引擎启动失败）"; fi

if ! want par; then :
elif start par4 "$NEW_BIN" 4 ""; then
  grep -q '4 slots' "$PROBE_LOG" || note "4 路启动" "FAIL（日志没有 4 slots，槽位未启用）"
  if python3 tools/conc_verify.py conc --tag conc4 --slots 4; then
    [[ -f logs/conc/new1.json ]] && python3 tools/conc_verify.py cmp --a new1 --b conc4 && note "4 路并发 == 单路串行（逐位）" PASS \
      || note "4 路并发 == 单路串行（逐位）" FAIL
  else note "4 路并发" "FAIL（请求异常）"; fi
  python3 tools/conc_verify.py ctl --slots 4 && note "4 路控制（PING/取消/断连）" PASS \
    || note "4 路控制（PING/取消/断连）" FAIL
  c="$(crashed)"; [[ -z "$c" ]] || note "4 路引擎日志" "FAIL: $c"
  probe_stop
else note "4 路" "FAIL（引擎启动失败）"; fi

if ! want ovf; then :
elif start ovf2 "$NEW_BIN" 2 16384 GDEC_RCKPT_MAX=0; then
  if python3 tools/conc_verify.py ovf; then
    if grep -q 'aborting the later request' "$PROBE_LOG"; then note "池溢出中断后来者" PASS
    else note "池溢出中断后来者" "FAIL（日志里没有 aborting the later request）"; fi
  else note "池溢出中断后来者" FAIL; fi
  grep -E 'pool dry|failed:' "$PROBE_LOG" | head -8
  c="$(crashed)"; [[ -z "$c" ]] || note "溢出引擎日志" "FAIL: $c"
  probe_stop
else note "池溢出" "FAIL（引擎启动失败）"; fi

echo
echo "==== 并发验证汇总（NEW_BIN=$NEW_BIN${OLD_BIN:+, OLD_BIN=$OLD_BIN}）===="
for i in "${!NAMES[@]}"; do printf '  %-32s %s\n' "${NAMES[$i]}" "${RES[$i]}"; done
if (( fails == 0 && ${#NAMES[@]} > 0 )); then echo "PASS"; exit 0; else echo "FAIL"; exit 1; fi
