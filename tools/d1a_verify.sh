#!/usr/bin/env bash
# D1a 一键验收（并发时 prefill 用更小的分段），前台运行约 7 分钟，最后一行 D1A VERIFY: PASS/FAIL。
#   bash tools/d1a_verify.sh
#   BIN=build/qwenox.x CONC_CHUNK=8192 STALL_MAX=1.0 PP_MAX=10 bash tools/d1a_verify.sh
# 用生产环境（start_hgn.sh --check）+ PARALLEL=2 起两次测试引擎（端口 8732，kvsnap/RAM 检查点关，
# 开 warmup 以免首个大 chunk 的一次性开销算进 PP）：
#   c16  QWENOX_CONC_PREFILL_CHUNK=0     → 并发时仍用 16384（D1a 之前的行为）
#   c8   QWENOX_CONC_PREFILL_CHUNK=8192  → D1a
# 每个引擎：A 单独 decode 1536 token → B 单独 32K prefill → A decode 期间提交 B。
# 判定：测量有效；A 并发 == A 单独（逐 token）；c8 的最大停顿 < STALL_MAX 秒；32K 单独 prefill 变慢 ≤ PP_MAX%。
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source tools/probe_lib.sh
BIN="${BIN:-build/qwenox-engine}"
CONC_CHUNK="${CONC_CHUNK:-8192}"
export PROBE_BINARY="$BIN"
probe_precheck || { echo "D1A VERIFY: FAIL（预检未通过）"; exit 1; }
for f in start_hgn.sh start_gguf.sh tools/serve_common.sh service.conf; do grep -q $'\r' "$f" && sed -i 's/\r$//' "$f"; done
chk="$(bash start_hgn.sh --check 2>&1)" || { echo "$chk"; echo "D1A VERIFY: FAIL（start_hgn.sh --check）"; exit 1; }
mapfile -t PENV < <(sed -n 's/^ENV //p' <<<"$chk")
CMDLINE="$(sed -n 's/^CMD //p' <<<"$chk")"
eval "BASE=($CMDLINE)"
trap 'probe_stop' EXIT
trap 'exit 130' INT TERM
mkdir -p logs/d1a
fail=0

run_one() {  # run_one <tag> <QWENOX_CONC_PREFILL_CHUNK>
  local tag=$1 cc=$2 e i
  for e in $(compgen -e | grep '^QWENOX_'); do unset "$e"; done
  for e in "${PENV[@]}"; do export "$e"; done
  export QWENOX_KVSNAP=0 QWENOX_RCKPT=0 QWENOX_NGRAM_CHUNK=16 QWENOX_PARALLEL=2 QWENOX_CONC_PREFILL_CHUNK="$cc"
  unset QWENOX_NOWARMUP
  local cmd=("${BASE[@]}")
  cmd[0]="$BIN"
  for i in "${!cmd[@]}"; do [[ "${cmd[$i]}" == --port ]] && cmd[$((i + 1))]=8732; done
  probe_start "d1a-$tag" "${cmd[@]}" || return 1
  grep -q 'dense: 8-bit' "$PROBE_LOG" || echo "[$tag] 提示：日志里没有 HQ 标志 'dense: 8-bit'（service.conf 可能不是 HQ 权重）"
  if (( cc > 0 )); then
    grep -q "prefill chunk $cc" "$PROBE_LOG" || { echo "[$tag] 日志里没有 'prefill chunk $cc'：$BIN 不含 D1a"; probe_stop; return 1; }
  else
    grep -q 'QWENOX_CONC_PREFILL_CHUNK' "$PROBE_LOG" && { echo "[$tag] CONC_PREFILL_CHUNK=0 却生效了"; probe_stop; return 1; }
  fi
  python3 tools/d1a_verify.py run --tag "$tag"
  local rc=$?
  echo "[$tag] 让出次数: $(grep -c 'prefill lends GPU' "$PROBE_LOG")"
  grep -E 'hipError|Segmentation|Aborted|FATAL|GUARD PAGE' "$PROBE_LOG" | head -3
  probe_stop
  return $rc
}

rm -f logs/d1a/c16.json logs/d1a/c8.json
run_one c16 0 || fail=1
(( fail )) || run_one c8 "$CONC_CHUNK" || fail=1
if (( fail )); then echo "D1A VERIFY: FAIL（引擎或测量异常，见上）"; exit 1; fi
python3 tools/d1a_verify.py gate --a c16 --b c8 --stall-max "${STALL_MAX:-1.0}" --pp-max "${PP_MAX:-10}"
