#!/usr/bin/env bash
# 一键 B0 验证（MTP KV 进共享页池 + rckpt 恢复后保持 MTP 投机），前台运行，
# 最后一行 B0 VERIFY: PASS/FAIL。全部跑完约 60-80 分钟。
#   bash tools/b0_verify.sh
#   NEW_BIN=build/gdec.conc REF_BIN=build/gdec.ref bash tools/b0_verify.sh
#   B0_STAGES="mem ab" bash tools/b0_verify.sh     # 只跑某几段
# REF_BIN = B0 之前的二进制（同一套源码只差 B0 的提交）。四段：
#   mem  PARALLEL=4 启动 REF 与 NEW，比较 GPU 占用：旧版每个额外槽位按 MAX_CONTEXT
#        预分配 MTP KV（256K 时 BF16 0.5 GiB / FP32 1 GiB）+ MTP keys 32 MiB，现在 MTP
#        层在共享池里；NEW 至少要少预期值的 90%（BF16 生产配置约 1.6 GiB）
#   conc tools/conc_verify.sh：NEW 单路 == REF 单路（逐 token + spec 统计），
#        4 路并发 == 单路，池溢出中断后来者
#   a5   tools/a5_verify.sh（A5_BIN=NEW）：分页 p1/p2 与不分页 off 逐位一致，
#        含 MTP/chain/采样/SNAPS 切点/多轮切回
#   ab   tools/b0_ab.py：drafter 1/4 的切换对话场景，NEW 恢复 rckpt 后仍投机，
#        两次恢复同一检查点逐位一致，接受率与整段重算相当；SNAPS 切点检查点同样可投机
# 前提：生产服务已停（start_hgn.sh / start_gguf.sh 按 Ctrl+C）。
# 回报：从 "==== B0 汇总 ====" 往下的内容。
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source tools/probe_lib.sh
NEW_BIN="${NEW_BIN:-build/gdec}"
REF_BIN="${REF_BIN:-build/gdec.ref}"
B0_STAGES=" ${B0_STAGES:-mem conc a5 ab} "
want() { [[ "$B0_STAGES" == *" $1 "* ]]; }
T0=$SECONDS
export PROBE_BINARY="$NEW_BIN"
probe_precheck || exit 1
[[ -x "$REF_BIN" ]] || { echo "找不到 REF_BIN=$REF_BIN（B0 之前的引擎二进制）" >&2; exit 1; }
for f in start_hgn.sh start_gguf.sh tools/serve_common.sh start_win.sh service.conf; do
  [[ -f $f ]] && grep -q $'\r' "$f" && sed -i 's/\r$//' "$f"
done
chk="$(bash start_hgn.sh --check 2>&1)" || { echo "$chk"; echo "B0 VERIFY: FAIL（start_hgn.sh --check 失败）"; exit 1; }
mapfile -t PENV < <(sed -n 's/^ENV //p' <<<"$chk")
CMDLINE="$(sed -n 's/^CMD //p' <<<"$chk")"
eval "BASE=($CMDLINE)"
PROBE_CAP_GB="$(source service.conf; echo "${MEMORY_CAP_GB:-86}")"; export PROBE_CAP_GB
trap 'probe_stop' EXIT
trap 'exit 130' INT TERM
mkdir -p logs

declare -a NAMES RES
fails=0
note() { NAMES+=("$1"); RES+=("$2"); [[ $2 == PASS* ]] || fails=$((fails + 1)); echo ">>> $1: $2"; }
cnt() { local n; n="$(grep -c -- "$1" "$2" 2>/dev/null)"; echo "${n:-0}"; }
crashed() {
  grep -E 'hipError|HIP error|Segmentation|Aborted|FATAL|GUARD PAGE|terminate called|watchdog fired' "$PROBE_LOG" | head -3
}

# start <tag> <binary> [extra env...]   生产环境 + 端口 8732 + kvsnap 关
start() {
  local tag=$1 bin=$2; shift 2
  local e cmd=("${BASE[@]}") i
  for e in $(compgen -e | grep '^GDEC_'); do unset "$e"; done
  for e in "${PENV[@]}"; do export "$e"; done
  export GDEC_KVSNAP=0
  for e in "$@"; do export "$e"; done
  cmd[0]="$bin"
  for i in "${!cmd[@]}"; do [[ "${cmd[$i]}" == --port ]] && cmd[$((i + 1))]=8732; done
  probe_start "$tag" "${cmd[@]}"
}
# 引擎就绪后的 GPU 占用（GiB）：amdgpu sysfs 的 VRAM + GTT（Linux；Windows 版看 devarena 日志）
gpu_used() {
  local s=0 f
  for f in /sys/class/drm/card*/device/mem_info_vram_used /sys/class/drm/card*/device/mem_info_gtt_used; do
    [[ -r $f ]] && s=$((s + $(cat "$f")))
  done
  awk -v s="$s" 'BEGIN{printf "%.2f", s / 1073741824}'
}

if want mem; then
  echo; echo "================ mem：PARALLEL=4 显存占用 ================"
  declare -A MEM
  for t in ref new; do
    b="$REF_BIN"; [[ $t == new ]] && b="$NEW_BIN"
    if start "b0-mem-$t" "$b" GDEC_PARALLEL=4; then
      sleep 3
      MEM[$t]="$(gpu_used)"
      echo "[b0-mem-$t] GPU 占用（VRAM+GTT）${MEM[$t]} GiB"
      grep -E 'memory\[slots\]|sequence slots' "$PROBE_LOG" | head -2
      c="$(crashed)"; [[ -z "$c" ]] || note "mem $t 引擎日志" "FAIL: $c"
      probe_stop
    else note "mem $t 启动（PARALLEL=4）" "FAIL（见 logs/b0-mem-$t.log）"; fi
  done
  # 预期节省：3 个额外槽位 ×（MTP K+V：maxctx×512×2×元素字节 + keys：maxctx/4×128×4）
  mc=262144 kb=4 i
  for i in "${!BASE[@]}"; do [[ "${BASE[$i]}" == --maxctx ]] && mc="${BASE[$((i + 1))]}"; done
  printf '%s\n' "${PENV[@]}" | grep -q '^GDEC_QSA_KV_BF16=' && kb=2
  exp="$(awk -v m="$mc" -v k="$kb" 'BEGIN{printf "%.2f", 3 * (m*512*2*k + m/4*128*4) / 1073741824}')"
  a="${MEM[ref]:-}"; b="${MEM[new]:-}"
  if [[ -n "$a" && -n "$b" && "$a" != 0.00 ]]; then
    d="$(awk -v a="$a" -v b="$b" 'BEGIN{printf "%.2f", a-b}')"
    k="PARALLEL=4 GPU 占用下降（预期 $exp GiB）"
    if awk -v d="$d" -v e="$exp" 'BEGIN{exit !(d >= 0.9 * e)}'; then note "$k" "PASS（$a → $b GiB，省 $d GiB）"
    else note "$k" "FAIL（$a → $b GiB，省 $d GiB）"; fi
  else note "PARALLEL=4 GPU 占用对比" "FAIL（读不到 sysfs 占用：ref='$a' new='$b'）"; fi
fi

if want conc; then
  echo; echo "================ conc：tools/conc_verify.sh（NEW vs REF） ================"
  probe_stop
  STAGES="seq par ovf" NEW_BIN="$NEW_BIN" OLD_BIN="$REF_BIN" bash tools/conc_verify.sh 2>&1 | tee logs/b0-conc.txt
  rc=${PIPESTATUS[0]}
  note "conc_verify（单路 NEW==REF、4 路==单路、溢出）" "$([[ $rc == 0 ]] && echo PASS || echo "FAIL（见 logs/b0-conc.txt）")"
fi

if want a5; then
  echo; echo "================ a5：tools/a5_verify.sh（A5_BIN=$NEW_BIN） ================"
  SKIP_BUILD=1 A5_BIN="$NEW_BIN" bash tools/a5_verify.sh 2>&1 | tee logs/b0-a5.txt
  rc=${PIPESTATUS[0]}
  note "a5_verify（分页 p1/p2 == 不分页 off）" "$([[ $rc == 0 ]] && echo PASS || echo "FAIL（见 logs/b0-a5.txt）")"
fi

if want ab; then
  echo; echo "================ ab：切换对话 + MTP 投机（tools/b0_ab.py） ================"
  python3 tools/b0_ab.py reset
  for t in ck nock ref; do
    b="$NEW_BIN"; ex=(GDEC_RCKPT_MIN=1024)
    [[ $t == ref ]] && b="$REF_BIN"
    [[ $t == nock ]] && ex=(GDEC_RCKPT=0)
    if start "b0-ab-$t" "$b" "${ex[@]}"; then
      python3 tools/b0_ab.py run --tag "$t" || note "ab $t 请求" "FAIL（请求异常）"
      echo "[b0-ab-$t] rckpt restored $(cnt 'rckpt: restored' "$PROBE_LOG") 次（mtp live $(cnt '(mtp live)' "$PROBE_LOG") 次），cow $(cnt '\[kvpage\] cow' "$PROBE_LOG") 次"
      c="$(crashed)"; [[ -z "$c" ]] || note "ab $t 引擎日志" "FAIL: $c"
      probe_stop
    else note "ab $t 启动" "FAIL（见 logs/b0-ab-$t.log）"; fi
  done
  n="$(cnt '(mtp live)' logs/b0-ab-ck.log)"
  note "NEW 恢复检查点时 MTP 保持 live（6 次）" "$( (( n >= 6 )) && echo "PASS（$n 次）" || echo "FAIL（$n 次）")"
  python3 tools/b0_ab.py compare 2>&1 | tee logs/b0-ab-compare.txt
  rc=${PIPESTATUS[0]}
  note "b0_ab compare（恢复后投机、重复恢复逐位一致、接受率）" "$([[ $rc == 0 ]] && echo PASS || echo FAIL)"
fi

echo
echo "==== B0 汇总（NEW_BIN=$NEW_BIN, REF_BIN=$REF_BIN，用时 $(( (SECONDS - T0) / 60 )) 分钟）===="
for i in "${!NAMES[@]}"; do printf '  %-48s %s\n' "${NAMES[$i]}" "${RES[$i]}"; done
if (( fails == 0 && ${#NAMES[@]} > 0 )); then echo "B0 VERIFY: PASS"; exit 0; fi
echo "B0 VERIFY: FAIL"; exit 1
