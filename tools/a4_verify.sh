#!/usr/bin/env bash
# 一键 A4 验证（SSD KV 缓存：按页内容寻址分块 + MTP 段 + 稀疏检查点 + 后台写），
# 前台运行，最后一行 A4 VERIFY: PASS/FAIL。a4 部分约 15 分钟，加 a5 回归约 50 分钟。
#   bash tools/a4_verify.sh
#   NEW_BIN=build/gdec.conc bash tools/a4_verify.sh
#   A4_STAGES="e1 e2 lru bad" bash tools/a4_verify.sh     # 只跑某几段（e2/lru/bad 依赖 e1）
# 快照目录固定为 data/kvsnap-a4（每次从空目录开始，不碰生产的 data/kvsnap）。五段：
#   e1   空目录 + 预埋一个旧格式 s_* 目录和 tmp_ 残留：启动时应被清掉；三组对话
#        （A 带 SNAPS 切点、B 与 A 共享前 6000 token、C 带切点后切回）产生 gold，
#        B 只新写不共享的页；KVSYNC 报的计数/字节 == 磁盘文件
#   e2   重启同一目录：三个新连接各自命中 SSD 检查点（MTP live）；a2/c2 逐 token + spec
#        统计 == e1 的 gold。b2 的检查点借用了 A 算的页（A、B prefill 分段不同，数值
#        不逐位相同，与 vLLM 前缀缓存一样），只要求命中 + 投机，与 gold 的差异只报告；
#        再重启一次（rep）：b2/c2 的恢复 == 第一次恢复（确定性）
#   lru  按磁盘内容算一个上限，使只放得下最新的 2 个检查点；重启后启动时按 LRU
#        淘汰，剩余文件 == 预测；C 的切点检查点仍能恢复且 == gold
#   bad  把所有页文件各翻一个字节：恢复失败（bad page）不致命，请求整段重算完成，
#        坏检查点和页被删掉并重写
#   a5   tools/a5_verify.sh（A5_BIN=NEW）：分页/不分页逐位一致，含 kvsnap + rckpt 同开
# 前提：生产服务已停（start_hgn.sh / start_gguf.sh 按 Ctrl+C）。
# 回报：从 "==== A4 汇总 ====" 往下的内容。
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source tools/probe_lib.sh
NEW_BIN="${NEW_BIN:-build/gdec}"
A4_STAGES=" ${A4_STAGES:-e1 e2 lru bad a5} "
want() { [[ "$A4_STAGES" == *" $1 "* ]]; }
T0=$SECONDS
export PROBE_BINARY="$NEW_BIN"
probe_precheck || exit 1
for f in start_hgn.sh start_gguf.sh tools/serve_common.sh start_win.sh service.conf; do
  [[ -f $f ]] && grep -q $'\r' "$f" && sed -i 's/\r$//' "$f"
done
chk="$(bash start_hgn.sh --check 2>&1)" || { echo "$chk"; echo "A4 VERIFY: FAIL（start_hgn.sh --check 失败）"; exit 1; }
mapfile -t PENV < <(sed -n 's/^ENV //p' <<<"$chk")
CMDLINE="$(sed -n 's/^CMD //p' <<<"$chk")"
eval "BASE=($CMDLINE)"
PROBE_CAP_GB="$(source service.conf; echo "${MEMORY_CAP_GB:-86}")"; export PROBE_CAP_GB
DIR="$PWD/data/kvsnap-a4"
trap 'probe_stop' EXIT
trap 'exit 130' INT TERM
mkdir -p logs

declare -a NAMES RES
fails=0
note() { NAMES+=("$1"); RES+=("$2"); [[ $2 == PASS* ]] || fails=$((fails + 1)); echo ">>> $1: $2"; }
pf() { [[ $1 == 0 ]] && echo PASS || echo "FAIL${2:+（$2）}"; }
cnt() { local n; n="$(grep -c -- "$1" "$2" 2>/dev/null)"; echo "${n:-0}"; }
crashed() {
  grep -E 'hipError|HIP error|Segmentation|Aborted|FATAL|GUARD PAGE|terminate called|watchdog fired' "$PROBE_LOG" | head -3
}
ab() { python3 tools/a4_ab.py "$@" --dir "$DIR"; }

# start <tag> [extra env...]   生产环境 + 端口 8732 + 单槽 + kvsnap 指向 $DIR
start() {
  local tag=$1; shift
  local e cmd=("${BASE[@]}") i
  for e in $(compgen -e | grep '^GDEC_'); do unset "$e"; done
  for e in "${PENV[@]}"; do export "$e"; done
  unset GDEC_KVSNAP GDEC_KVSNAP_MAX_GB
  export GDEC_KVSNAP_DIR="$DIR" GDEC_KVSNAP_MIN=1024 GDEC_PARALLEL=1
  for e in "$@"; do export "$e"; done
  cmd[0]="$NEW_BIN"
  for i in "${!cmd[@]}"; do [[ "${cmd[$i]}" == --port ]] && cmd[$((i + 1))]=8732; done
  probe_start "$tag" "${cmd[@]}"
}
# finish <stage>：崩溃检查 + 停引擎
finish() {
  local c; c="$(crashed)"
  note "$1 引擎日志无崩溃" "$([[ -z "$c" ]] && echo PASS || echo "FAIL: ${c%%$'\n'*}")"
  grep -E 'kvsnap: (index|removed|evicted|restored|restore of|save skipped|capture of)' "$PROBE_LOG" | head -12
  probe_stop
}

if want e1; then
  echo; echo "================ e1：保存（空目录 + 旧格式残留） ================"
  rm -rf "$DIR"; mkdir -p "$DIR/s_8192_deadbeef"
  echo old > "$DIR/s_8192_deadbeef/meta"; echo torn > "$DIR/tmp_c_0.kvc"
  python3 tools/a4_ab.py reset
  if start a4-e1; then
    l="$(grep -m1 'pre-A4 snapshot' "$PROBE_LOG")"
    note "启动时删除旧格式 s_* 目录与 tmp_ 残留" \
      "$([[ -n "$l" && ! -e $DIR/s_8192_deadbeef && ! -e $DIR/tmp_c_0.kvc ]] && echo "PASS（$l）" || echo "FAIL（${l:-无日志}）")"
    ab save; note "e1 三组对话 + 页共享 + gold 投机" "$(pf $?)"
    n="$(cnt 'kvsnap: saved' "$PROBE_LOG")"; k="$(cnt 's, mtp; total' "$PROBE_LOG")"
    note "e1 写出 8 个检查点，都带 MTP 状态" "$( ((n == 8 && k == 8)) && echo "PASS（$n 个）" || echo "FAIL（saved $n，带 mtp $k）")"
    ab check-disk; note "e1 索引 == 磁盘" "$(pf $?)"
    finish e1
  else note "e1 启动" "FAIL（见 logs/a4-e1.log）"; fi
fi

if want e2; then
  echo; echo "================ e2：重启后从 SSD 恢复 ================"
  if start a4-e2; then
    ab run --tag restart; rc=$?
    n="$(cnt 'kvsnap: restored' "$PROBE_LOG")"; k="$(cnt 'kvsnap: restored.*(mtp live)' "$PROBE_LOG")"
    note "e2 三次 SSD 恢复，MTP 都 live" "$( ((rc == 0 && n == 3 && k == 3)) && echo "PASS" || echo "FAIL（rc $rc，restored $n，mtp live $k）")"
    ab compare --tag restart --only a2,c2; note "e2 a2/c2 恢复后续写 == gold（逐 token + spec 统计）" "$(pf $?)"
    # b2 的检查点里 16-22 页是 A 算的（内容寻址去重）：A、B 的 prefill 分段不同，
    # GEMM 按 M 选 kernel，这几页与 B 的 live KV 数值上不逐位相同（前缀缓存的固有性质），
    # 所以 b2 只要求命中 + 投机，与 gold 的差异只报告；确定性由下面的 rep 检查
    ab compare --tag restart --only b2 --hit; note "e2 b2 命中 SSD 检查点且仍在投机（与 gold 不要求逐位）" "$(pf $?)"
    ab check-disk; note "e2 索引 == 磁盘" "$(pf $?)"
    finish e2
  else note "e2 启动" "FAIL（见 logs/a4-e2.log）"; fi
  echo; echo "================ e2 rep：再重启一次，恢复是确定的 ================"
  if start a4-rep; then
    ab run --tag rep --only b2,c2; rc=$?   # c2 最后跑：lru 段要它仍是最新的检查点
    n="$(cnt 'kvsnap: restored.*(mtp live)' "$PROBE_LOG")"
    note "rep 两次 SSD 恢复，MTP 都 live" "$( ((rc == 0 && n == 2)) && echo PASS || echo "FAIL（rc $rc，mtp live $n）")"
    ab compare --tag rep --only b2,c2 --ref restart; note "rep b2/c2 == 第一次恢复（逐 token + spec 统计）" "$(pf $?)"
    ab check-disk; note "rep 索引 == 磁盘" "$(pf $?)"
    finish rep
  else note "rep 启动" "FAIL（见 logs/a4-rep.log）"; fi
fi

if want lru; then
  echo; echo "================ lru：容量上限 → 启动时 LRU 淘汰 ================"
  out="$(ab predict --keep 2)"; echo "$out"; cap="$(tail -n 1 <<<"$out")"
  if [[ "$cap" =~ ^[0-9.]+$ ]] && start a4-lru GDEC_KVSNAP_MAX_GB="$cap"; then
    n="$(cnt 'kvsnap: evicted' "$PROBE_LOG")"
    note "lru 上限 $cap GiB：启动时淘汰旧检查点" "$( ((n >= 1)) && echo "PASS（$n 个）" || echo "FAIL（没有 evicted 日志）")"
    ab check-lru; note "lru 剩余文件 == 预测（最新 2 个检查点及其页）" "$(pf $?)"
    ab run --tag lru --only c2 && ab compare --tag lru --only c2
    note "lru 后 C 切点检查点仍可恢复且 == gold" "$(pf $?)"
    ab check-disk; note "lru 索引 == 磁盘" "$(pf $?)"
    finish lru
  else note "lru 启动（cap '$cap'）" "FAIL（见 logs/a4-lru.log）"; fi
fi

if want bad; then
  echo; echo "================ bad：页文件损坏 ================"
  ab corrupt
  if start a4-bad; then
    before="$(ab sync)"
    ab run --tag bad --only c2 && ab compare --tag bad --only c2 --miss
    rc=$?
    l="$(grep -m1 'kvsnap: restore of .*bad page' "$PROBE_LOG")"
    note "bad 坏页被校验和识破，请求整段重算完成" "$( ((rc == 0)) && [[ -n "$l" ]] && echo PASS || echo "FAIL（rc $rc，${l:-无 bad page 日志}）")"
    after="$(ab sync)"
    echo "  KVSYNC 前 $before"; echo "  KVSYNC 后 $after"
    [[ "$after" == *"'ckpts': 1,"* ]]; note "bad 坏检查点删除，只剩本次新存的 1 个" "$(pf $? "$after")"
    ab check-disk; note "bad 索引 == 磁盘" "$(pf $?)"
    finish bad
  else note "bad 启动" "FAIL（见 logs/a4-bad.log）"; fi
fi
[[ " $A4_STAGES " == *" a5 "* ]] || rm -rf "$DIR"

if want a5; then
  echo; echo "================ a5：tools/a5_verify.sh（A5_BIN=$NEW_BIN） ================"
  probe_stop
  rm -rf "$DIR"
  SKIP_BUILD=1 A5_BIN="$NEW_BIN" bash tools/a5_verify.sh 2>&1 | tee logs/a4-a5.txt
  rc=${PIPESTATUS[0]}
  note "a5_verify（分页 p1/p2 == 不分页 off，含 kvsnap）" "$([[ $rc == 0 ]] && echo PASS || echo "FAIL（见 logs/a4-a5.txt）")"
fi

echo
echo "==== A4 汇总（NEW_BIN=$NEW_BIN，用时 $(( (SECONDS - T0) / 60 )) 分钟）===="
for i in "${!NAMES[@]}"; do printf '  %-48s %s\n' "${NAMES[$i]}" "${RES[$i]}"; done
if (( fails == 0 && ${#NAMES[@]} > 0 )); then echo "A4 VERIFY: PASS"; exit 0; fi
echo "A4 VERIFY: FAIL"; exit 1
