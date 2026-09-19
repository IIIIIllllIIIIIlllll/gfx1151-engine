#!/usr/bin/env bash
# run_capped.sh — 在受约束的 systemd scope 里跑引擎/重负载，保护桌面会话。
#
# 四层防护（2026-09-12 三次 OOM 后立,09-13 补 PSI 层）:
#   1. MemoryMax:引擎的 cgroup 可计费内存(anon+cache)封顶,超额先在
#      自己的 scope 里被换出/被杀,轮不到 VS Code。
#   2. oom_score_adj=1000:内核 OOM 时优先杀本进程。
#   3. 看门狗:系统 available 低于 6G 时立即杀掉负载进程。
#   4. PSI 看门狗:系统内存 PSI avg10>=45 持续 6s 杀负载。scope 计费内存
#      远低于 MemoryMax 时(如 51G/82G),oomd 的 scope 级 kill 线不触发,
#      会在系统级改杀 VS Code(09-13 07:12 事故);available 被 page cache
#      撑高也看不出 reclaim 抖动,只有 PSI 和 oomd 看的是同一个信号。
# 注意:GPU GTT 锁页(权重 arena ~68G)不计入 cgroup,MemoryMax 管不住它;
# 真正的总量纪律仍是"同一时刻最多一个重进程",本脚本只是保险。
#
# 用法: tools/run_capped.sh [cap_gb] -- <command...>
#   默认 cap 78G(68G arena + KV/工作区余量,32K 测试足够)。
set -u
CAP=78
if [ $# -ge 2 ] && [ "$1" -eq "$1" ] 2>/dev/null; then CAP=$1; shift; fi
[ "${1:-}" = "--" ] && shift
[ $# -ge 1 ] || { echo "usage: run_capped.sh [cap_gb] -- <cmd...>" >&2; exit 2; }

UNIT="capped-$$"
systemd-run --user --scope --unit="$UNIT" \
  -p MemoryMax="${CAP}G" \
  -p ManagedOOMMemoryPressure=kill \
  -p ManagedOOMMemoryPressureLimit=80% \
  -- bash -c 'echo 1000 > /proc/self/oom_score_adj 2>/dev/null; exec "$@"' _ "$@" &
RUNPID=$!

# 看门狗:available < 6G 就杀负载(每 2s 一查)
(
  hits=0
  while kill -0 "$RUNPID" 2>/dev/null; do
    AV=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
    if [ "$AV" -lt 6 ]; then
      echo "run_capped: watchdog fired, available=${AV}G < 6G, killing $UNIT" >&2
      systemctl --user stop "$UNIT.scope" 2>/dev/null
      pkill -KILL -P "$RUNPID" 2>/dev/null
      kill -KILL "$RUNPID" 2>/dev/null
      break
    fi
    # PSI 层:2026-09-13 07:12 事故——scope 计费内存(51G)远低于 MemoryMax(82G),
    # oomd 的 scope 级 kill 线不触发,系统级触发后杀了 VS Code。available 被
    # page cache 撑高,看不出 reclaim 抖动。直接监视 oomd 所看的系统内存 PSI,
    # 抢在 oomd(>50% 持续 20s)之前杀负载。阈值定为 >=48 持续 12s:引擎加载
    # 期有 ~1 分钟的 reclaim 抖动(psi 40-60),加载完自行回落,45/6s 会把
    # 正常加载也杀掉(09-13 07:29 采样实证);12s 窗口仍比 oomd 的 20s 快。
    P=$(awk '/^some/{for(i=1;i<=NF;i++) if($i~/^avg10=/){sub("avg10=","",$i); print int($i)}}' /proc/pressure/memory)
    if [ "${P:-0}" -ge 48 ]; then
      hits=$((hits + 1))
      if [ "$hits" -ge 6 ]; then
        echo "run_capped: PSI watchdog fired, memory avg10=${P}% sustained ${hits}x, killing $UNIT" >&2
        systemctl --user stop "$UNIT.scope" 2>/dev/null
        pkill -KILL -P "$RUNPID" 2>/dev/null
        kill -KILL "$RUNPID" 2>/dev/null
        break
      fi
    else
      hits=0
    fi
    sleep 2
  done
) &
WD=$!
wait "$RUNPID"
RC=$?
kill "$WD" 2>/dev/null
exit $RC
