#!/usr/bin/env bash
# 一键 A5 验证（分页 KV 在生产环境默认开启），前台运行，最后一行 A5 VERIFY: PASS/FAIL。
# 约 25-35 分钟。
#   bash tools/a5_verify.sh
#   SKIP_BUILD=1 bash tools/a5_verify.sh     # 已经编译过，跳过 ktest + 编译
#   A5_CTX=65536 bash tools/a5_verify.sh     # 用更小的 maxctx 快速跑（默认 = service.conf 的 MAX_CONTEXT）
#   A5_BIN=build/gdec.conc SKIP_BUILD=1 bash tools/a5_verify.sh   # 测另一个引擎二进制
# 之前 A1–A3 的分页测试全是串行解码。生产实际走的是 chain/MTP 投机 + 验证回滚、
# 采样、多轮续写、SNAPS 切点、kvsnap + rckpt 同时开、MTP 权重和视觉塔都加载、
# maxctx 256K。本脚本用 start.sh 的真实环境变量和命令行（只换端口 8732）跑三个引擎：
#   off = KV_PAGED=0（不分页，基准）
#   p1  = 生产默认（GDEC_KV_PAGED=1）
#   p2  = GDEC_KV_PAGED=2（打乱物理页 + FIFO 自检）
# 每个引擎：tools/a5_ab.py run（32 个请求：新 prompt × 串行/MTP/ngram/chain、采样、
# 多轮对话 A→B→切回 A）+ tools/ngram_regress.py（投机回滚/EOS/取消/续写/采样回退）。
# 判定：p1、p2 与 off 逐 token 一致、spec 统计一致（切回 A 的两轮只比较 p1 vs p2，
# 且必须命中缓存）；日志里页表启动行正确、切回时 rckpt 恢复、无守卫页/FATAL。
# 前提：生产服务已停（start.sh Ctrl+C）。
# 回报：从 "==== A5 汇总 ====" 往下的内容。
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source tools/probe_lib.sh
mkdir -p logs
T0=$SECONDS

declare -A R
order=()
step() { order+=("$1"); R["$1"]="$2"; }
summary() {
  echo
  echo "==== A5 汇总 ===="
  bad=0
  for k in "${order[@]}"; do
    printf '%-46s %s\n' "$k" "${R[$k]}"
    [[ "${R[$k]}" == PASS* ]] || bad=1
  done
}
cnt() { local n; n="$(grep -c -- "$1" "$2" 2>/dev/null)"; echo "${n:-0}"; }
die() { step "$1" "FAIL（$2）"; summary; echo "A5 VERIFY: FAIL（$2）"; exit 1; }

if [[ "${SKIP_BUILD:-0}" != 1 ]]; then
  echo "================ 1/5 ktest ================"
  bash build.sh test 2>&1 | tee logs/a5-ktest.txt
  rc=${PIPESTATUS[0]}
  step "ktest（ALL）" "$([[ $rc == 0 ]] && echo PASS || echo FAIL)"
  echo "================ 2/5 build engine ================"
  bash build.sh engine 2>&1 | tail -n 20
  rc=${PIPESTATUS[0]}
  [[ $rc == 0 ]] || die "build engine" "引擎编译失败，把第一条错误贴回来"
  step "build engine" PASS
fi

probe_precheck || exit 1
for n in 8192 32768; do
  [[ -f data/qsa-oracle/$n.json ]] || { echo "缺少 data/qsa-oracle/$n.json（git checkout HEAD -- data/qsa-oracle）" >&2; exit 1; }
done

echo
echo "================ 3/5 start.sh --check：取生产环境 ================"
# 在 Windows 上编辑过的文件可能带 CRLF 换行，bash 会报 "set: pipefail 无效的选项名"，
# source service.conf 也会把 \r 带进每个值。这里直接转成 LF。
for f in start.sh start_win.sh service.conf; do
  if [[ -f $f ]] && grep -q $'\r' "$f"; then
    sed -i 's/\r$//' "$f" && echo "已把 $f 的 CRLF 换行转成 LF"
  fi
done
# start.sh 要求 build/gdec 和 build/gdec-api 都在；测试用的 checkout 往往只编过引擎。
# 本脚本不启动 API，这里只是为了让 --check 通过（SKIP_BUILD=1 时也会补编）。
if [[ ! -x build/gdec-api ]]; then
  echo "build/gdec-api 不存在（start.sh --check 需要），编译 API 中……"
  bash build.sh api 2>&1 | tail -n 5
  rc=${PIPESTATUS[0]}
  [[ $rc == 0 && -x build/gdec-api ]] || die "build api" "API 编译失败，把上面的错误贴回来"
  step "build api（start.sh 需要）" PASS
fi
chk="$(bash start.sh --check 2>&1)"; rc=$?
echo "$chk" | grep -v '^ENV \|^CMD '
[[ $rc == 0 ]] || die "start.sh --check" "start.sh --check 失败，见上面的错误"
mapfile -t PENV < <(sed -n 's/^ENV //p' <<<"$chk")
CMDLINE="$(sed -n 's/^CMD //p' <<<"$chk")"
[[ ${#PENV[@]} -gt 0 && -n "$CMDLINE" ]] || die "start.sh --check" "没有 ENV/CMD 输出（start.sh 是旧版？）"
printf '  %s\n' "${PENV[@]}"
if printf '%s\n' "${PENV[@]}" | grep -qx 'GDEC_KV_PAGED=1'; then
  step "生产环境默认 GDEC_KV_PAGED=1" PASS
else
  die "生产环境默认 GDEC_KV_PAGED=1" "start.sh 没有导出 GDEC_KV_PAGED=1，检查 service.conf 里的 KV_PAGED"
fi
eval "ENGINE=($CMDLINE)"
[[ -n "${A5_BIN:-}" ]] && ENGINE[0]="$A5_BIN"   # 测试其他引擎二进制（如 build/gdec.conc）
CTX="${A5_CTX:-}"
for i in "${!ENGINE[@]}"; do
  case "${ENGINE[$i]}" in
    --port) ENGINE[$((i + 1))]=8732 ;;
    --maxctx) if [[ -n "$CTX" ]]; then ENGINE[$((i + 1))]="$CTX"; else CTX="${ENGINE[$((i + 1))]}"; fi ;;
  esac
done
[[ "$CTX" =~ ^[1-9][0-9]*$ ]] || die "engine 命令行" "取不到 maxctx"
(( CTX >= 40960 )) || die "engine 命令行" "maxctx $CTX 太小，32K 用例需要至少 40960"
POOL="$(printf '%s\n' "${PENV[@]}" | sed -n 's/^GDEC_KV_POOL_TOKENS=//p')"
PAGES=$(( ((POOL > CTX ? POOL : CTX) + 255) / 256 ))
PROBE_CAP_GB="$(source service.conf; echo "${MEMORY_CAP_GB:-86}")"
export PROBE_CAP_GB
echo "引擎命令：${ENGINE[*]}"
echo "maxctx $CTX，页池 ${POOL:-0} token → 期望 $PAGES 页；内存上限 ${PROBE_CAP_GB} GiB"
[[ " ${ENGINE[*]} " == *" --vision-tower "* ]] || echo "注意：没有加载视觉塔（VISION_FILE 为空）"

SNAPDIR="$PWD/data/kvsnap-a5"
set_env() {  # <off|p1|p2>
  local v
  for v in $(compgen -e | grep '^GDEC_'); do unset "$v"; done
  for v in "${PENV[@]}"; do export "$v"; done
  export GDEC_KVSNAP_DIR="$SNAPDIR"     # 不碰生产的 data/kvsnap
  # 单槽：off（不分页）只能跑 1 个槽；多槽时切回对话 A 会被派到还留着 A 的槽（cont），
  # 不走 rckpt 恢复，缓存命中也和 off 不同。并发由 tools/conc_verify.sh 覆盖。
  export GDEC_PARALLEL=1
  case "$1" in
    off) unset GDEC_KV_PAGED GDEC_KV_POOL_TOKENS ;;
    p2) export GDEC_KV_PAGED=2 ;;
  esac
}

rm -f logs/a5-off.log logs/a5-p1.log logs/a5-p2.log
python3 tools/a5_ab.py reset
trap 'probe_stop; rm -rf "$SNAPDIR"' EXIT
trap 'exit 130' INT TERM

echo
echo "================ 4/5 三个引擎：off / p1 / p2 ================"
declare -A NG
for t in off p1 p2; do
  set_env "$t"
  rm -rf "$SNAPDIR"; mkdir -p "$SNAPDIR"
  echo
  echo "---- $t（GDEC_KV_PAGED=${GDEC_KV_PAGED:-未设置}，已用 $(( (SECONDS - T0) / 60 )) 分钟）----"
  if ! probe_start "a5-$t" "${ENGINE[@]}"; then
    step "$t 引擎启动" "FAIL（见 logs/a5-$t.log）"; continue
  fi
  step "$t 引擎启动" PASS
  echo "[a5-$t] 页表日志: $(grep '\[kvpage\]' "$PROBE_LOG" | head -n 2 | tr '\n' ' ')"
  python3 tools/a5_ab.py run --tag "$t" 2>&1
  rc=$?
  step "$t a5_ab run（32 个请求）" "$([[ $rc == 0 ]] && echo PASS || echo "FAIL（rc $rc）")"
  python3 tools/ngram_regress.py --keep-going --output "logs/a5-$t-ngram.json" 2>&1 | tail -n 12
  NG[$t]=${PIPESTATUS[0]}
  echo "[a5-$t] ngram_regress rc=${NG[$t]}"
  crash="$(grep -E 'hipError|Segmentation|Aborted|Assertion|watchdog fired|\[kvpage\] FATAL' "$PROBE_LOG" | head -n 5)"
  [[ -z "$crash" ]] || echo "$crash"
  step "$t 无崩溃" "$([[ -z "$crash" ]] && echo PASS || echo "FAIL（${crash%%$'\n'*}）")"
  grep -E 'rckpt: restored|kvsnap: restored' "$PROBE_LOG" | head -n 6 || true
  probe_stop
done
trap - EXIT
rm -rf "$SNAPDIR"

echo
echo "================ 5/5 日志检查 + 逐 token 比较 ================"
L=logs
l="$(grep -m1 'paged QSA KV on' $L/a5-off.log 2>/dev/null || true)"
step "off: 没有开分页" "$([[ -z "$l" ]] && echo PASS || echo "FAIL（$l）")"
l="$(grep -m1 'paged QSA KV on' $L/a5-p1.log 2>/dev/null || true)"
step "p1: 分页 $PAGES 页 lowest-first" \
  "$([[ "$l" == *"on: $PAGES pages"*lowest-first* ]] && echo PASS || echo "FAIL（${l:-无启动行}）")"
l="$(grep -m1 'paged QSA KV on' $L/a5-p2.log 2>/dev/null || true)"
step "p2: 分页 $PAGES 页 scrambled" \
  "$([[ "$l" == *"on: $PAGES pages"*scrambled* ]] && echo PASS || echo "FAIL（${l:-无启动行}）")"
l="$(grep -h -m1 'GDEC_KV_PAGED ignored' $L/a5-p1.log $L/a5-p2.log 2>/dev/null || true)"
step "生产 kernel 组合支持分页（无 ignored）" "$([[ -z "$l" ]] && echo PASS || echo "FAIL（$l）")"
for t in p1 p2; do
  n="$(cnt 'rckpt: restored' $L/a5-$t.log)"
  step "$t: 切回对话 A 时 rckpt 恢复" "$( (( n >= 1 )) && echo "PASS（$n 次）" || echo "FAIL（0 次）")"
done
v="${NG[off]:-?}/${NG[p1]:-?}/${NG[p2]:-?}"
step "ngram_regress 退出码三者相同（off/p1/p2）" \
  "$([[ "${NG[off]:-x}" == "${NG[p1]:-y}" && "${NG[off]:-x}" == "${NG[p2]:-z}" ]] && echo "PASS（$v）" || echo "FAIL（$v）")"
hits="$(grep -H 'GUARD PAGE WRITTEN\|\[kvpage\] FATAL\|\[kvpage\] ERROR' $L/a5-off.log $L/a5-p1.log $L/a5-p2.log 2>/dev/null | head -n 10)"
[[ -z "$hits" ]] || echo "$hits"
step "无 GUARD PAGE WRITTEN / FATAL" "$([[ -z "$hits" ]] && echo PASS || echo FAIL)"
python3 tools/a5_ab.py compare 2>&1 | tee logs/a5-compare.txt
rc=${PIPESTATUS[0]}
step "a5_ab compare（逐 token + spec 统计一致）" "$([[ $rc == 0 ]] && echo PASS || echo FAIL)"

summary
echo "--- 逐 token 比较 ---"; cat logs/a5-compare.txt
echo "--- 缓存计数 ---"
for t in off p1 p2; do
  f=$L/a5-$t.log
  [[ -f $f ]] && printf '%-4s rckpt restored %-3s evicted %-3s kvsnap saved %-3s restored %-3s ngram rc %s  %s\n' "$t" \
    "$(cnt 'rckpt: restored' "$f")" "$(cnt 'rckpt evicted' "$f")" \
    "$(cnt 'kvsnap: saved' "$f")" "$(cnt 'kvsnap: restored' "$f")" "${NG[$t]:-?}" \
    "$(grep -m1 '\[kvpage\] paged' "$f" | cut -c1-60 || true)"
done
echo "用时 $(( (SECONDS - T0) / 60 )) 分钟"
if (( bad )); then
  echo "A5 VERIFY: FAIL（见上表；日志在 logs/a5-*.log、logs/a5-compare.txt）"
  exit 1
fi
echo "A5 VERIFY: PASS"
