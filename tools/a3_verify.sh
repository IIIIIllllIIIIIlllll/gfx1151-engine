#!/usr/bin/env bash
# 一键 A3 验证（RAM 检查点钉住 KV 页 + 写时复制 + 池满 LRU 淘汰），前台运行，
# 最后一行 A3 VERIFY: PASS/FAIL。约 30-40 分钟。
#   bash tools/a3_verify.sh
#   SKIP_BUILD=1 bash tools/a3_verify.sh     # 已经编译过，跳过 ktest + 编译
#   A3_MATRIX=0 bash tools/a3_verify.sh      # 跳过第 3 步回归矩阵（约省 10 分钟）
# 步骤：
#   1. bash build.sh test：ktest 全过，含 kvpool_alloc_table 与新增 kvpool_pin_cow
#   2. bash build.sh engine
#   3. a1_verify.sh prod prodp1 prodp2：snapshot_regress 9 个用例逐字一致（回归）
#   4. 切换对话场景（tools/rckpt_ab.py switch），4 个引擎：
#        off=prod 不分页 / p1=prodp1 / p2=prodp2 / p1big=prodp1+GDEC_KV_POOL_TOKENS=81920
#      A → A+问题2 → A+问题3 → B → A+问题2 → A+问题3 → C → A+问题2
#      分页时切走再切回也必须命中 A 的检查点（restored 5 次，不分页只有 2 次），
#      且输出与第一次恢复逐 token 一致；p2 日志里要有写时复制（cow）记录
#   5. 池满淘汰场景（rckpt_ab.py evict），maxctx 16384（64 页）：off vs p2
#      A(8100) → B2(6000) → C2(12000) → A+问题2：p2 至少淘汰 2 个检查点，输出与 off 一致
#   6. 所有日志无 GUARD PAGE WRITTEN / FATAL
# 前提：生产服务已停（start_hgn.sh / start_gguf.sh 按 Ctrl+C）。
# 回报：从 "==== A3 汇总 ====" 往下的内容。
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source tools/probe_lib.sh
mkdir -p logs
export OLD_BINARY=""

declare -A R
order=()
step() { order+=("$1"); R["$1"]="$2"; }
summary() {
  echo
  echo "==== A3 汇总 ===="
  bad=0
  for k in "${order[@]}"; do
    printf '%-44s %s\n' "$k" "${R[$k]}"
    [[ "${R[$k]}" == PASS* ]] || bad=1
  done
}
cnt() { local n; n="$(grep -c -- "$1" "$2" 2>/dev/null)"; echo "${n:-0}"; }

if [[ "${SKIP_BUILD:-0}" != 1 ]]; then
  echo "================ 1/6 ktest ================"
  bash build.sh test 2>&1 | tee logs/a3-ktest.txt
  rc=${PIPESTATUS[0]}
  step "ktest（ALL）" "$([[ $rc == 0 ]] && echo PASS || echo FAIL)"
  for n in kvpool_alloc_table kvpool_pin_cow; do
    l="$(grep -m1 "^$n" logs/a3-ktest.txt || true)"
    step "ktest $n" "$([[ "$l" == *PASS* ]] && echo PASS || echo "FAIL${l:+（$l）}")"
  done
  grep -h '  kvpool' logs/a3-ktest.txt 2>/dev/null | head -n 10
  echo "================ 2/6 build engine ================"
  bash build.sh engine 2>&1 | tail -n 20
  rc=${PIPESTATUS[0]}
  if [[ $rc != 0 ]]; then
    step "build engine" FAIL
    summary
    echo "A3 VERIFY: FAIL（引擎编译失败，把第一条错误贴回来）"; exit 1
  fi
  step "build engine" PASS
fi

probe_precheck || exit 1
for n in 8192 32768; do
  [[ -f data/qsa-oracle/$n.json ]] || { echo "缺少 data/qsa-oracle/$n.json（git checkout HEAD -- data/qsa-oracle）" >&2; exit 1; }
done

# 旧日志不能混进本次检查
rm -f logs/a1-prod.log logs/a1-prodp1.log logs/a1-prodp2.log logs/a3-sw-*.log logs/a3-ev-*.log

if [[ "${A3_MATRIX:-1}" != 0 ]]; then
  echo
  echo "================ 3/6 回归：生产配置 不分页 vs 分页 ================"
  bash tools/a1_verify.sh prod prodp1 prodp2 2>&1 | tee logs/a3-matrix.txt
  rc=${PIPESTATUS[0]}
  step "matrix prod prodp1 prodp2" "$([[ $rc == 0 ]] && echo PASS || echo FAIL)"
  for c in prod prodp1 prodp2; do
    echo "a1-$c: rckpt restored $(cnt 'rckpt: restored' logs/a1-$c.log) 次，cow $(cnt '\[kvpage\] cow' logs/a1-$c.log) 次，evicted $(cnt 'rckpt evicted' logs/a1-$c.log) 次"
  done
fi

# ---- 4/5 步的引擎：与 a1_verify.sh prod* 相同的生产 kernel 环境，关 kvsnap，检查点门槛 1024 ----
MODEL_DIR="${MODEL_DIR:-models}"
PROD_VARS=(GDEC_QSA_KV_BF16 GDEC_QSA_WMMA GDEC_QSA_WMMA_BTV GDEC_GEMM_WMMA GDEC_GDN_FUSED
           GDEC_MOE_LT GDEC_MOE_LT_BF16 GDEC_GR_BF16 GDEC_GDN_STREAM GDEC_GDN_WAVE
           GDEC_NOWARMUP GDEC_INDEX_FUSED2 GDEC_PP_MOE_OUT GDEC_INDEX_STREAM_SELECT)
set_env() {  # <off|p1|p2|p1big>
  unset GDEC_KV_PAGED GDEC_KV_POOL_TOKENS GDEC_PREFILL_CHUNK GDEC_RCKPT GDEC_RCKPT_MAX "${PROD_VARS[@]}"
  local v; for v in "${PROD_VARS[@]}"; do export "$v=1"; done
  export GDEC_PREFILL_CHUNK=16384 GDEC_KVSNAP=0 GDEC_RCKPT_MIN=1024
  case "$1" in
    p1) export GDEC_KV_PAGED=1 ;;
    p2) export GDEC_KV_PAGED=2 ;;
    p1big) export GDEC_KV_PAGED=1 GDEC_KV_POOL_TOKENS=81920 ;;
  esac
}
# run_scn <switch|evict> <tag> <maxctx>
run_scn() {
  local scn=$1 tag=$2 ctx=$3 lt
  lt="a3-${scn:0:2}-$tag"
  set_env "$tag"
  echo
  echo "---- $scn / $tag（GDEC_KV_PAGED=${GDEC_KV_PAGED:-未设置}，POOL_TOKENS=${GDEC_KV_POOL_TOKENS:-默认}，maxctx $ctx）----"
  probe_start "$lt" build/gdec "$MODEL_DIR/qwen38-flash-next-w4b.hgn" \
      "$MODEL_DIR/qwen38-flash-next-w4b.overlay.hgn" --serve --port 8732 --maxctx "$ctx" \
    || { echo "[$lt] 启动失败" >&2; return 1; }
  echo "[$lt] 页表日志: $(grep '\[kvpage\]' "$PROBE_LOG" | head -n 2 | tr '\n' ' ')"
  python3 tools/rckpt_ab.py "$scn" --tag "$tag" 2>&1
  local rc=$?
  grep -E 'hipError|Segmentation|Aborted|Assertion|watchdog fired|\[kvpage\] FATAL' "$PROBE_LOG" | head -n 5 || true
  grep -E 'rckpt: restored|rckpt evicted' "$PROBE_LOG" | head -n 12 || true
  probe_stop
  return $rc
}

trap probe_stop EXIT
trap 'exit 130' INT TERM
python3 tools/rckpt_ab.py reset

echo
echo "================ 4/6 切换对话：检查点跨对话复用 ================"
for t in off p1 p2 p1big; do
  run_scn switch "$t" 40960; rc=$?
  step "switch $t 跑完" "$([[ $rc == 0 ]] && echo PASS || echo "FAIL（见 logs/a3-sw-$t.log）")"
done
n="$(cnt 'rckpt: restored' logs/a3-sw-off.log)"
step "switch off: restored 2 次（行为不变）" "$([[ $n == 2 ]] && echo "PASS" || echo "FAIL（$n 次）")"
for t in p1 p2 p1big; do
  n="$(cnt 'rckpt: restored' logs/a3-sw-$t.log)"
  step "switch $t: restored 5 次（切回命中）" "$([[ $n == 5 ]] && echo "PASS" || echo "FAIL（$n 次）")"
done
n="$(cnt '\[kvpage\] cow' logs/a3-sw-p2.log)"
grep -m3 '\[kvpage\] cow' logs/a3-sw-p2.log 2>/dev/null
step "switch p2: 有写时复制（cow）" "$( (( n >= 1 )) && echo "PASS（$n 次）" || echo "FAIL（0 次）")"
l="$(grep -m1 'pool enlarged' logs/a3-sw-p1big.log 2>/dev/null || true)"
step "switch p1big: 池扩大启动行" "$([[ -n "$l" ]] && echo "PASS（${l#*pool enlarged: }）" || echo "FAIL（无 pool enlarged 行）")"

echo
echo "================ 5/6 池满淘汰：maxctx 16384（64 页）================"
for t in off p2; do
  run_scn evict "$t" 16384; rc=$?
  step "evict $t 跑完" "$([[ $rc == 0 ]] && echo PASS || echo "FAIL（见 logs/a3-ev-$t.log）")"
done
n="$(cnt 'rckpt evicted' logs/a3-ev-p2.log)"
step "evict p2: 淘汰 >=2 个检查点" "$( (( n >= 2 )) && echo "PASS（$n 次）" || echo "FAIL（$n 次）")"
n="$(cnt 'rckpt: restored' logs/a3-ev-p2.log)"
step "evict p2: 被淘汰的 A 不再命中" "$([[ $n == 0 ]] && echo PASS || echo "FAIL（restored $n 次）")"
trap - EXIT

echo
echo "================ 逐 token 比较 ================"
python3 tools/rckpt_ab.py compare 2>&1 | tee logs/a3-compare.txt
rc=${PIPESTATUS[0]}
step "rckpt_ab compare（输出逐 token 一致）" "$([[ $rc == 0 ]] && echo PASS || echo FAIL)"

echo
echo "================ 6/6 守卫页 / FATAL 检查 ================"
hits="$(grep -H 'GUARD PAGE WRITTEN\|\[kvpage\] FATAL\|\[kvpage\] ERROR' \
          logs/a1-prod*.log logs/a3-sw-*.log logs/a3-ev-*.log 2>/dev/null | head -n 10)"
if [[ -n "$hits" ]]; then
  echo "$hits"; step "无 GUARD PAGE WRITTEN / FATAL" FAIL
else
  echo "（无）"; step "无 GUARD PAGE WRITTEN / FATAL" PASS
fi

summary
[[ -f logs/a3-matrix.txt ]] && { echo "--- 回归矩阵 ---"; sed -n '/==== 汇总 ====/,$p' logs/a3-matrix.txt; }
echo "--- 逐 token 比较 ---"; cat logs/a3-compare.txt
echo "--- 检查点计数 ---"
for f in logs/a1-prod*.log logs/a3-sw-*.log logs/a3-ev-*.log; do
  [[ -f $f ]] && printf '%-24s restored %-3s cow %-4s evicted %-3s %s\n' "${f#logs/}" \
    "$(cnt 'rckpt: restored' "$f")" "$(cnt '\[kvpage\] cow' "$f")" "$(cnt 'rckpt evicted' "$f")" \
    "$(grep -m1 '\[kvpage\]' "$f" | cut -c1-70 || true)"
done
if (( bad )); then
  echo "A3 VERIFY: FAIL（见上表；日志在 logs/a3-*.txt、logs/a3-sw-*.log、logs/a3-ev-*.log）"
  exit 1
fi
echo "A3 VERIFY: PASS"
