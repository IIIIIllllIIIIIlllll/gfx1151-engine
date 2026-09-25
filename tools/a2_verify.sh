#!/usr/bin/env bash
# 一键 A2 验证（页分配器 + 按需分配），前台运行，最后一行 A2 VERIFY: PASS/FAIL。
# 约 25-35 分钟（A2_BIG=0 可省掉最后一步大 maxctx，约省 5 分钟）。
#   bash tools/a2_verify.sh
#   SKIP_BUILD=1 bash tools/a2_verify.sh     # 已经编译过，跳过编译
#   A2_BIG=0 bash tools/a2_verify.sh         # 不跑 131072 maxctx 那一轮
# 步骤：
#   1. bash build.sh test：ktest 全过，含新增 kvpool_alloc_table 与 A1c 的两个 paged_bits
#   2. bash build.sh engine
#   3. a1_verify.sh prod prodp1 prodp2：不分页 / 按需分配(低页优先) / 按需分配(打乱+FIFO复用)
#      snapshot_regress 9 个用例逐字一致
#   4. prodp2 引擎日志检查：每个请求的物理页确实在轮换（>=2 种 "logical 0 -> phys"）、
#      有 "released" 释放记录；所有日志里没有 GUARD PAGE WRITTEN / FATAL
#   5. btv_kvsnap_verify.sh 两轮：不分页保存→分页(2)恢复；分页(2)保存→不分页恢复
#   6. A1_CTX=131072 下 prod prodp2 再比一次（页表 512 项）
# 前提：生产服务已停（start_hgn.sh / start_gguf.sh 按 Ctrl+C）。
# 回报：从 "==== A2 汇总 ====" 往下的内容。
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p logs
export OLD_BINARY=""

declare -A R
order=()
step() { order+=("$1"); R["$1"]="$2"; }
summary() {
  echo
  echo "==== A2 汇总 ===="
  bad=0
  for k in "${order[@]}"; do
    printf '%-40s %s\n' "$k" "${R[$k]}"
    [[ "${R[$k]}" == PASS* ]] || bad=1
  done
}

if [[ "${SKIP_BUILD:-0}" != 1 ]]; then
  echo "================ 1/6 ktest ================"
  bash build.sh test 2>&1 | tee logs/a2-ktest.txt
  rc=${PIPESTATUS[0]}
  step "ktest（ALL）" "$([[ $rc == 0 ]] && echo PASS || echo FAIL)"
  for n in kvpool_alloc_table qsa_wmma_btv_paged_bits qsa_wmma_rm_paged_bits; do
    l="$(grep -m1 "^$n" logs/a2-ktest.txt || true)"
    step "ktest $n" "$([[ "$l" == *PASS* ]] && echo PASS || echo "FAIL${l:+（$l）}")"
  done
  grep -h '  kvpool:' logs/a2-ktest.txt 2>/dev/null | head -n 10
  echo "================ 2/6 build engine ================"
  bash build.sh engine 2>&1 | tail -n 20
  rc=${PIPESTATUS[0]}
  if [[ $rc != 0 ]]; then
    step "build engine" FAIL
    summary
    echo "A2 VERIFY: FAIL（引擎编译失败，把第一条错误贴回来）"; exit 1
  fi
  step "build engine" PASS
fi

# 旧日志不能混进本次检查
rm -f logs/a1-prod.log logs/a1-prodp1.log logs/a1-prodp2.log logs/btv-*.log

echo
echo "================ 3/6 生产配置：不分页 vs 按需分配 ================"
bash tools/a1_verify.sh prod prodp1 prodp2 2>&1 | tee logs/a2-matrix.txt
rc=${PIPESTATUS[0]}
step "matrix prod prodp1 prodp2" "$([[ $rc == 0 ]] && echo PASS || echo FAIL)"

echo
echo "================ 4/6 分配器日志检查 ================"
L=logs/a1-prodp2.log
if [[ -f $L ]]; then
  nphys="$(grep -o 'seq map: logical 0 -> phys [0-9]*' "$L" | sort -u | wc -l)"
  nmap="$(grep -c 'seq map: logical 0 -> phys' "$L" || true)"
  nrel="$(grep -c '\[kvpage\] released' "$L" || true)"
  echo "prodp2: 序列映射 $nmap 次，logical 0 落在 $nphys 个不同物理页；释放记录 $nrel 条"
  grep -m3 'seq map: logical 0' "$L"; grep -m3 '\[kvpage\] released' "$L"
  step "prodp2 物理页轮换（>=2 种）" "$( (( nphys >= 2 )) && echo "PASS（$nphys 种）" || echo "FAIL（$nphys 种）")"
  step "prodp2 有页释放" "$( (( nrel >= 1 )) && echo "PASS（$nrel 条）" || echo "FAIL（0 条）")"
else
  step "prodp2 引擎日志" "FAIL（没有 $L）"
fi
p1="$(grep -m1 '\[kvpage\]' logs/a1-prodp1.log 2>/dev/null || true)"
step "prodp1 启动行 on-demand alloc=lowest-first" "$([[ "$p1" == *lowest-first* ]] && echo PASS || echo "FAIL（${p1:-无}）")"
# 第 6 步会覆盖 logs/a1-prod*.log，先留一份
rm -rf logs/a2-mid; mkdir -p logs/a2-mid
cp logs/a1-prod.log logs/a1-prodp1.log logs/a1-prodp2.log logs/a2-mid/ 2>/dev/null || true

echo
echo "================ 5/6 kvsnap 跨布局保存/恢复 ================"
echo "---- 5a: 不分页保存 → GDEC_KV_PAGED=2 恢复 ----"
RESTORE_KVP=2 bash tools/btv_kvsnap_verify.sh 2>&1 | tee logs/a2-kvsnap-a.txt
rc=${PIPESTATUS[0]}
step "kvsnap save=off restore=p2" "$([[ $rc == 0 ]] && echo PASS || echo FAIL)"
# btv 脚本每轮都写 logs/btv-{save,restore}.log，每轮留一份给最后的守卫页检查
rm -rf logs/a2-btv; mkdir -p logs/a2-btv
for f in logs/btv-*.log; do [[ -f $f ]] && cp "$f" "logs/a2-btv/a-${f#logs/}"; done
echo "---- 5b: GDEC_KV_PAGED=2 保存 → 不分页恢复 ----"
SAVE_KVP=2 bash tools/btv_kvsnap_verify.sh 2>&1 | tee logs/a2-kvsnap-b.txt
rc=${PIPESTATUS[0]}
step "kvsnap save=p2 restore=off" "$([[ $rc == 0 ]] && echo PASS || echo FAIL)"
for f in logs/btv-*.log; do [[ -f $f ]] && cp "$f" "logs/a2-btv/b-${f#logs/}"; done

if [[ "${A2_BIG:-1}" != 0 ]]; then
  echo
  echo "================ 6/6 maxctx 131072（512 页）================"
  A1_CTX=131072 bash tools/a1_verify.sh prod prodp2 2>&1 | tee logs/a2-big.txt
  rc=${PIPESTATUS[0]}
  step "maxctx 131072 prod vs prodp2" "$([[ $rc == 0 ]] && echo PASS || echo FAIL)"
  nb="$(grep -o 'seq map: logical 0 -> phys [0-9]*' logs/a1-prodp2.log 2>/dev/null | sort -u | wc -l)"
  echo "131072 prodp2: logical 0 落在 $nb 个不同物理页"
fi

echo
echo "================ 守卫页 / FATAL 检查 ================"
hits="$(grep -H 'GUARD PAGE WRITTEN\|\[kvpage\] FATAL\|\[kvpage\] ERROR' \
          logs/a2-mid/*.log logs/a1-prod*.log logs/a2-btv/*.log 2>/dev/null | head -n 10)"
if [[ -n "$hits" ]]; then
  echo "$hits"; step "无 GUARD PAGE WRITTEN / FATAL" FAIL
else
  echo "（无）"; step "无 GUARD PAGE WRITTEN / FATAL" PASS
fi

summary
echo "--- 矩阵表 ---"
sed -n '/==== 汇总 ====/,$p' logs/a2-matrix.txt
[[ -f logs/a2-big.txt ]] && sed -n '/==== 汇总 ====/,$p' logs/a2-big.txt
echo "--- [kvpage] 启动行 ---"
for f in logs/a2-mid/*.log logs/a2-btv/*.log; do
  [[ -f $f ]] && printf '%-34s %s\n' "${f#logs/}" "$(grep -m1 '\[kvpage\]' "$f" || echo '（无 [kvpage] 行 = 不分页）')"
done
if (( bad )); then
  echo "A2 VERIFY: FAIL（见上表；日志在 logs/a2-*.txt、logs/a2-mid/、logs/a2-btv/）"
  exit 1
fi
echo "A2 VERIFY: PASS"
