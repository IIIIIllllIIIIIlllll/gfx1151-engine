#!/usr/bin/env bash
# 一键 A1c 验证（生产配置 BF16 KV + WMMA + BTV 下的分页），前台运行，最后一行
# A1C VERIFY: PASS/FAIL。约 20-30 分钟。
#   bash tools/a1c_verify.sh
#   OLD_BINARY=~/gdec-base/build/gdec bash tools/a1c_verify.sh   # 顺带做 A1b 的新旧二进制对照
#   SKIP_BUILD=1 bash tools/a1c_verify.sh                         # 已经编译过，跳过编译
# 步骤：
#   1. bash build.sh test：ktest 全过，含新增 qsa_wmma_btv_paged_bits / qsa_wmma_rm_paged_bits
#   2. bash build.sh engine
#   3. a1_verify.sh prod prodp1 prodp2 [prodold]：不分页 / 恒等页表 / 反转页表 [/ 改动前二进制]
#      snapshot_regress 9 个用例逐字一致
#   4. btv_kvsnap_verify.sh 两轮：不分页保存→反转页表恢复；反转页表保存→不分页恢复
# 前提：生产服务已停（start_hgn.sh / start_gguf.sh 按 Ctrl+C）。
# 回报：从 "==== A1c 汇总 ====" 往下的内容。
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p logs

if [[ -z "${OLD_BINARY:-}" && -x "$HOME/gdec-base/build/gdec" ]]; then
  OLD_BINARY="$HOME/gdec-base/build/gdec"
  echo "发现 $OLD_BINARY，顺带跑 prodold（A1b 新旧二进制对照）"
fi
export OLD_BINARY="${OLD_BINARY:-}"

declare -A R
order=()
step() { order+=("$1"); R["$1"]="$2"; }

if [[ "${SKIP_BUILD:-0}" != 1 ]]; then
  echo "================ 1/4 ktest ================"
  bash build.sh test 2>&1 | tee logs/a1c-ktest.txt
  rc=${PIPESTATUS[0]}
  step "ktest（ALL）" "$([[ $rc == 0 ]] && echo PASS || echo FAIL)"
  for n in qsa_wmma_btv_paged_bits qsa_wmma_rm_paged_bits; do
    l="$(grep -m1 "^$n" logs/a1c-ktest.txt || true)"
    step "ktest $n" "$([[ "$l" == *PASS* ]] && echo PASS || echo "FAIL${l:+（$l）}")"
  done
  echo "================ 2/4 build engine ================"
  bash build.sh engine 2>&1 | tail -n 20
  rc=${PIPESTATUS[0]}
  if [[ $rc != 0 ]]; then
    step "build engine" FAIL
    echo; echo "==== A1c 汇总 ===="
    for k in "${order[@]}"; do printf '%-34s %s\n' "$k" "${R[$k]}"; done
    echo "A1C VERIFY: FAIL（引擎编译失败，把第一条错误贴回来）"; exit 1
  fi
  step "build engine" PASS
fi

echo
echo "================ 3/4 生产配置分页逐位一致矩阵 ================"
cfgs=(prod prodp1 prodp2)
[[ -n "$OLD_BINARY" ]] && cfgs+=(prodold)
bash tools/a1_verify.sh "${cfgs[@]}" 2>&1 | tee logs/a1c-matrix.txt
rc=${PIPESTATUS[0]}
step "matrix ${cfgs[*]}" "$([[ $rc == 0 ]] && echo PASS || echo FAIL)"

echo
echo "================ 4/4 kvsnap 跨布局保存/恢复 ================"
echo "---- 4a: 不分页保存 → GDEC_KV_PAGED=2 恢复 ----"
RESTORE_KVP=2 bash tools/btv_kvsnap_verify.sh 2>&1 | tee logs/a1c-kvsnap-a.txt
rc=${PIPESTATUS[0]}
step "kvsnap save=off restore=p2" "$([[ $rc == 0 ]] && echo PASS || echo FAIL)"
echo "---- 4b: GDEC_KV_PAGED=2 保存 → 不分页恢复 ----"
SAVE_KVP=2 bash tools/btv_kvsnap_verify.sh 2>&1 | tee logs/a1c-kvsnap-b.txt
rc=${PIPESTATUS[0]}
step "kvsnap save=p2 restore=off" "$([[ $rc == 0 ]] && echo PASS || echo FAIL)"

echo
echo "==== A1c 汇总 ===="
bad=0
for k in "${order[@]}"; do
  printf '%-34s %s\n' "$k" "${R[$k]}"
  [[ "${R[$k]}" == PASS ]] || bad=1
done
echo "--- 矩阵表 ---"
sed -n '/==== 汇总 ====/,$p' logs/a1c-matrix.txt
grep -h '\[kvpage\]\|页表日志' logs/a1c-matrix.txt logs/a1c-kvsnap-*.txt 2>/dev/null | sort -u
if (( bad )); then
  echo "A1C VERIFY: FAIL（见上表；失败步骤的日志在 logs/a1c-*.txt 与 logs/a1-*.log / logs/btv-*.log）"
  exit 1
fi
echo "A1C VERIFY: PASS"
