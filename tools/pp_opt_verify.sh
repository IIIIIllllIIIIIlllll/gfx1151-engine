#!/usr/bin/env bash
# 2026-09-24 prefill 优化一键验证（前台运行，末行 PP OPT VERIFY: PASS/FAIL）。
#   bash tools/pp_opt_verify.sh [32k|8k|...]      默认 32k
# 同一二进制、同一生产 env 下对比：
#   new = 默认（k64 GEMM 分流 + iproj rocBLAS solution + MTP 索引 tiled）
#   old = GDEC_GEMM_NO_K64=1 GDEC_IPROJ_SGEMM=1 GDEC_MTP_INDEX_SGEMM=1（旧路径）
# 判据：GEN=48 贪心 ids 一致、SPEC=256 投机生成 ids 一致；打印两边 prefill tok/s
# 与 MTP depth acc 供参考（速度不作为 PASS 条件）。需先 bash build.sh engine。
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
L=${1:-32k}
OLD=(GDEC_GEMM_NO_K64=1 GDEC_IPROJ_SGEMM=1 GDEC_MTP_INDEX_SGEMM=1)
fail=0
ids() { grep '^ids:' "$1" | tr ' ' '\n' | grep -E '^[0-9]+$' | head -n -3; }
speed() { tr '\r' '\n' <"$1" | grep -o 'prefill: .*tok/s' | sed 's/.*= //' | paste -sd' '; }
run() { local tag=$1; shift; "$@" >/dev/null 2>&1 || { echo "  $tag 运行失败（见 logs/$tag.log）"; fail=1; }; }

echo "== 1/2 贪心 GEN=48（$L）"
run ppv_g_new env GEN=48 bash tools/pp_prod.sh "$L" ppv_g_new
run ppv_g_old env GEN=48 bash tools/pp_prod.sh "$L" ppv_g_old "${OLD[@]}"
echo "  new prefill: $(speed logs/ppv_g_new.log)"
echo "  old prefill: $(speed logs/ppv_g_old.log)"
if [[ -s logs/ppv_g_new.log ]] && diff -q <(ids logs/ppv_g_new.log) <(ids logs/ppv_g_old.log) >/dev/null &&
   [[ $(ids logs/ppv_g_new.log | wc -l) -gt 0 ]]; then
  echo "  ids 一致 OK"
else
  echo "  ids 不一致 FAIL"; fail=1
fi

echo "== 2/2 MTP 投机 SPEC=256（$L）"
run ppv_s_new env SPEC=256 bash tools/pp_prod.sh "$L" ppv_s_new
run ppv_s_old env SPEC=256 bash tools/pp_prod.sh "$L" ppv_s_old "${OLD[@]}"
for t in new old; do
  echo "  $t: $(tr '\r' '\n' <logs/ppv_s_$t.log | grep -E '^spec depth acc' | cut -c1-80)"
done
if [[ -s logs/ppv_s_new.log ]] && diff -q <(ids logs/ppv_s_new.log) <(ids logs/ppv_s_old.log) >/dev/null &&
   [[ $(ids logs/ppv_s_new.log | wc -l) -gt 0 ]]; then
  echo "  spec ids 一致 OK"
else
  echo "  spec ids 不一致 FAIL"; fail=1
fi

[[ $fail == 0 ]] && echo "PP OPT VERIFY: PASS" || echo "PP OPT VERIFY: FAIL"
exit $fail
