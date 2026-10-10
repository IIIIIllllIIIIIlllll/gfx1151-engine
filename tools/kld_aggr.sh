#!/usr/bin/env bash
# kld_aggr.sh — prefill-aggr 分支的 KLD 验收（v2 权重，一键，末行 PASS/FAIL）
#   bash tools/kld_aggr.sh            约 15 分钟
# 基准：BF16 logits（~/Workspace/gfx-1151-kvsnap/data/kld/ 下 bf16_c8192 / bf16_c512）。
# 对比：BASE_BIN（分支起点 de0d650，默认 build/qwenox.base） vs build/qwenox-engine（HEAD），同机背靠背。
#   c8192：两者 + HEAD 的两个消融（QWENOX_QSA_UNION=0 / QWENOX_INDEX_F32=1，只报告不判定）
#   c512 ：两者
# 门槛：mean_kld(HEAD) <= mean_kld(BASE) + 0.002 且 same_top 下降 <= 0.5 个百分点。
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
KD=${KD:-$HOME/Workspace/gfx-1151-kvsnap/data/kld}
BASE_BIN=${BASE_BIN:-build/qwenox.base}
for f in "$KD/bf16_c8192.kld" "$KD/bf16_c512.kld" "$BASE_BIN" build/qwenox-engine; do
  [[ -e "$f" ]] || { echo "缺少 $f"; echo "KLD AGGR: FAIL"; exit 1; }
done
bash tools/aggr_serve.sh stop >/dev/null 2>&1
if pgrep -x qwenox >/dev/null; then echo "已有 qwenox 在跑"; echo "KLD AGGR: FAIL"; exit 1; fi
export MODEL_FILE=./models/qwen38-flash-next-v2.hgn NGRAM_FILE=./models/qwen38-flash-next-ngram.hgn
export OVERLAY_FILE= VISION_FILE= PARALLEL=1

run() {  # tag ref maxctx bin [env...]
  local tag=$1 ref=$2 mc=$3 bin=$4; shift 4
  BIN=$bin MAXCTX=$mc bash tools/kld_engine.sh "$ref" "aggr_$tag" "$@" > "/tmp/kld_aggr_$tag.out" 2>&1
  local s; s=$(grep -E '^kld_summary' "logs/kld_aggr_$tag.log" | tail -1)
  [[ -n "$s" ]] || { echo "$tag: 无 kld_summary（见 logs/kld_aggr_$tag.log）"; tail -5 "logs/kld_aggr_$tag.log"; }
  echo "$tag $s" >> /tmp/kld_aggr_results.txt
  echo "$tag $s" | tr '\t' ' '
}
rm -f /tmp/kld_aggr_results.txt
run c8192_base "$KD/bf16_c8192.kld" 8448 "$BASE_BIN"
run c8192_head "$KD/bf16_c8192.kld" 8448 build/qwenox-engine
run c8192_head_nounion "$KD/bf16_c8192.kld" 8448 build/qwenox-engine QWENOX_QSA_UNION=0
run c8192_head_idxf32 "$KD/bf16_c8192.kld" 8448 build/qwenox-engine QWENOX_INDEX_F32=1
run c512_base "$KD/bf16_c512.kld" 4096 "$BASE_BIN"
run c512_head "$KD/bf16_c512.kld" 4096 build/qwenox-engine

python3 - <<'PY'
import re
R = {}
for line in open("/tmp/kld_aggr_results.txt"):
    tag = line.split()[0]
    d = dict(re.findall(r"(\w+)=([\d.]+)", line))
    if "mean_kld" in d: R[tag] = {k: float(v) for k, v in d.items()}
print()
print(f"{'run':22s} {'mean_kld':>10s} {'p99.9':>8s} {'same_top':>9s} {'ppl_q':>8s}")
for t, d in R.items():
    print(f"{t:22s} {d['mean_kld']:10.6f} {d['p999_kld']:8.4f} {d['same_top']:9.3f} {d['ppl_q']:8.4f}")
ok = True
for ctx in ("c8192", "c512"):
    b, h = R.get(ctx + "_base"), R.get(ctx + "_head")
    if not b or not h:
        print(f"{ctx}: 缺结果 -> FAIL"); ok = False; continue
    dk, dt = h["mean_kld"] - b["mean_kld"], h["same_top"] - b["same_top"]
    good = dk <= 0.002 and dt >= -0.5
    ok &= good
    print(f"{ctx}: HEAD-BASE mean_kld {dk:+.6f} (门 +0.002), same_top {dt:+.3f} pp (门 -0.5) -> {'PASS' if good else 'FAIL'}")
print("KLD AGGR:", "PASS" if ok else "FAIL")
PY
