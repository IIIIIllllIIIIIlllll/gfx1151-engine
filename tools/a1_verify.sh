#!/usr/bin/env bash
# 一键 A1 逐位一致矩阵（前台运行，全程有输出，结束时给出 PASS/FAIL）。
# 依次以 off / GDEC_KV_PAGED=1 / GDEC_KV_PAGED=2 启动引擎 → 跑 snapshot_regress
# → 停引擎，最后逐用例比较各配置的生成文本（必须一字不差）。
#   bash tools/a1_verify.sh                  # 三种配置，maxctx 40960，约 5-10 分钟
#   bash tools/a1_verify.sh off p2           # 只跑指定配置
#   A1_CTX=131072 bash tools/a1_verify.sh p2 # 大 maxctx（512 页）
#   A1_NGRAM=1 bash tools/a1_verify.sh       # 额外跑 ngram_regress
#   OLD_BINARY=/tmp/gdec-base/build/gdec bash tools/a1_verify.sh prod prodold
#       # 生产 kernel 配置下，新二进制 vs 改动前二进制（A1b 回归，必须逐字一致）
#   bash tools/a1_verify.sh prod prodp1 prodp2
#       # A1c：生产配置（BF16+WMMA+BTV）下关闭分页 / 恒等页表 / 反转页表，必须逐字一致
# 注意：off/p1/p2 走 FP32 KV，prod* 走 BF16，两组数值本来就不同，不要放在同一次里比较。
# 前提：生产服务已停（start.sh Ctrl+C）；已 bash build.sh engine。
# 结果：logs/a1-<cfg>.log（引擎日志）、logs/a1-<cfg>.txt / .json（回归输出）。
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source tools/probe_lib.sh

missing=0
for n in 2048 2051 2052 8192 32768; do
  [[ -f data/qsa-oracle/$n.json ]] || { echo "缺少 data/qsa-oracle/$n.json" >&2; missing=1; }
done
if (( missing )); then
  echo "oracle 数据缺失。它在 git 里是跟踪的，先试：git checkout HEAD -- data/qsa-oracle" >&2
  echo "若 HEAD 里也没有，从 Mac 拷：scp -r <mac>:.../gfx1151-engine/data/qsa-oracle data/" >&2
  exit 1
fi
probe_precheck || exit 1

if (( $# )); then cfgs=("$@"); else cfgs=(off p1 p2); fi
for c in "${cfgs[@]}"; do
  case "$c" in off|p1|p2|prod|prodp1|prodp2) ;; prodold)
    [[ -x "${OLD_BINARY:-}" ]] || { echo "prodold 需要 OLD_BINARY=<改动前编译的 gdec>" >&2; exit 2; } ;;
    *) echo "未知配置 $c（可选 off p1 p2 prod prodp1 prodp2 prodold）" >&2; exit 2 ;;
  esac
done
CTX="${A1_CTX:-40960}"
MODEL_DIR="${MODEL_DIR:-models}"
args=("$MODEL_DIR/qwen38-flash-next-w4b.hgn" "$MODEL_DIR/qwen38-flash-next-w4b.overlay.hgn"
      --serve --port 8732 --maxctx "$CTX")
# 与 run_ngram_probe.sh 相同的环境（关 kvsnap）
export GDEC_KVSNAP=0
export GDEC_NGRAM_MIN="${GDEC_NGRAM_MIN:-4}" GDEC_NGRAM_MAX="${GDEC_NGRAM_MAX:-16}"
PROD_VARS=(GDEC_QSA_KV_BF16 GDEC_QSA_WMMA GDEC_QSA_WMMA_BTV GDEC_GEMM_WMMA GDEC_GDN_FUSED
           GDEC_MOE_LT GDEC_MOE_LT_BF16 GDEC_GR_BF16 GDEC_GDN_STREAM GDEC_GDN_WAVE
           GDEC_NOWARMUP GDEC_INDEX_FUSED2 GDEC_PP_MOE_OUT GDEC_INDEX_STREAM_SELECT)
# prod / prodold：start.sh:103-115 的生产 kernel 配置（BF16 KV + WMMA + BTV，16K chunk）
set_env() {
  unset GDEC_KV_PAGED GDEC_PREFILL_CHUNK "${PROD_VARS[@]}"
  [[ -z "${A1_CHUNK:-}" ]] || export GDEC_PREFILL_CHUNK="$A1_CHUNK"
  case "$1" in
    p1) export GDEC_KV_PAGED=1 ;;
    p2) export GDEC_KV_PAGED=2 ;;
    prod|prodold|prodp1|prodp2)
      local v; for v in "${PROD_VARS[@]}"; do export "$v=1"; done
      export GDEC_PREFILL_CHUNK="${A1_CHUNK:-16384}"
      [[ "$1" == prodp1 ]] && export GDEC_KV_PAGED=1
      [[ "$1" == prodp2 ]] && export GDEC_KV_PAGED=2
      ;;
  esac
}

trap probe_stop EXIT
trap 'exit 130' INT TERM

ran=()
for c in "${cfgs[@]}"; do
  set_env "$c"
  rm -f "logs/a1-$c.json" "logs/a1-$c.txt"   # 不让上一次的结果混进汇总
  bin="${PROBE_BINARY:-build/gdec}"; [[ "$c" == prodold ]] && bin="$OLD_BINARY"
  echo
  echo "================ 配置 $c（GDEC_KV_PAGED=${GDEC_KV_PAGED:-未设置}，BF16=${GDEC_QSA_KV_BF16:-0}，maxctx $CTX，$bin）================"
  probe_start "a1-$c" "$bin" "${args[@]}" || { echo "[$c] 启动失败，跳过" >&2; continue; }
  kvp="$(grep -m1 '\[kvpage\]' "$PROBE_LOG" || true)"
  echo "[$c] 页表日志: ${kvp:-（无 [kvpage] 行）}"
  rm -f logs/ngram-snapshots.json
  python3 tools/snapshot_regress.py 2>&1 | tee "logs/a1-$c.txt"
  if [[ -f logs/ngram-snapshots.json ]]; then
    cp logs/ngram-snapshots.json "logs/a1-$c.json"; ran+=("$c")
  else
    echo "[$c] snapshot_regress 没有产出结果，引擎日志末尾：" >&2; tail -n 20 "$PROBE_LOG" >&2
  fi
  if [[ "${A1_NGRAM:-0}" == 1 ]]; then
    python3 tools/ngram_regress.py --keep-going --output "logs/a1-$c-ngram.json" 2>&1 | tail -n 3
  fi
  grep -E 'hipError|Segmentation|Aborted|Assertion|watchdog fired' "$PROBE_LOG" | head -n 5 || true
  probe_stop
done
trap - EXIT

echo
echo "================ 汇总 ================"
python3 - "$CTX" "${cfgs[@]}" <<'EOF'
import json, sys
from pathlib import Path
ctx, cfgs = sys.argv[1], sys.argv[2:]
res = {}
for c in cfgs:
    p = Path(f'logs/a1-{c}.json')
    res[c] = {r['name']: r for r in json.loads(p.read_text())} if p.exists() else None
ok = [c for c in cfgs if res[c] is not None]
bad = [c for c in cfgs if res[c] is None]
for c in bad:
    print(f'{c}: 没有结果（启动或回归失败，见 logs/a1-{c}.log）')
if not ok:
    print('A1 VERIFY: FAIL（没有任何结果）'); sys.exit(1)
ref = ok[0]
names = list(res[ref])
print(f"{'用例':<14}" + ''.join(f'{c + " vs gold":>14}' for c in ok) + f"   与 {ref} 一致")
diverge = 0
for n in names:
    row = f'{n:<14}'
    for c in ok:
        r = res[c].get(n)
        row += f"{'—' if r is None else ('MATCH' if r['match'] else 'DIFF'):>14}"
    same = all(res[c].get(n) and res[c][n]['actual'] == res[ref][n]['actual'] for c in ok)
    diverge += not same
    bad_cfg = [c for c in ok if not (res[c].get(n) and res[c][n]['actual'] == res[ref][n]['actual'])]
    print(row + ('   yes' if same else f'   NO: {",".join(bad_cfg)}'))
print()
print('说明：vs gold 一列允许有既有 DIFF（改动前就是 8/9，cmp4-2 DIFF）；')
print(f'     关键是最后一列：各配置之间的生成文本必须一字不差。')
if bad or diverge:
    print(f'A1 VERIFY: FAIL（{diverge} 个用例不一致，{len(bad)} 个配置无结果）'); sys.exit(1)
print(f'A1 VERIFY: PASS（{", ".join(ok)} 全部 {len(names)} 个用例逐字一致，maxctx {ctx}）')
EOF
