#!/usr/bin/env bash
# 一键验证集成转换器 tools/flashnext2hgn.py（原始 BF16 safetensors → 高质量 hgn，
# 有/无 imatrix 都能转）。跑完打印 PASS / FAIL。完整转换约 1 小时、占 ~130 GiB。
#   bash tools/convert_verify.sh
#     SRC=~/Models/Qwen3.8-Flash-Next     原始权重目录
#     IMATRIX=<file>|none                  默认 ~/Models/BF16/Qwen3.8-Flash-Next-BF16/imatrix_unsloth.gguf_file
#     OUT=~/Models/hq/conv                 输出目录        NAME=qwen38-flash-next-hq
#     JOBS=28                              专家量化进程数
#     KMAX=0.060（有 imatrix）/ 0.066（无） KLD 上限
#     BIN=build/gdec                       引擎二进制
#     REF_OVERLAY=~/Models/hq/qwen38-flash-next-w4b.overlay-q8.hgn  tools/hgn_hq.py 的输出（缺则跳过对照）
#     REF_CONV=<旧 flashnext2hgn.py>       --classic 对照；默认取 git e2f4d9f 版本，不是 git 仓库则跳过
#     SKIP_CONVERT=1                       OUT 里已有完整转换，跳过第 3 步的转换本身
# 检查项：
#   1. 合成小模型自测（tools/convert_selftest.py）：--classic 与旧转换器逐字节相同、HQ 只改专家、
#      legacy/GGUF imatrix 一致、overlay 正确、结果与进程数无关
#   2. 真实权重 --only-overlay：与 hgn_hq.py 的 overlay 除文件头名字外逐字节相同
#   3. 完整转换：基座张量表（名字/dtype/形状）与生产 w4b.hgn 相同；overlay = 第 2 步；
#      专家抽查误差下降（转换器内置，失败即非 0 退出）；start.sh 依次加载 base、overlay、mtp
#   4. KLD（BF16 基准 64 chunk × 512）<= KMAX
#   5. MTP 投机 smoke（8K prompt，256 token，gamma 3）：commit/round >= 3.0
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
SRC=${SRC:-$HOME/Models/Qwen3.8-Flash-Next}
IMATRIX=${IMATRIX:-$HOME/Models/BF16/Qwen3.8-Flash-Next-BF16/imatrix_unsloth.gguf_file}
OUT=${OUT:-$HOME/Models/hq/conv}
NAME=${NAME:-qwen38-flash-next-hq}
JOBS=${JOBS:-28}
[[ $IMATRIX == none ]] && KMAX=${KMAX:-0.066} || KMAX=${KMAX:-0.060}
BIN=${BIN:-build/gdec}
REF_OVERLAY=${REF_OVERLAY:-$HOME/Models/hq/qwen38-flash-next-w4b.overlay-q8.hgn}
KREF=data/kld/bf16_c512.kld
export PYTHONPATH=${PYTHONPATH:-$HOME/Workspace/pylib}
export PROBE_CAP_GB=${PROBE_CAP_GB:-100}
PY="nice -n 10 python3"
fail=0
note() { echo; echo "== $(date +%H:%M:%S) $*"; }
bad() { echo "FAIL: $*" >&2; fail=1; }
mkdir -p logs

case "$(realpath -m "$OUT")/" in */models/*) echo "OUT 不能在 models/ 下：$OUT"; echo FAIL; exit 1 ;; esac
for f in "$SRC/config.json" "$BIN" "$KREF"; do [[ -e $f ]] || { echo "缺少 $f"; echo FAIL; exit 1; }; done
IMARG=()
if [[ $IMATRIX != none ]]; then
  [[ -f $IMATRIX ]] || { echo "缺少 imatrix $IMATRIX（没有就 IMATRIX=none）"; echo FAIL; exit 1; }
  IMARG=(--imatrix "$IMATRIX")
fi
if pgrep -af '(^|/)(gdec[^/[:space:]]*|flash_serve|serve_api\.py|llama-server|llama-perplexity|llama-cli)([[:space:]]|$)' >/dev/null; then
  echo "GPU 上已有引擎/llama.cpp 在跑，请先停掉"; echo FAIL; exit 1
fi
svm0=$(journalctl -k -b 2>/dev/null | grep -c svm_range_cpu_invalidate_pagetables)
svm_check() {
  local n
  n=$(journalctl -k -b 2>/dev/null | grep -c svm_range_cpu_invalidate_pagetables)
  if (( n > svm0 )); then echo "内核日志出现 svm_range_cpu_invalidate_pagetables（$svm0 -> $n），停止" >&2; echo FAIL; exit 1; fi
  free -g | awk 'NR==2{print "   [mem] used " $3 " GiB, cache " $6 " GiB, available " $7 " GiB"}'
}
evict() {  # 把转换读写过的大文件移出 page cache，再上 GPU
  python3 - "$@" <<'PY'
import os, sys
n = 0
for root in sys.argv[1:]:
    for dp, _d, fs in os.walk(root):
        for f in fs:
            p = os.path.join(dp, f)
            try:
                fd = os.open(p, os.O_RDONLY)
            except OSError:
                continue
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            os.close(fd)
            n += 1
print(f"   [cache] fadvise DONTNEED on {n} files")
PY
}
chk="$(bash start_hgn.sh --check 2>&1)" || { echo "$chk"; echo FAIL; exit 1; }
eval "C=($(sed -n 's/^CMD //p' <<<"$chk"))"
PROD_BASE=${C[1]}
PROD_MTP=${C[3]:-}

# ---- 1. 合成自测 ---------------------------------------------------------------
note "1. 合成小模型自测"
REFARG=()
if [[ -n ${REF_CONV:-} ]]; then REFARG=(--ref "$REF_CONV")
elif git rev-parse --git-dir >/dev/null 2>&1 && git cat-file -e e2f4d9f:tools/flashnext2hgn.py 2>/dev/null; then
  git show e2f4d9f:tools/flashnext2hgn.py >logs/flashnext2hgn_ref.py; REFARG=(--ref logs/flashnext2hgn_ref.py)
fi
$PY tools/convert_selftest.py "${REFARG[@]}" 2>&1 | tee logs/convert_selftest.log | grep -E '^(PASS|FAIL|SKIP)'
grep -q '^CONVERT_SELFTEST PASS' logs/convert_selftest.log || bad "合成自测失败（logs/convert_selftest.log）"

# ---- 2. --only-overlay ----------------------------------------------------------
note "2. 真实权重 --only-overlay"
OV=$OUT/ovl
mkdir -p "$OV"
t0=$SECONDS
$PY tools/flashnext2hgn.py "$SRC" --out "$OV" --name "$NAME" --only-overlay >logs/convert_ovl.log 2>&1 \
  || bad "--only-overlay 失败（logs/convert_ovl.log）"
grep '^wrote' logs/convert_ovl.log; echo "   $((SECONDS - t0)) s"
OVF=$OV/$NAME.overlay.hgn
if [[ -f $REF_OVERLAY && -f $OVF ]]; then
  if [[ $(stat -c %s "$OVF") == $(stat -c %s "$REF_OVERLAY") ]] &&
     cmp -s <(head -c 40 "$OVF"; tail -c +105 "$OVF") <(head -c 40 "$REF_OVERLAY"; tail -c +105 "$REF_OVERLAY"); then
    echo "与 $(basename "$REF_OVERLAY") 逐字节相同（文件头名字除外）  OK"
  else bad "overlay 与 $REF_OVERLAY 不同"; fi
else echo "   跳过对照（没有 $REF_OVERLAY）"; fi

# ---- 3. 完整转换 ---------------------------------------------------------------
note "3. 完整转换 → $OUT（${IMARG[*]:-无 imatrix}，jobs $JOBS）"
if [[ -z ${SKIP_CONVERT:-} ]]; then
  need=$((135 * 1024 * 1024))
  avail=$(df -k --output=avail "$OUT" | tail -1)
  (( avail > need )) || { echo "磁盘剩余 $((avail / 1048576)) GiB，需要 ~135 GiB"; echo FAIL; exit 1; }
  t0=$SECONDS
  OMP_NUM_THREADS=1 $PY tools/flashnext2hgn.py "$SRC" --out "$OUT" --name "$NAME" "${IMARG[@]}" \
    --jobs "$JOBS" >logs/convert_full.log 2>&1
  rc=$?
  grep -E '^(HQ|classic) mode|check |^wrote|FAIL' logs/convert_full.log
  echo "   rc=$rc $(( (SECONDS - t0) / 60 )) min"
  (( rc == 0 )) || bad "转换失败（logs/convert_full.log）"
fi
B=$OUT/$NAME.hgn O=$OUT/$NAME.overlay.hgn M=$OUT/$NAME-mtp.hgn
for f in "$B" "$O" "$M" "$OUT/start.sh" "$OUT/tokenizer/tokenizer.json"; do [[ -f $f ]] || bad "缺少输出 $f"; done
if (( ! fail )); then
  $PY - "$B" "$PROD_BASE" "$M" "$PROD_MTP" <<'PY' || bad "基座张量表与生产不一致"
import sys; sys.path.insert(0, "tools")
from hgn_hq import read_hgn
a, b = read_hgn(sys.argv[1]), read_hgn(sys.argv[2])
ta = {n: v[:2] for n, v in a.items()}; tb = {n: v[:2] for n, v in b.items()}
d = sorted(n for n in set(ta) | set(tb) if ta.get(n) != tb.get(n))
print(f"   base {len(a)} tensors vs production {len(b)}: {len(d)} differ in name/dtype/shape {d[:5]}")
if len(sys.argv) > 4 and sys.argv[4]:
    m, p = read_hgn(sys.argv[3]), read_hgn(sys.argv[4])
    same = {n: v[:2] for n, v in m.items()} == {n: v[:2] for n, v in p.items()}
    print(f"   mtp sidecar {len(m)} tensors, table {'==' if same else '!='} production")
sys.exit(1 if d else 0)
PY
  cmp -s "$O" "$OVF" && echo "   overlay == 第 2 步  OK" || bad "完整转换的 overlay 与 --only-overlay 不同"
  bash -n "$OUT/start.sh" || bad "start.sh 语法错误"
  if grep -n 'engine_cmd' "$OUT/start.sh" | grep -q "$NAME.overlay.hgn" &&
     awk -v a="$NAME.hgn\"" -v b="$NAME.overlay.hgn" -v c="$NAME-mtp.hgn" \
       '/engine_cmd/{ if(index($0,a)) i=NR; if(index($0,b)) j=NR; if(index($0,c)) k=NR } END{exit !(i && i<j && j<k)}' "$OUT/start.sh"; then
    echo "   start.sh 加载顺序 base → overlay → mtp  OK"
  else bad "start.sh 没有按 base、overlay、mtp 加载"; fi
fi
evict "$SRC" "$OUT" "${IMATRIX%/*}"; svm_check

# ---- 4. KLD --------------------------------------------------------------------
note "4. KLD vs BF16（64 chunk × 512）"
if (( ! fail )); then
  out=$(env BIN=$BIN CHUNKS=64 MODEL="$B" OVERLAY="$O" bash tools/kld_engine.sh "$KREF" conv_kld 2>&1)
  grep -qE 'hipError|Segmentation|Aborted|FATAL' logs/kld_conv_kld.log && bad "KLD 崩溃（logs/kld_conv_kld.log）"
  k=$(sed -n 's/^Mean    KLD: *\([0-9.]*\).*/\1/p' <<<"$out" | head -1)
  t=$(sed -n 's/^Same top p: *\([0-9.]*\).*/\1/p' <<<"$out" | head -1)
  p=$(sed -n 's/^Mean PPL(Q) *: *\([0-9.]*\).*/\1/p' <<<"$out" | head -1)
  echo "KLD=${k:-?} top1=${t:-?}% PPL=${p:-?}   [旧 hgn 0.163；GGUF Q4_K_XL 0.049–0.051]"
  if [[ -z ${k:-} ]]; then bad "KLD 没有输出"
  elif awk -v a="$k" -v m="$KMAX" 'BEGIN{exit !(a <= m)}'; then echo "<= $KMAX  OK"
  else bad "KLD $k > $KMAX"; fi
  svm_check
fi

# ---- 5. MTP 投机 ---------------------------------------------------------------
note "5. MTP 投机 smoke：8K prompt，256 token，gamma 3"
if (( ! fail )); then
  env BIN=$BIN MODEL="$B" OVERLAY="$O" MTP="$M" SPEC=256 GAMMA=3 bash tools/pp_prod.sh 8k conv_spec >/dev/null 2>&1
  grep -qE 'hipError|Segmentation|Aborted|FATAL' logs/conv_spec.log && bad "spec 崩溃（logs/conv_spec.log）"
  # base + overlay + mtp sidecar = "base+2 overlay"; the sidecar's fc is q8g64 (dtype 7)
  grep -q 'base+2 overlay' logs/conv_spec.log && grep -q 'fc_hidden dtype=7' logs/conv_spec.log \
    && echo "   MTP sidecar 已加载（base+2 overlay，fc_hidden dtype=7）" \
    || bad "MTP sidecar 没有加载（logs/conv_spec.log 里没有 base+2 overlay / fc_hidden dtype=7）"
  r=$(tr '\r' '\n' <logs/conv_spec.log | grep -ao 'spec: [0-9]* tokens in [0-9.]* s = [0-9.]* tok/s | rounds=[0-9]* commit/round=[0-9.]*' | tail -1)
  echo "${r:-没有 spec 输出}"
  c=$(sed 's/.*commit\/round=\([0-9.]*\).*/\1/' <<<"$r")
  if [[ -z ${r:-} ]]; then bad "spec 没有输出"
  elif awk -v a="$c" 'BEGIN{exit !(a >= 3.0)}'; then echo "commit/round >= 3.0  OK"
  else bad "commit/round $c < 3.0"; fi
  svm_check
fi

echo
if (( fail )); then echo FAIL; exit 1; else echo PASS; fi
