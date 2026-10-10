#!/usr/bin/env bash
# gamma_trace.sh — 采集逐轮 MTP 接受轨迹（QWENOX_SPEC_TRACE=1），供 tools/gamma_sim.py 离线回放调 GammaCtl。
# 前台约 23 分钟，最后一行 GAMMA TRACE: PASS/FAIL（PASS = 全部跑完且日志里都有 gamma-trace 行）。
#   bash tools/gamma_trace.sh
#   BIN=build/qwenox-engine CTX=8192 SPEC=512 GOFFS="..." SOFFS="..." SEEDS="1 2" bash tools/gamma_trace.sh
# 文本：data/qsa-oracle/131072.tokens 的不同起点各取 CTX 个 token（真实长文，无重复段）。
#   greedy：γ=7 × GOFFS（12 个起点，最深，给出每个位置的接受前缀）；自适应 × 前 6 个起点（各 γ 的每轮耗时）
#   采样  ：γ=7 与 γ=3 × SOFFS × SEEDS（温度 0.7 top_k 20 top_p 0.8，同 gamma_adapt_verify）
#   额外上下文 XSET（高接受率文本）：x1 = 8k 基准（tok8192，重复多）、x2 = src/gpu/parts/40_model.inc 一段 C++、
#     x3 = src/api/*.cpp；各跑 greedy γ7/γ4、采样 γ7/γ3（seed 1）。只跑这组：GOFFS= AOFFS= SOFFS= bash tools/gamma_trace.sh
# 日志：logs/gtr/<g|s>_o<起点>_<γ>[_s<seed>].log
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
BIN="${BIN:-build/qwenox-engine}"
SRC="${SRC:-data/qsa-oracle/131072.tokens}"
CTX="${CTX:-8192}"
NS="${SPEC:-512}"
GOFFS="${GOFFS-0 10240 16384 20480 30720 32768 40960 49152 61440 65536 81920 98304}"
AOFFS="${AOFFS-0 16384 32768 49152 65536 98304}"
SOFFS="${SOFFS-0 20480 40960 61440 81920 98304}"
SEEDS="${SEEDS:-1 2}"
XSET="${XSET-1 2 3}"
SMP="0.7,20,0.8"
mkdir -p logs/gtr
exec > >(tee logs/gamma_trace.out) 2>&1
T0="$(date '+%Y-%m-%d %H:%M:%S')"
echo "== gamma_trace  $T0  BIN=$BIN CTX=$CTX SPEC=$NS"
[[ -x "$BIN" ]] || { echo "缺二进制: $BIN"; echo "GAMMA TRACE: FAIL"; exit 1; }
[[ -f "$SRC" ]] || { echo "缺 token 文件: $SRC"; echo "GAMMA TRACE: FAIL"; exit 1; }
grep -E '^(MODEL_FILE|OVERLAY_FILE)=' service.conf

# 防 amdgpu SVM 死锁：先把 GGUF 权重逐出 page cache（同 d2_gate_check.sh）
mapfile -t GF < <(bash start_gguf.sh --check 2>/dev/null | sed -n 's/^CMD //p' | tr ' ' '\n' | grep -E '\.gguf$')
(( ${#GF[@]} )) && python3 - "${GF[@]}" <<'PY'
import os, sys
for f in sys.argv[1:]:
    try:
        fd = os.open(f, os.O_RDONLY); os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED); os.close(fd)
    except OSError:
        pass
PY
svm_check() {
  if journalctl -k --since "$T0" --no-pager 2>/dev/null | grep -q svm_range_cpu_invalidate_pagetables; then
    echo "!! amdgpu SVM 死锁（svm_range_cpu_invalidate_pagetables），中止，需重启"; echo "GAMMA TRACE: FAIL"; exit 1
  fi
}
total=$(wc -l <"$SRC")
tokf() {  # 起点 → token 文件
  local T=logs/gtr/tok_o$1.txt
  (( $1 + CTX <= total )) || { echo "起点 $1 + $CTX 超出 $SRC（$total）"; echo "GAMMA TRACE: FAIL"; exit 1; }
  [[ -s $T ]] || tail -n +$(($1 + 1)) "$SRC" | head -n "$CTX" > "$T"
  echo "$T"
}
xtok() {  # x 编号 → token 文件（x1 基准文本；x2/x3 源码经 tools/qwentok.py 编码）
  local T=logs/gtr/tok_x$1.txt
  if [[ ! -s $T ]]; then
    case $1 in
      1) for d in data/ppbench "$HOME/ppbench"; do [[ -f $d/tok8192.txt ]] && head -n "$CTX" "$d/tok8192.txt" > "$T" && break; done ;;
      2|3) python3 - "$1" "$CTX" "$T" <<'PY'
import glob, sys
sys.path.insert(0, 'tools')
import qwentok
x, n, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
if x == '2':
    txt = open('src/gpu/parts/40_model.inc', errors='replace').read()[120000:]
else:
    txt = ''.join(open(f, errors='replace').read() for f in sorted(glob.glob('src/api/*.cpp')))
ids = qwentok.Tokenizer().encode(txt.replace('\r\n', '\n'))[:n]
assert len(ids) == n, (x, len(ids))
open(out, 'w').write('\n'.join(map(str, ids)) + '\n')
PY
         ;;
    esac
  fi
  [[ -s $T ]] || { echo "生成 $T 失败"; echo "GAMMA TRACE: FAIL"; exit 1; }
  echo "$T"
}
fail=0
run() {  # 标签 起点|x<n> γ [K=V ...]
  local L="gtr/$1" o=$2 G=$3; shift 3
  local T; if [[ $o == x* ]]; then T=$(xtok "${o#x}"); else T=$(tokf "$o"); fi
  printf '  .. %-18s（%s）' "$L" "$(date +%T)"
  SPEC=$NS GAMMA=$G BIN=$BIN bash tools/pp_prod.sh "$T" "$L" QWENOX_SPEC_TRACE=1 "$@" > "logs/$L.stdout" 2>&1
  local rc=$?
  svm_check
  if (( rc )); then echo " FAIL rc=$rc"; tail -n 5 "logs/$L.stdout"; fail=1; return; fi
  grep -aq 'dense: 8-bit' "logs/$L.log" || { echo " 日志没有 'dense: 8-bit'，不是 HQ 权重"; echo "GAMMA TRACE: FAIL"; exit 1; }
  grep -aq 'gamma-trace:' "logs/$L.log" || { echo " 日志没有 gamma-trace 行（$BIN 不含 QWENOX_SPEC_TRACE？）"; fail=1; return; }
  grep -aoE '^spec(-sample)?: [0-9]+ tokens in [0-9.]+ s = [0-9.]+ tok/s \| rounds=[0-9]+ commit/round=[0-9.]+' "logs/$L.log" | tail -1
}
for o in $GOFFS; do run "g_o${o}_7" "$o" 7; done
for o in $AOFFS; do run "g_o${o}_0" "$o" 0; done
for s in $SEEDS; do
  for o in $SOFFS; do
    for G in 7 3; do run "s_o${o}_${G}_s$s" "$o" "$G" "QWENOX_SPEC_SAMPLE=$SMP,$s"; done
  done
done
for x in $XSET; do
  for G in 7 4; do run "g_x${x}_$G" "x$x" "$G"; done
  for G in 7 3; do run "s_x${x}_${G}_s1" "x$x" "$G" "QWENOX_SPEC_SAMPLE=$SMP,1"; done
done
echo "== 结束 $(date +%T)"
echo "GAMMA TRACE: $( (( fail )) && echo FAIL || echo PASS)"
(( fail == 0 ))
