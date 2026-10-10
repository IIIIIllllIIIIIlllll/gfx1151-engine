#!/usr/bin/env bash
# gamma_real_check.sh — 真实文本上自适应 γ vs 固定 γ（多样本汇总），前台约 8 分钟，最后一行 GAMMA REAL CHECK: PASS/FAIL。
#   bash tools/gamma_real_check.sh
#   BIN=build/qwenox-engine CTX=8192 SPEC=512 OFFS="0 16384 ..." GAMMAS="4 5 0" MIN_RATIO=0.98 bash tools/gamma_real_check.sh
# 起因：d2_gate_check（09-29）单样本里 8K 真实文本自适应 40.5 vs γ4 45.4 tok/s（-11%）。换 γ 会改变 greedy 轨迹，
# 单样本噪声 ±10–17%，所以这里取 data/qsa-oracle/131072.tokens 的 6 个不同起点、各 CTX 个 token 当上下文，
# 每个 γ 各生成 SPEC 个 token，按 总 token / 总秒 汇总（与 138de39 gamma_adapt_verify 同口径）。
# 判定：自适应汇总 tok/s ≥ MIN_RATIO × γ=4 汇总（γ=4 = 旧 greedy 默认）。
# 日志：logs/grc/*.log，汇总 logs/gamma_real_check.out
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
BIN="${BIN:-build/qwenox-engine}"
SRC="${SRC:-data/qsa-oracle/131072.tokens}"
CTX="${CTX:-8192}"
NS="${SPEC:-512}"
OFFS="${OFFS:-0 16384 32768 49152 65536 98304}"
GAMMAS="${GAMMAS:-4 5 0}"
MINR="${MIN_RATIO:-0.98}"
OUT=logs/gamma_real_check.out
mkdir -p logs/grc
exec > >(tee "$OUT") 2>&1
T0="$(date '+%Y-%m-%d %H:%M:%S')"
echo "== gamma_real_check  $T0  BIN=$BIN CTX=$CTX SPEC=$NS OFFS=[$OFFS] GAMMAS=[$GAMMAS]"
[[ -x "$BIN" ]] || { echo "缺二进制: $BIN"; echo "GAMMA REAL CHECK: FAIL"; exit 1; }
[[ -f "$SRC" ]] || { echo "缺 token 文件: $SRC"; echo "GAMMA REAL CHECK: FAIL"; exit 1; }
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
    echo "!! amdgpu SVM 死锁（svm_range_cpu_invalidate_pagetables），中止，需重启"; echo "GAMMA REAL CHECK: FAIL"; exit 1
  fi
}
total=$(wc -l <"$SRC")
fail=0
for o in $OFFS; do
  (( o + CTX <= total )) || { echo "起点 $o + $CTX 超出 $SRC（$total）"; echo "GAMMA REAL CHECK: FAIL"; exit 1; }
  T=logs/grc/tok_o$o.txt
  tail -n +$((o + 1)) "$SRC" | head -n "$CTX" > "$T"
  for G in $GAMMAS; do
    L="grc/o${o}_g$G"
    printf '  .. 起点 %-6s γ=%-6s（%s）' "$o" "$([[ $G == 0 ]] && echo 自适应 || echo $G)" "$(date +%T)"
    SPEC=$NS GAMMA=$G BIN=$BIN bash tools/pp_prod.sh "$T" "$L" > "logs/$L.stdout" 2>&1
    rc=$?
    svm_check
    if (( rc )); then echo " FAIL rc=$rc"; tail -n 5 "logs/$L.stdout"; fail=1; continue; fi
    grep -aq 'dense: 8-bit' "logs/$L.log" || { echo " 日志没有 'dense: 8-bit'，不是 HQ 权重"; echo "GAMMA REAL CHECK: FAIL"; exit 1; }
    grep -aoE '^spec: [0-9]+ tokens in [0-9.]+ s = [0-9.]+ tok/s \| rounds=[0-9]+ commit/round=[0-9.]+' "logs/$L.log" | tail -1 | sed 's/^/  /'
  done
done
echo "== 测量结束 $(date +%T)"
(( fail )) && { echo "GAMMA REAL CHECK: FAIL（有运行失败，见上）"; exit 1; }

python3 - "$MINR" "$OFFS" "$GAMMAS" <<'PY'
import re, sys
from pathlib import Path
minr = float(sys.argv[1]); offs = sys.argv[2].split(); gammas = [int(g) for g in sys.argv[3].split()]
spec_re = re.compile(r'^spec: (\d+) tokens in ([\d.]+) s = ([\d.]+) tok/s \| rounds=(\d+) commit/round=([\d.]+)')
ad_re = re.compile(r'gamma-adapt: rounds per γ ([\d: ]+)\|')
name = lambda g: '自适应' if g == 0 else f'γ={g}'
R = {}
for o in offs:
    for g in gammas:
        txt = Path(f'logs/grc/o{o}_g{g}.log').read_text(errors='replace').splitlines()
        m = next((spec_re.match(l) for l in reversed(txt) if spec_re.match(l)), None)
        if not m:
            print(f'缺 spec 行: o{o} g{g}'); print('GAMMA REAL CHECK: FAIL'); sys.exit(1)
        d = dict(tok=int(m[1]), sec=float(m[2]), tps=float(m[3]), rounds=int(m[4]), cpr=float(m[5]))
        if g == 0:
            a = next((ad_re.search(l) for l in txt if ad_re.search(l)), None)
            d['hist'] = {int(k): int(v) for k, v in (x.split(':') for x in a[1].split())} if a else {}
        R[(o, g)] = d
print('\n  起点     ' + ''.join(f'{name(g):>18}' for g in gammas) + '   自适应用到的 γ（轮数）')
for o in offs:
    row = ''.join(f'{R[(o, g)]["tps"]:>8.1f} tok/s {R[(o, g)]["cpr"]:>4.2f}' for g in gammas)
    h = R.get((o, 0), {}).get('hist', {})
    print(f'  {o:>7}  {row}   ' + ' '.join(f'{k}:{v}' for k, v in sorted(h.items())))
print('  （每格：tok/s  每轮提交）')
P = {}
for g in gammas:
    tok = sum(R[(o, g)]['tok'] for o in offs); sec = sum(R[(o, g)]['sec'] for o in offs)
    rounds = sum(R[(o, g)]['rounds'] for o in offs)
    P[g] = tok / sec
    extra = ''
    if g == 0:
        hs = {}
        for o in offs:
            for k, v in R[(o, 0)].get('hist', {}).items():
                hs[k] = hs.get(k, 0) + v
        n = sum(hs.values())
        if n:
            extra = f'  平均 γ {sum(k * v for k, v in hs.items()) / n:.2f}'
    print(f'  汇总 {name(g):>6}: {P[g]:.1f} tok/s  每轮 {sec * 1000 / rounds:.1f} ms  每轮提交 {tok / rounds:.2f}{extra}')
if 0 not in P or 4 not in P:
    print('GAMMAS 里要有 0 和 4'); print('GAMMA REAL CHECK: FAIL'); sys.exit(1)
ratio = P[0] / P[4]
best = max((g for g in gammas if g), key=lambda g: P[g])
print(f'  自适应 / γ=4 = {ratio:.3f}（要求 ≥ {minr}）；固定 γ 里最好的是 γ={best} {P[best]:.1f} tok/s，自适应为其 {P[0] / P[best]:.3f}')
ok = ratio >= minr
print('GAMMA REAL CHECK: ' + ('PASS' if ok else 'FAIL'))
sys.exit(0 if ok else 1)
PY
