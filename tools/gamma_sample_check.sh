#!/usr/bin/env bash
# gamma_sample_check.sh — 采样模式（API 默认 auto = 自适应 γ）vs 固定 γ=3，多样本汇总。最后一行 GAMMA SAMPLE CHECK: PASS/FAIL。
#   bash tools/gamma_sample_check.sh                      # 约 13 分钟（18 个上下文×seed × 2 个配置）
#   CFGS="3 0 4 0:QWENOX_SPEC_ADAPT_R=0.25,QWENOX_SPEC_ADAPT_MIN=2" SEEDS="1 2" MIN_RATIO=0.98 bash tools/gamma_sample_check.sh
# 09-29 两状态 GammaCtl：真实长文 0 ÷ γ3 = 1.004（γ4 0.953），x 1.040（γ4 1.032）→ PASS。
# 采样没法像 gamma_force_check 那样固定轨迹（接受是随机拒绝采样），换 γ 就换了随机轨迹，单样本 ±15–27%，
# 所以按 总 token / 总秒 汇总：真实长文 = data/qsa-oracle/131072.tokens 的 OFFS 起点 × SEEDS；
# x = 高接受率/代码文本（同 gamma_trace.sh XSET：x1 tok8192，x2 40_model.inc 一段，x3 src/api/*.cpp）× SEEDS。
# 采样参数同 API 默认附近：温度 0.7 top_k 20 top_p 0.8。CFGS 每项 "γ[:K=V[,K=V]]"，γ=0 = 自适应。
# 判定：第一个 γ=0 配置，真实长文汇总 ≥ MIN_RATIO × γ3，x 汇总 ≥ MIN_RATIO × γ3。
# 日志：logs/gsc/*.log，汇总 logs/gamma_sample_check.out
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
BIN="${BIN:-build/qwenox-engine}"
SRC="${SRC:-data/qsa-oracle/131072.tokens}"
CTX="${CTX:-8192}"
NS="${SPEC:-512}"
OFFS="${OFFS-0 20480 40960 61440 81920 98304}"
XSET="${XSET-1 2 3}"
SEEDS="${SEEDS:-1 2}"
CFGS="${CFGS:-3 0}"
MINR="${MIN_RATIO:-0.98}"
SMP="0.7,20,0.8"
mkdir -p logs/gsc
exec > >(tee logs/gamma_sample_check.out) 2>&1
T0="$(date '+%Y-%m-%d %H:%M:%S')"
echo "== gamma_sample_check  $T0  BIN=$BIN CTX=$CTX SPEC=$NS SEEDS=[$SEEDS] CFGS=[$CFGS]"
[[ -x "$BIN" ]] || { echo "缺二进制: $BIN"; echo "GAMMA SAMPLE CHECK: FAIL"; exit 1; }
[[ -f "$SRC" ]] || { echo "缺 token 文件: $SRC"; echo "GAMMA SAMPLE CHECK: FAIL"; exit 1; }
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
die() { echo "$1"; echo "GAMMA SAMPLE CHECK: FAIL"; exit 1; }
svm_check() {
  journalctl -k --since "$T0" --no-pager 2>/dev/null | grep -q svm_range_cpu_invalidate_pagetables \
    && die "!! amdgpu SVM 死锁（svm_range_cpu_invalidate_pagetables），中止，需重启"
}
total=$(wc -l <"$SRC")
ctxfile() {  # o<起点> | x<n> → token 文件
  local c=$1 T=logs/gsc/tok_$1.txt
  if [[ ! -s $T ]]; then
    case $c in
      o*) local o=${c#o}; (( o + CTX <= total )) || die "起点 $o + $CTX 超出 $SRC（$total）"
          tail -n +$((o + 1)) "$SRC" | head -n "$CTX" > "$T" ;;
      x1) for d in data/ppbench "$HOME/ppbench"; do [[ -f $d/tok8192.txt ]] && head -n "$CTX" "$d/tok8192.txt" > "$T" && break; done ;;
      x2|x3) python3 - "${c#x}" "$CTX" "$T" <<'PY'
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
  [[ -s $T ]] || die "生成 $T 失败"
  echo "$T"
}
CTXS=""; for o in $OFFS; do CTXS+=" o$o"; done; for x in $XSET; do CTXS+=" x$x"; done
for c in $CTXS; do
  T=$(ctxfile "$c")
  for s in $SEEDS; do
    for cfg in $CFGS; do
      G=${cfg%%:*}; KV=(); [[ $cfg == *:* ]] && IFS=, read -ra KV <<<"${cfg#*:}"
      tag=$(echo "$cfg" | tr ':=,.' '____'); L="gsc/${c}_s${s}_$tag"
      printf '  .. %-6s s%-2s %-28s（%s）' "$c" "$s" "$cfg" "$(date +%T)"
      SPEC=$NS GAMMA=$G BIN=$BIN bash tools/pp_prod.sh "$T" "$L" "QWENOX_SPEC_SAMPLE=$SMP,$s" "${KV[@]}" > "logs/$L.stdout" 2>&1 \
        || die " 运行失败，见 logs/$L.stdout"
      svm_check
      grep -aq 'dense: 8-bit' "logs/$L.log" || die " 日志没有 'dense: 8-bit'，不是 HQ 权重"
      grep -aoE '^spec-sample: [0-9]+ tokens in [0-9.]+ s = [0-9.]+ tok/s \| rounds=[0-9]+ commit/round=[0-9.]+' "logs/$L.log" | tail -1 \
        | grep . || die " 日志没有 spec-sample 行"
    done
  done
done
echo "== 测量结束 $(date +%T)"

python3 - "$MINR" "$CTXS" "$SEEDS" "$CFGS" <<'PY'
import re, sys
from pathlib import Path
minr = float(sys.argv[1]); ctxs = sys.argv[2].split(); seeds = sys.argv[3].split(); cfgs = sys.argv[4].split()
spec_re = re.compile(r'^spec-sample: (\d+) tokens in ([\d.]+) s = ([\d.]+) tok/s \| rounds=(\d+) commit/round=([\d.]+)')
ad_re = re.compile(r'gamma-adapt: rounds per γ ([\d: ]+)\|')
tag = lambda c: c.translate(str.maketrans(':=,.', '____'))
show = lambda c: c.replace('QWENOX_SPEC_ADAPT_', '').replace('QWENOX_SPEC_', '')
R = {}
for c in ctxs:
    for s in seeds:
        for g in cfgs:
            txt = Path(f'logs/gsc/{c}_s{s}_{tag(g)}.log').read_text(errors='replace').splitlines()
            m = next((spec_re.match(l) for l in reversed(txt) if spec_re.match(l)), None)
            a = next((ad_re.search(l) for l in txt if ad_re.search(l)), None)
            R[(c, s, g)] = dict(tok=int(m[1]), sec=float(m[2]), tps=float(m[3]), rounds=int(m[4]),
                                hist=a[1].strip() if a else '')
if '3' not in cfgs:
    print('CFGS 里要有 3（基准）'); print('GAMMA SAMPLE CHECK: FAIL'); sys.exit(1)
w = max(8, max(len(show(g)) for g in cfgs) + 2)
print('\n  上下文/seed ' + ''.join(f'{show(g):>{w}}' for g in cfgs) + '   （tok/s）')
for c in ctxs:
    for s in seeds:
        print(f'  {c:>6} s{s:<3} ' + ''.join(f'{R[(c, s, g)]["tps"]:>{w}.1f}' for g in cfgs))
adapt = next((g for g in cfgs if g.split(':')[0] == '0'), None)
ok = True
for name, sub in (('真实长文', [c for c in ctxs if c[0] == 'o']), ('高接受率/代码 x', [c for c in ctxs if c[0] == 'x'])):
    if not sub:
        continue
    keys = [(c, s) for c in sub for s in seeds]
    print(f'\n  {name}（{len(keys)} 个样本汇总）:')
    P = {}
    for g in cfgs:
        tok = sum(R[k + (g,)]['tok'] for k in keys); sec = sum(R[k + (g,)]['sec'] for k in keys)
        rounds = sum(R[k + (g,)]['rounds'] for k in keys)
        hs = {}
        for k in keys:
            for kv in R[k + (g,)]['hist'].split():
                a, b = map(int, kv.split(':')); hs[a] = hs.get(a, 0) + b
        n = sum(hs.values())
        P[g] = tok / sec
        extra = f'  平均 γ {sum(a * b for a, b in hs.items()) / n:.2f}' if n else ''
        print(f'    {show(g):<{w}} {P[g]:6.2f} tok/s  ÷γ3 {P[g] / P["3"]:.3f}  每轮 {sec * 1000 / rounds:.1f} ms  每轮提交 {tok / rounds:.2f}{extra}')
    if adapt:
        r = P[adapt] / P['3']
        print(f'    判定：{show(adapt)} ÷ γ3 = {r:.3f}（要求 ≥ {minr}）')
        ok &= r >= minr
print('GAMMA SAMPLE CHECK: ' + ('PASS' if ok else 'FAIL'))
sys.exit(0 if ok else 1)
PY
