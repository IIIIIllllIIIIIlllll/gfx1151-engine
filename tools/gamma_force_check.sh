#!/usr/bin/env bash
# gamma_force_check.sh — 固定轨迹下比较 γ 策略（greedy MTP）。最后一行 GAMMA FORCE CHECK: PASS/FAIL。
#   bash tools/gamma_force_check.sh                       # 首次约 30 分钟（含参考序列），之后每个配置约 6 分钟
#   CFGS="4 3 5 0 0:QWENOX_SPEC_ADAPT_R=0.26" bash tools/gamma_force_check.sh
#   BIN=build/qwenox-engine CTX=8192 SPEC=512 OFFS="..." XSET="1 2 3" MIN_RATIO=0.99 bash tools/gamma_force_check.sh
# 为什么要固定轨迹：换 γ 会让 greedy 在近并列处翻转、生成内容分叉，单样本 tok/s 差 ±15–27%
# （gamma_real_check 6 起点汇总仍有 ±7% 噪声）。这里先用普通 greedy 解码（--gen）生成参考序列，
# 再让每个配置带 QWENOX_SPEC_FORCE=参考序列 跑 --spec-gen：验收对照参考 id（verify 照常跑、耗时真实，
# 草稿接受与否仍由 MTP 决定），所有配置提交完全相同的 token —— 成对比较，只剩计时噪声。
# 上下文：data/qsa-oracle/131072.tokens 的 OFFS 起点（真实长文）+ XSET（x1 = tok8192 重复多的基准文本，
#   x2 = src/gpu/parts/40_model.inc 一段 C++，x3 = src/api/*.cpp），各取 CTX 个 token。
# 配置 CFGS：空格分隔，每项 "γ[:K=V[,K=V]]"，γ=0 为自适应（GammaCtl）。
# 判定：第一个 γ=0 的配置，在真实文本汇总和 x 汇总上都 ≥ MIN_RATIO × γ=4；且每次运行的 ids 与参考一致。
# 日志：logs/gfc/*.log，汇总 logs/gamma_force_check.out；参考序列缓存 logs/gfc/ref_*.txt（删掉即重做）。
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
BIN="${BIN:-build/qwenox-engine}"
SRC="${SRC:-data/qsa-oracle/131072.tokens}"
CTX="${CTX:-8192}"
NS="${SPEC:-512}"
OFFS="${OFFS-0 10240 16384 20480 30720 32768 40960 49152 61440 65536 81920 98304}"
XSET="${XSET-1 2 3}"
CFGS="${CFGS:-4 3 5 0}"
MINR="${MIN_RATIO:-0.99}"
mkdir -p logs/gfc
exec > >(tee logs/gamma_force_check.out) 2>&1
T0="$(date '+%Y-%m-%d %H:%M:%S')"
echo "== gamma_force_check  $T0  BIN=$BIN CTX=$CTX SPEC=$NS CFGS=[$CFGS]"
[[ -x "$BIN" ]] || { echo "缺二进制: $BIN"; echo "GAMMA FORCE CHECK: FAIL"; exit 1; }
[[ -f "$SRC" ]] || { echo "缺 token 文件: $SRC"; echo "GAMMA FORCE CHECK: FAIL"; exit 1; }
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
die() { echo "$1"; echo "GAMMA FORCE CHECK: FAIL"; exit 1; }
svm_check() {
  journalctl -k --since "$T0" --no-pager 2>/dev/null | grep -q svm_range_cpu_invalidate_pagetables \
    && die "!! amdgpu SVM 死锁（svm_range_cpu_invalidate_pagetables），中止，需重启"
}
total=$(wc -l <"$SRC")
ctxfile() {  # o<起点> | x<n> → token 文件
  local c=$1 T=logs/gfc/tok_$1.txt
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
ids() { grep -a '^ids:' "logs/$1.log" | tail -1 | sed -E 's/^ids://; s/[^0-9 ].*$//'; }
CTXS=""; for o in $OFFS; do CTXS+=" o$o"; done; for x in $XSET; do CTXS+=" x$x"; done
NP=$CTX
for c in $CTXS; do
  T=$(ctxfile "$c"); REF=logs/gfc/ref_${c}_$NS.txt
  if [[ ! -s $REF ]]; then
    printf '  .. 参考 %-6s（%s）' "$c" "$(date +%T)"
    env -u SPEC -u GAMMA GEN=$((NS + 2)) BIN=$BIN bash tools/pp_prod.sh "$T" "gfc/ref_$c" > "logs/gfc/ref_$c.stdout" 2>&1 || die " 参考序列失败，见 logs/gfc/ref_$c.stdout"
    svm_check
    grep -aq 'dense: 8-bit' "logs/gfc/ref_$c.log" || die " 日志没有 'dense: 8-bit'，不是 HQ 权重"
    # 用逐行 fflush 的 "step N chosen=X"（step 0 = prefill argmax，跳过）；--gen 的 ids: 行会被
    # 结尾 stderr 的 "decode: ..." 从 4K 缓冲边界切开，不能用
    grep -aE '^step [0-9]+ chosen=' "logs/gfc/ref_$c.log" | sed -E 's/^step [0-9]+ chosen=([0-9]+).*/\1/' | tail -n +2 > "$REF"
    (( $(wc -l <"$REF") >= NS )) || die " 参考序列太短（$(wc -l <"$REF")）"
    grep -aq '^spec' "logs/gfc/ref_$c.log" && die " 参考运行走了 spec-gen（应为普通 --gen）"
    echo "$(wc -l <"$REF") ids"
  fi
  for cfg in $CFGS; do
    G=${cfg%%:*}; KV=(); [[ $cfg == *:* ]] && IFS=, read -ra KV <<<"${cfg#*:}"
    tag=$(echo "$cfg" | tr ':=,.' '____'); L="gfc/${c}_$tag"
    printf '  .. %-6s %-28s（%s）' "$c" "$cfg" "$(date +%T)"
    SPEC=$NS GAMMA=$G BIN=$BIN bash tools/pp_prod.sh "$T" "$L" "QWENOX_SPEC_FORCE=$ROOT/$REF" "${KV[@]}" > "logs/$L.stdout" 2>&1 \
      || die " 运行失败，见 logs/$L.stdout"
    svm_check
    grep -aq 'spec-gen force:' "logs/$L.log" || die " 日志没有 'spec-gen force:'（$BIN 不含 QWENOX_SPEC_FORCE？）"
    got=$(ids "$L" | tr ' ' '\n' | grep -v '^$' | tail -n +$((NP + 2)) | head -n "$NS" | md5sum)
    want=$(head -n "$NS" "$REF" | md5sum)
    [[ $got == "$want" ]] || die " ids 与参考不一致（强制轨迹失效）"
    grep -aoE '^spec: [0-9]+ tokens in [0-9.]+ s = [0-9.]+ tok/s \| rounds=[0-9]+ commit/round=[0-9.]+' "logs/$L.log" | tail -1
  done
done
echo "== 测量结束 $(date +%T)"

python3 - "$MINR" "$CTXS" "$CFGS" <<'PY'
import re, sys
from pathlib import Path
minr = float(sys.argv[1]); ctxs = sys.argv[2].split(); cfgs = sys.argv[3].split()
spec_re = re.compile(r'^spec: (\d+) tokens in ([\d.]+) s = ([\d.]+) tok/s \| rounds=(\d+) commit/round=([\d.]+)')
ad_re = re.compile(r'gamma-adapt: rounds per γ ([\d: ]+)\|')
tag = lambda c: c.translate(str.maketrans(':=,.', '____'))
show = lambda c: c.replace('QWENOX_SPEC_ADAPT_', '').replace('QWENOX_SPEC_', '')
R = {}
for c in ctxs:
    for g in cfgs:
        txt = Path(f'logs/gfc/{c}_{tag(g)}.log').read_text(errors='replace').splitlines()
        m = next((spec_re.match(l) for l in reversed(txt) if spec_re.match(l)), None)
        a = next((ad_re.search(l) for l in txt if ad_re.search(l)), None)
        R[(c, g)] = dict(tok=int(m[1]), sec=float(m[2]), tps=float(m[3]), rounds=int(m[4]),
                         hist=a[1].strip() if a else '')
w = max(8, max(len(show(g)) for g in cfgs) + 2)
print('\n  上下文 ' + ''.join(f'{show(g):>{w}}' for g in cfgs) + '   （tok/s；自适应列后附 γ 轮数分布）')
for c in ctxs:
    ad = next((R[(c, g)]['hist'] for g in cfgs if g.split(':')[0] == '0'), '')
    print(f'  {c:>6} ' + ''.join(f'{R[(c, g)]["tps"]:>{w}.1f}' for g in cfgs) + f'   {ad}')
if '4' not in cfgs:
    print('CFGS 里要有 4（基准）'); print('GAMMA FORCE CHECK: FAIL'); sys.exit(1)
adapt = next((g for g in cfgs if g.split(':')[0] == '0'), None)
ok = True
for name, sub in (('真实长文', [c for c in ctxs if c[0] == 'o']), ('高接受率 x', [c for c in ctxs if c[0] == 'x'])):
    if not sub:
        continue
    P = {g: sum(R[(c, g)]['tok'] for c in sub) / sum(R[(c, g)]['sec'] for c in sub) for g in cfgs}
    ms = {g: sum(R[(c, g)]['sec'] for c in sub) * 1000 / sum(R[(c, g)]['rounds'] for c in sub) for g in cfgs}
    print(f'\n  {name}（{len(sub)} 个上下文汇总）:')
    for g in cfgs:
        print(f'    {show(g):<{w}} {P[g]:6.2f} tok/s  ÷γ4 {P[g] / P["4"]:.3f}  每轮 {ms[g]:.1f} ms')
    if adapt:
        r = P[adapt] / P['4']
        print(f'    判定：{show(adapt)} ÷ γ4 = {r:.3f}（要求 ≥ {minr}）')
        ok &= r >= minr
print('GAMMA FORCE CHECK: ' + ('PASS' if ok else 'FAIL'))
sys.exit(0 if ok else 1)
PY
