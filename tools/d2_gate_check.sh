#!/usr/bin/env bash
# d2_gate_check.sh — 独立复核 D2（多序列合批 verify）闸门，当前 HEAD 二进制 + 生产 HQ 权重，前台约 20 分钟。
#   bash tools/d2_gate_check.sh
#   BIN=build/qwenox-engine CTXS="8192 65536" SPEC=256 GATE=1.25 bash tools/d2_gate_check.sh
#   SMP=0.7,20,0.8,1 bash tools/d2_gate_check.sh      # 改测采样（temp,top_k,top_p,seed）
# 每个上下文长度（取 data/qsa-oracle/131072.tokens 的前 N 个 token，真实文本、无重复）：
#   1) QWENOX_VBENCH：P 行 verify 成本。T(P) = 同一序列连续 P 行；T(R×B) = R 行拆成 B 段、取自 prompt 里
#      相距很远的 B 处（专家分布近似 B 个不同序列）
#   2) 单路投机 γ=1,2,3,4,7 与自适应 γ：tok/s、每轮 ms Tr(γ)、每轮提交 E(γ)
# 另跑一次 1K 上下文的 VBENCH：C(p) = T_N(p) − T_1K(p) 近似"每个序列按上下文长度付的 attention/indexer 成本"。
# 合批（B 个序列各 γb，p=γb+1，R=Bp≤16），分时基线 = 单路最好（固定 γ 与自适应取最大，分时总吞吐 = 单路）：
#   上限  一轮 = B·Tr − [B·T(p) − T(R)]                     同序列行：专家并集按同序列算、只读一份 KV → 偏乐观
#   估算  一轮 = B·Tr − [B·T(p) − T(R×B)] + (B−1)·C(p)       跨序列专家 + 每多一个序列多读一份 KV（v1：draft/判定各付各的）
#   理想  一轮 = Tr + [T(R×B) − T(p)] + (B−1)·C(p)           同上，但 draft/判定/回滚也完美共享（任何合批方案的天花板）
#   估算仍偏乐观：同一文档的远段之间，专家重叠比不同对话多；也没算合批的 host 开销。
#   C(p) 可能为负（09-29 实测 1K 的 VBENCH 反而比 8K 慢 4–8 ms，原因未查），按 0 计；同次 64K 只比 8K
#   贵 2–4 ms（QSA 每个 query 读的 block 数固定），所以每多一个序列的 KV 成本本来就小。
# 09-29 结果（HEAD 5eee344，HQ，greedy）：基线 γ=4 45.4/45.7 tok/s；B=4 γ1 估算 1.36×/1.34×，
#   B=2 γ3 1.23×/1.26× → FAIL。P>8 有悬崖（P=8 79 ms → P=10 180 ms），R 必须 ≤8。
# 判定：所有长度下"估算"最高 < GATE（1.25×）→ PASS（执行方"D2 不值得做"的结论成立）；否则 FAIL（要重新评估）。
# 日志：logs/d2g_*.log，汇总 logs/d2_gate_check.out
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
BIN="${BIN:-build/qwenox-engine}"
SRC="${SRC:-data/qsa-oracle/131072.tokens}"
CTXS="${CTXS:-8192 65536}"
NS="${SPEC:-256}"
GATE="${GATE:-1.25}"
SMP="${SMP:-}"
PS="1,2,3,4,5,6,7,8,10,12,14,16,4x2,6x2,8x2,16x2,8x4,12x4,16x4"
GAMMAS="1 2 3 4 7 0"
OUT=logs/d2_gate_check.out
mkdir -p logs logs/d2g
exec > >(tee "$OUT") 2>&1
T0="$(date '+%Y-%m-%d %H:%M:%S')"
echo "== d2_gate_check  $T0  BIN=$BIN CTXS=$CTXS SPEC=$NS ${SMP:+SMP=$SMP }GATE=$GATE"
[[ -x "$BIN" ]] || { echo "缺二进制: $BIN"; echo "D2 GATE CHECK: FAIL"; exit 1; }
[[ -f "$SRC" ]] || { echo "缺 token 文件: $SRC"; echo "D2 GATE CHECK: FAIL"; exit 1; }
grep -E '^(MODEL_FILE|OVERLAY_FILE)=' service.conf

# 防 amdgpu SVM 死锁：先把 GGUF 权重逐出 page cache（同 pp128k_bench.sh / gamma_128k_verify.sh）
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
    echo "!! amdgpu SVM 死锁（svm_range_cpu_invalidate_pagetables），中止，需重启"; echo "D2 GATE CHECK: FAIL"; exit 1
  fi
}
extra=()
[[ -n "$SMP" ]] && extra=("QWENOX_SPEC_SAMPLE=$SMP")
fail=0
run() {  # run LABEL TOKFILE [ENV=...]   额外环境变量为空时不传
  local L=$1 T=$2; shift 2
  echo "  .. $L（$(date +%T)）"
  BIN=$BIN bash tools/pp_prod.sh "$T" "$L" "$@" > "logs/d2g/$L.stdout" 2>&1
  local rc=$?
  svm_check
  if (( rc )); then echo "  FAIL: $L rc=$rc"; tail -n 5 "logs/d2g/$L.stdout"; fail=1; return 1; fi
  grep -aq 'dense: 8-bit' "logs/$L.log" || { echo "  $L: 日志没有 'dense: 8-bit'，不是 HQ 权重，停止"; echo "D2 GATE CHECK: FAIL"; exit 1; }
}

head -n 1024 "$SRC" > logs/d2g/tok1024.txt
echo "-- ctx 1024：VBENCH（扣除上下文相关成本用）"
run d2g_1024_vb logs/d2g/tok1024.txt "QWENOX_VBENCH=1,2,3,4,5,6,7,8" && grep -a '^vbench P=' logs/d2g_1024_vb.log | sed 's/^/     /'
for N in $CTXS; do
  T=logs/d2g/tok$N.txt
  head -n "$N" "$SRC" > "$T"
  (( $(wc -w <"$T") == N )) || { echo "token 文件不足 $N"; echo "D2 GATE CHECK: FAIL"; exit 1; }
  echo "-- ctx $N：VBENCH"
  run "d2g_${N}_vb" "$T" "QWENOX_VBENCH=$PS" || continue
  grep -a '^vbench P=' "logs/d2g_${N}_vb.log" | sed 's/^/     /'
  echo "-- ctx $N：单路投机 γ ∈ {$GAMMAS}（0 = 自适应）"
  for G in $GAMMAS; do
    SPEC=$NS GAMMA=$G run "d2g_${N}_g$G" "$T" "${extra[@]}" || continue
    grep -ahE '^spec(-sample)?: |^chain(-sample)?: [0-9]+ tokens' "logs/d2g_${N}_g$G.log" | tail -1 | sed 's/^/     /'
  done
done
echo "== 测量结束 $(date +%T)"
(( fail )) && { echo "D2 GATE CHECK: FAIL（有运行失败，见上）"; exit 1; }

python3 - "$GATE" $CTXS <<'PY'
import re, sys
from pathlib import Path
gate = float(sys.argv[1]); ctxs = sys.argv[2:]
spec_re = re.compile(r'^(?:spec|chain)(?:-sample)?: (\d+) tokens in ([\d.]+) s = ([\d.]+) tok/s \| rounds=(\d+) commit/round=([\d.]+)')

def vb(path):
    T = {}
    for ln in Path(path).read_text(errors='replace').splitlines():
        m = re.match(r'vbench P=(\d+)(?: segs=(\d+))? median=([\d.]+)', ln)
        if m:
            T[(int(m[1]), int(m[2] or 1))] = float(m[3])
    return T

T1K = vb('logs/d2g_1024_vb.log')
worst, worst_up = 0.0, 0.0
for N in ctxs:
    T = vb(f'logs/d2g_{N}_vb.log')
    S = {}
    for g in (1, 2, 3, 4, 7, 0):
        for ln in reversed(Path(f'logs/d2g_{N}_g{g}.log').read_text(errors='replace').splitlines()):
            m = spec_re.match(ln)
            if m:
                sec, tps, rounds, cpr = float(m[2]), float(m[3]), int(m[4]), float(m[5])
                S[g] = dict(tps=tps, tr=sec * 1000 / rounds, e=cpr)
                break
    print(f'\n==== ctx {N} ====')
    print('  T(P) ms:   ' + '  '.join(f'{p}:{v:.1f}' for (p, b), v in sorted(T.items()) if b == 1))
    print('  T(R×B) ms: ' + '  '.join(f'{p}×{b}:{v:.1f}' for (p, b), v in sorted(T.items()) if b > 1))
    print('  C(p)=T_N−T_1K ms: ' + '  '.join(f'{p}:{T[(p, 1)] - T1K[(p, 1)]:.1f}' for p in range(1, 9)
                                          if (p, 1) in T and (p, 1) in T1K))
    for g in sorted(S, key=lambda x: (x == 0, x)):
        s = S[g]
        print(f'  单路 γ={"自适应" if g == 0 else g}: {s["tps"]:.1f} tok/s  Tr={s["tr"]:.1f} ms/轮  E={s["e"]:.2f} token/轮')
    base_g = max(S, key=lambda g: S[g]['tps'])
    base = S[base_g]['tps']
    print(f'  分时基线 = 单路最好 {base:.1f} tok/s（γ={"自适应" if base_g == 0 else base_g}）')
    print(f'  {"B":>2} {"γb":>3} {"R":>3} | {"上限 ×":>7} | {"估算 ms/轮":>10} {"估算 tok/s":>10} {"每人":>6} {"估算 ×":>7} | {"理想 ×":>7}')
    best, best_up = 0.0, 0.0
    for B, gb in ((2, 1), (2, 2), (2, 3), (2, 7), (4, 1), (4, 2), (4, 3)):
        p, R = gb + 1, B * (gb + 1)
        need = [(p, 1), (R, 1), (R, B)]
        if gb not in S or any(k not in T for k in need) or (p, 1) not in T1K:
            print(f'  {B:>2} {gb:>3} {R:>3} | 缺数据'); continue
        tr, e = S[gb]['tr'], S[gb]['e']
        C = max(0.0, T[(p, 1)] - T1K[(p, 1)])
        up = B * tr - (B * T[(p, 1)] - T[(R, 1)])
        est = B * tr - (B * T[(p, 1)] - T[(R, B)]) + (B - 1) * C
        ideal = tr + (T[(R, B)] - T[(p, 1)]) + (B - 1) * C
        x_up, x_est, x_id = (B * e / v * 1000 / base for v in (up, est, ideal))
        best, best_up = max(best, x_est), max(best_up, x_up)
        print(f'  {B:>2} {gb:>3} {R:>3} | {x_up:>7.2f} | {est:>10.1f} {B * e / est * 1000:>10.1f} '
              f'{e / est * 1000:>6.1f} {x_est:>7.2f} | {x_id:>7.2f}')
    print(f'  → ctx {N}：合批估算最高 {best:.2f}×（上限 {best_up:.2f}×），闸门 {gate:.2f}×')
    worst, worst_up = max(worst, best), max(worst_up, best_up)
ok = worst < gate
print(f'\n结论：合批估算最高 {worst:.2f}×（同序列上限 {worst_up:.2f}×），{"低于" if ok else "达到"}闸门 {gate:.2f}×')
print('D2 GATE CHECK: ' + ('PASS（执行方"D2 不值得做"的结论成立）' if ok else 'FAIL（估算过闸门，D2 需重新评估）'))
sys.exit(0 if ok else 1)
PY
