#!/usr/bin/env bash
# D1a 数值检查：prefill 分段更细会不会让结果离真值更远？前台约 3 分钟，最后一行 D1A KLD: PASS/FAIL。
#   bash tools/d1a_kld.sh
#   BIN=build/qwenox-engine REF=data/kld/bf16_c8192.kld TOL=1.05 bash tools/d1a_kld.sh
# 用现有的 BF16 基准（CTX=8192 × 8 段，见 KLD.md 长上下文一节），同一二进制跑两次：
#   整段   QWENOX_PREFILL_CHUNK=8192：每段一次 prefill（与 KLD.md 的 0.0392 相同）
#   分两段 QWENOX_PREFILL_CHUNK=4096：4096 + 4096，被统计的后半正好落在第二段
# 判定：分两段的 mean KLD ≤ 整段 × TOL。没有 16K 的 BF16 基准，这里用 8K→4K 代表 16K→8K 的分段变化。
# 注意：分段与不分段不逐位一致，两者直接互比 KLD 约 0.015（16K 上下文实测），但对真值两者一样好
# （09-29：整段 0.03917 / 分段 0.03876）——引擎互比不能当精度判据。
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
BIN="${BIN:-build/qwenox-engine}"
REF="${REF:-data/kld/bf16_c8192.kld}"
TOL="${TOL:-1.05}"
[[ -x "$BIN" ]] || { echo "缺二进制: $BIN"; echo "D1A KLD: FAIL"; exit 1; }
[[ -f "$REF" ]] || { echo "缺 BF16 基准: $REF（写法见 KLD.md）"; echo "D1A KLD: FAIL"; exit 1; }

for c in 8192 4096; do
  echo "== QWENOX_PREFILL_CHUNK=$c"
  BIN=$BIN MAXCTX=8448 bash tools/kld_engine.sh "$REF" "d1a_bf_c$c" "QWENOX_PREFILL_CHUNK=$c" \
    || { echo "D1A KLD: FAIL（运行失败）"; exit 1; }
  grep -aq 'dense: 8-bit' "logs/kld_d1a_bf_c$c.log" || { echo "不是 HQ 权重（日志无 'dense: 8-bit'）"; echo "D1A KLD: FAIL"; exit 1; }
done
grep -aq 'last chunk at 4096' logs/kld_d1a_bf_c4096.log \
  || { echo "4096 那次没有分段（日志无 'last chunk at 4096'）：$BIN 不含多段 KLD 支持"; echo "D1A KLD: FAIL"; exit 1; }

python3 - "$TOL" <<'PY'
import re, sys
tol = float(sys.argv[1])
def summ(c):
    s = open(f'logs/kld_d1a_bf_c{c}.log', errors='replace').read()
    m = re.search(r'kld_summary\t(.*)', s)
    return dict(kv.split('=') for kv in m[1].split('\t')) if m else None
a, b = summ(8192), summ(4096)
if not a or not b:
    print('缺 kld_summary 行'); print('D1A KLD: FAIL'); sys.exit(1)
for tag, d in (('整段 8192  ', a), ('分段 4096×2', b)):
    print(f'  {tag}: mean KLD {float(d["mean_kld"]):.6f}  p99.9 {float(d["p999_kld"]):.4f}  '
          f'top-1 {float(d["same_top"]):.3f}%  ppl {float(d["ppl_q"]):.4f}（BF16 {float(d["ppl_base"]):.4f}）')
ka, kb = float(a['mean_kld']), float(b['mean_kld'])
ok = kb <= ka * tol
print(f'  分段 / 整段 = {kb / ka:.3f}（允许 ≤ {tol}）')
print(f'D1A KLD: {"PASS" if ok else "FAIL"}')
sys.exit(0 if ok else 1)
PY
