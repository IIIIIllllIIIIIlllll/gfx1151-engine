#!/usr/bin/env bash
# 一键自检：引擎 --kld-base/--kld-save 与 llama-perplexity 的 KLD 文件双向兼容、统计口径一致。
# 没有 BF16 也能跑：用 llama.cpp 的 UD-Q4_K_XL 充当临时基准。
#   bash tools/kld_verify.sh            （约 4 分钟；CHUNKS=8 可改）
#   BIN=... 换引擎二进制；Q4=... 换 GGUF（第一个分片）
# 步骤：
#   1 llama Q4 写基准 q4.kld（BATCH=2048，n_seq=4）
#   2 llama Q4 对 q4.kld（BATCH=512，n_seq=1）       → llama 自身噪声底；验证多序列 = 单序列
#   3 引擎 对 q4.kld，同时 --kld-save eng.kld       → 读文件；PPL(base) 必须与第 2 步一致
#   4 引擎 对 eng.kld                               → 写文件 + 确定性：KLD≈0、top-1≈100%
#   5 llama Q4 对 eng.kld                           → llama 能读引擎写的文件；PPL 两两一致
# 结果：logs/kld_v_*.log，数据 data/kld/verify/（约 2 GB，可删）。末行 KLD VERIFY: PASS/FAIL。
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export CHUNKS=${CHUNKS:-8}
Q4=${Q4:-$(ls "$HOME"/App/llama.cpp/models/Qwen3.8-Flash-Next-UD-Q4_K_XL/*-00001-of-*.gguf 2>/dev/null | head -1)}
D=data/kld/verify
[[ -f "$Q4" ]] || { echo "找不到 Q4_K_XL GGUF（设 Q4=...）"; echo "KLD VERIFY: FAIL"; exit 1; }
[[ -x "${BIN:-build/gdec}" ]] || { echo "找不到引擎 ${BIN:-build/gdec}"; echo "KLD VERIFY: FAIL"; exit 1; }
if pgrep -x gdec >/dev/null || pgrep -f 'llama-(server|perplexity|cli)' >/dev/null; then
  echo "已有 gdec / llama 进程在跑（显存不够两个模型），先停掉再测:"
  pgrep -af 'gdec|llama-' | grep -v pgrep | cut -c1-150
  echo "KLD VERIFY: FAIL"; exit 1
fi
mkdir -p "$D"
rm -f "$D"/*.kld
step() { echo; echo "=== $1"; shift; "$@" || { echo "步骤失败 rc=$?"; echo "KLD VERIFY: FAIL"; exit 1; }; }
step "1 llama 写基准" env BATCH=2048 bash tools/kld_llama.sh base "$Q4" "$D/q4.kld" v_q4_base
step "2 llama 自比（n_seq=1）" env BATCH=512 bash tools/kld_llama.sh cmp "$Q4" "$D/q4.kld" v_q4_self
step "3 引擎 vs llama 基准" env SAVE="$D/eng.kld" CHUNKS= bash tools/kld_engine.sh "$D/q4.kld" v_eng_q4
step "4 引擎自比" env CHUNKS= bash tools/kld_engine.sh "$D/eng.kld" v_eng_self
step "5 llama 读引擎文件" bash tools/kld_llama.sh cmp "$Q4" "$D/eng.kld" v_q4_vs_eng

echo
python3 tools/kld_table.py llama_self=logs/kld_v_q4_self.log eng_vs_llamaQ4=logs/kld_v_eng_q4.log \
  eng_self=logs/kld_v_eng_self.log llamaQ4_vs_eng=logs/kld_v_q4_vs_eng.log
python3 - <<'EOF'
import sys
sys.path.insert(0, "tools")
from kld_table import parse
L = {k: parse(f"logs/kld_v_{k}.log") for k in ("q4_self", "eng_q4", "eng_self", "q4_vs_eng")}
ok = True
def check(name, cond, detail):
    global ok
    print(f"{'PASS' if cond else 'FAIL'}  {name}: {detail}")
    ok &= bool(cond)
def rel(a, b):
    return abs(a - b) / max(abs(b), 1e-12)
try:
    s, e, es, le = L["q4_self"], L["eng_q4"], L["eng_self"], L["q4_vs_eng"]
    check("llama 自比噪声底", s["kld"][0] < 5e-3 and s["top"][0] > 98,
          f"KLD {s['kld'][0]:.6f}  top-1 {s['top'][0]:.3f}%  (n_seq 4 写 / 1 读)")
    check("引擎读 llama 文件：PPL(base) 一致", rel(e["ppl_base"][0], s["ppl_base"][0]) < 1e-4,
          f"引擎 {e['ppl_base'][0]:.6f} vs llama {s['ppl_base'][0]:.6f}")
    check("引擎 vs llama Q4 合理", 0 < e["kld"][0] < 0.3 and e["top"][0] > 80,
          f"KLD {e['kld'][0]:.6f}  99.9% {e['p999'][0]:.4f}  top-1 {e['top'][0]:.3f}%")
    check("引擎自比（写文件 + 确定性）", es["kld"][0] < 1e-4 and es["top"][0] > 99.9,
          f"KLD {es['kld'][0]:.2e}  top-1 {es['top'][0]:.3f}%")
    # 文件只保存 [max-16, max] 内的 log-prob，更低的被截到下限，所以极少数 NLL>16 的目标 token
    # 在文件里被低估：PPL(base) 恒略低于真实 PPL（llama 自己也一样，约 0.1~0.2%）
    check("引擎自比 PPL(Q)≈PPL(base)（文件截断只会让 base 偏低）",
          0 <= es["ppl_q"][0] - es["ppl_base"][0] and rel(es["ppl_q"][0], es["ppl_base"][0]) < 5e-3,
          f"{es['ppl_q'][0]:.6f} vs {es['ppl_base'][0]:.6f}")
    check("引擎 PPL(Q) 两次一致", rel(es["ppl_q"][0], e["ppl_q"][0]) < 1e-6,
          f"{es['ppl_q'][0]:.6f} vs {e['ppl_q'][0]:.6f}")
    check("llama 读引擎文件：PPL(base) 一致", rel(le["ppl_base"][0], es["ppl_base"][0]) < 1e-4,
          f"llama {le['ppl_base'][0]:.6f} vs 引擎 {es['ppl_base'][0]:.6f}")
    check("llama Q4 PPL(Q) 两次一致", rel(le["ppl_q"][0], s["ppl_q"][0]) < 1e-3,
          f"{le['ppl_q'][0]:.6f} vs {s['ppl_q'][0]:.6f}")
except KeyError as err:
    check("日志解析", False, f"缺字段 {err}")
print("KLD VERIFY:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
EOF
