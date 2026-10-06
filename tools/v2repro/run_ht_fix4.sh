#!/usr/bin/env bash
# 死行 nan 修复：重编码 4 个 svh 含 0 的 ht 张量（v2_encode.py 已打
# G[~isfinite]=0 补丁），然后重组装 + 离线验证。旧 part mv 成 .baknan。
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export PYTHONPATH=/home/mark/Workspace/pylib
export HTENC_BEAM=${HTENC_BEAM:-192}
export HTENC_SWEEPS=${HTENC_SWEEPS:-3}
export HTENC_GUIDED=${HTENC_GUIDED:-4}
PY=python3
W=/home/mark/Models/v2repro

step() { echo; echo "===== $(date +%H:%M:%S) $* ====="; }

step "re-encode 4 dead-row tensors"
for n in layers.1.linear_attn.in_proj_qkv.weight \
         layers.1.linear_attn.in_proj_z.weight \
         layers.2.linear_attn.in_proj_qkv.weight \
         layers.2.linear_attn.in_proj_z.weight; do
  if [ -f "$W/parts/ht/$n.bin" ]; then
    mv "$W/parts/ht/$n.bin" "$W/parts/ht/$n.bin.baknan"
  fi
done
$PY tools/v2repro/v2_encode.py ht || exit 1
grep -c "nan" /dev/null 2>/dev/null  # noop
$PY - <<'PYEOF'
# 确认这四个张量的新 part 不再含 nan 报告（直接看大小存在即可，质量由 verify 抽查）
import os
W = "/home/mark/Models/v2repro"
for n in ("layers.1.linear_attn.in_proj_qkv.weight",
          "layers.1.linear_attn.in_proj_z.weight",
          "layers.2.linear_attn.in_proj_qkv.weight",
          "layers.2.linear_attn.in_proj_z.weight"):
    p = os.path.join(W, "parts", "ht", n + ".bin")
    assert os.path.isfile(p), p
    print("  ok", n, os.path.getsize(p))
PYEOF

step "assemble"
if [ -f "$W/out/qwen38-flash-next-v2.hgn" ]; then
  mv "$W/out/qwen38-flash-next-v2.hgn" "$W/out/qwen38-flash-next-v2.hgn.bak3"
fi
$PY tools/v2repro/v2_assemble.py || exit 1

step "offline verify"
$PY tools/v2repro/v2_verify.py
RC=$?
step "done rc=$RC"
exit $RC
