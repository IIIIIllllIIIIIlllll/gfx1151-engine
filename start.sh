#!/usr/bin/env bash
# start.sh 已按权重格式拆成两个启动器（参数与 --check 相同）。
cat >&2 <<'EOF'
错误：start.sh 已拆分，请按权重格式选择启动器：
  bash start_hgn.sh    hgn 权重（models/qwen38-flash-next-*.hgn）
  bash start_gguf.sh   GGUF 权重（models/Qwen3.8-Flash-Next-UD-Q4_K_XL-*.gguf，与 llama.cpp 同一份文件）
两者都读 service.conf，缺少权重时会列出缺失的文件。详见 QUICKSTART.md。
EOF
exit 1
