#!/usr/bin/env bash
# hgn 权重启动器（Linux）：models/ 下的 qwen38-flash-next-*.hgn。llama.cpp GGUF 权重请用 start_gguf.sh。
# 前台运行，Ctrl+C 同时停止本次启动的 API 和引擎；--check 只检查配置、不启动。
# 路径与部署参数见 service.conf（"hgn 权重"一段），同名环境变量优先。
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
LAUNCHER=start_hgn.sh
source "$ROOT/tools/serve_common.sh"
serve_init "$@"

for f in "$MODEL_FILE" "$OVERLAY_FILE" "$MTP_FILE" "$VISION_FILE"; do
  [[ -z "$f" || "$f" == *.hgn ]] || fail "$LAUNCHER 只接受 .hgn 权重：$f（GGUF 请用 bash start_gguf.sh）"
done
need 主模型 "$MODEL_FILE"
[[ -z "$OVERLAY_FILE" ]] || need overlay "$OVERLAY_FILE" '不需要可设 OVERLAY_FILE=""'
[[ -z "$MTP_FILE" ]] || need 'MTP 草稿' "$MTP_FILE" '设 MTP_FILE="" 退回 overlay 内置草稿头'
[[ -z "$VISION_FILE" ]] || need 视觉塔 "$VISION_FILE" '纯文本可设 VISION_FILE=""'
MISSING_HINT='hgn 权重由 tools/flashnext2hgn.py 转换生成（见 CONVERT.md）；手上是 llama.cpp GGUF 请用 bash start_gguf.sh（见 GGUF.md）。'

FORMAT=hgn
MAIN_MODEL=$MODEL_FILE
MODEL_ARGS=("$MODEL_FILE")
[[ -z "$OVERLAY_FILE" ]] || MODEL_ARGS+=("$OVERLAY_FILE")
[[ -z "$MTP_FILE" ]] || MODEL_ARGS+=("$MTP_FILE")
VISION=$VISION_FILE
serve_run
