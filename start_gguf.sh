#!/usr/bin/env bash
# GGUF 权重启动器（Linux）：models/ 下 Unsloth 的 Qwen3.8-Flash-Next UD-Q4_K_XL（与 llama.cpp 同一份文件，
# 不需要任何 .hgn）。hgn 权重请用 start_hgn.sh。
# 前台运行，Ctrl+C 同时停止本次启动的 API 和引擎；--check 只检查配置、不启动。
# 路径与部署参数见 service.conf（"GGUF 权重"一段），同名环境变量优先。
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
LAUNCHER=start_gguf.sh
source "$ROOT/tools/serve_common.sh"
serve_init "$@"

[[ -n "$GGUF_FILE" ]] || fail 'GGUF_FILE 不能为空'
for f in "$GGUF_FILE" "$GGUF_MTP_FILE" "$GGUF_VISION_FILE"; do
  [[ -z "$f" || "$f" == *.gguf ]] || fail "$LAUNCHER 只接受 .gguf 权重：$f（hgn 请用 bash start_hgn.sh）"
done
# 多分片：GGUF_FILE 是第 1 个分片 *-00001-of-0000N.gguf，其余分片必须在同一目录。
if [[ "$GGUF_FILE" =~ ^(.*-)00001-of-([0-9]{5})\.gguf$ ]]; then
  prefix=${BASH_REMATCH[1]}
  count=${BASH_REMATCH[2]}
  for ((i = 1; i <= 10#$count; i++)); do
    need "分片 $i/$((10#$count))" "$(printf '%s%05d-of-%s.gguf' "$prefix" "$i" "$count")"
  done
else
  need 主模型 "$GGUF_FILE"
fi
[[ -z "$GGUF_MTP_FILE" ]] || need 'MTP 草稿 sidecar' "$GGUF_MTP_FILE" '不用 MTP 投机可设 GGUF_MTP_FILE=""'
[[ -z "$GGUF_VISION_FILE" ]] || need '视觉塔 mmproj' "$GGUF_VISION_FILE" '纯文本可设 GGUF_VISION_FILE=""'
MISSING_HINT='GGUF 权重下载 Unsloth 的 Qwen3.8-Flash-Next UD-Q4_K_XL（全部分片 + mtp-*.gguf + mmproj），放进 models/（见 GGUF.md）；手上是 .hgn 请用 bash start_hgn.sh。'

# 引擎看到 *.gguf 基座即为纯 GGUF 启动（自动 GDEC_GGUF + GDEC_GGUF_DENSE=1）；
# MTP sidecar 总是显式传入（空串 = 不用），不让引擎去分片目录里猜。
export GDEC_GGUF_MTP="$GGUF_MTP_FILE"
FORMAT=GGUF
MAIN_MODEL=$GGUF_FILE
MODEL_ARGS=("$GGUF_FILE")
VISION=$GGUF_VISION_FILE
serve_run
