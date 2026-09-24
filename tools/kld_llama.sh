#!/usr/bin/env bash
# llama.cpp 侧 KLD（llama-perplexity），与 unsloth 表格同一套统计口径。
#   bash tools/kld_llama.sh base <model.gguf> <out.kld> [标签]   基准模型写 logits 文件
#   bash tools/kld_llama.sh cmp  <model.gguf> <ref.kld> [标签]   候选 GGUF 对比基准文件
# 分片 GGUF 传第一个分片（*-00001-of-0000N.gguf）即可。
# 环境变量：
#   LLAMA_PPL=路径     默认 ~/App/llama.cpp/llamacpp/llama.cpp-strix-halo/llama-perplexity
#   TEXT=语料          默认 wikitext-2-raw/wiki.test.raw（仅 base 用；cmp 的 token 取自 ref 文件）
#   CTX=512            每个 chunk 的 token 数（base 写进文件；cmp 必须 >= 文件里的 n_ctx）
#   CHUNKS=100         base 取多少个 chunk（-1 = 全部，wikitext-2 test 约 580 个）
#   BATCH=2048         一次并行 BATCH/CTX 个 chunk；UBATCH 默认同 BATCH
#   LLAMA_ARGS="-ngl 999 -fa on --threads 16"  额外参数（同 launch_config 的 Q4_K_XL 配置）。
#                      BF16（355 GB）放不进内存：用 "-ngl 0 -fa on --threads 16"，
#                      并把 BATCH/UBATCH 调大（如 4096），每一遍只从 SSD 流一次权重
# 输出：logs/kld_<标签>.log；打印 PPL/KLD/Same top 摘要行。ref 文件每 chunk ≈126 MB（CTX=512）。
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
[[ $# -ge 3 ]] || { sed -n '2,17p' "$0"; exit 1; }
mode=$1 model=$2 file=$3
LABEL=${4:-${mode}_$(basename "$model" .gguf)_$(date +%H%M%S)}
LLAMA_PPL=${LLAMA_PPL:-$HOME/App/llama.cpp/llamacpp/llama.cpp-strix-halo/llama-perplexity}
TEXT=${TEXT:-$HOME/App/llama.cpp/tools/perplexity/wikitext-2-raw/wiki.test.raw}
CTX=${CTX:-512}
CHUNKS=${CHUNKS:-100}
BATCH=${BATCH:-2048}
UBATCH=${UBATCH:-$BATCH}
read -r -a EXTRA <<<"${LLAMA_ARGS:--ngl 999 -fa on --threads 16}"
[[ -x "$LLAMA_PPL" ]] || { echo "找不到 llama-perplexity: $LLAMA_PPL"; exit 1; }
[[ -f "$model" ]] || { echo "找不到模型: $model"; exit 1; }
case "$mode" in
  base)
    [[ -f "$TEXT" ]] || { echo "找不到语料: $TEXT"; exit 1; }
    mkdir -p "$(dirname "$file")"
    ARGS=(-f "$TEXT" --chunks "$CHUNKS" --kl-divergence-base "$file") ;;
  cmp)
    [[ -f "$file" ]] || { echo "找不到基准文件: $file"; exit 1; }
    ARGS=(--kl-divergence-base "$file" --kl-divergence) ;;
  *) echo "模式只能是 base 或 cmp"; exit 1 ;;
esac
mkdir -p logs
LOG="logs/kld_$LABEL.log"
echo "[$LABEL] $mode model=$model file=$file ctx=$CTX batch=$BATCH/$UBATCH extra=${EXTRA[*]}"
t0=$SECONDS
"$LLAMA_PPL" -m "$model" -c "$CTX" -b "$BATCH" -ub "$UBATCH" "${EXTRA[@]}" "${ARGS[@]}" >"$LOG" 2>&1
rc=$?
if [[ $mode == base ]]; then
  grep -E 'Final estimate|computing over|calculating perplexity over|failed|error' "$LOG" | tail -5
  [[ -s "$file" ]] && ls -la "$file"
else
  grep -E '^(Mean PPL|Cor\(|Mean ln|Mean    KLD|Maximum KLD|99\.9%   KLD|Median  KLD|RMS Δp|Same top p)|failed|error' "$LOG"
fi
echo "[$LABEL] rc=$rc wall=$((SECONDS - t0))s log=$LOG"
exit $rc
