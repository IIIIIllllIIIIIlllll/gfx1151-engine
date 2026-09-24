#!/usr/bin/env bash
# 正式对比：同一个基准文件（应由 BF16 GGUF 生成，见 KLD.md）上跑引擎和 unsloth 量化。
#   bash tools/kld_bench.sh <ref.kld>
#     ONLY="eng q4 iq1"   选择要跑的项（默认全部）
#     ENG_VARIANTS="prod" 引擎变体，prod = start.sh 生产配置；
#                         再加 "name:K=V,K=V" 形式可做 env A/B（如 "nomtp:GDEC_X=1"）
#     BIN=build/gdec      引擎二进制
# 输出：每项 logs/kld_b_<项>.log，最后打印对比表（含 unsloth 公布值）。
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
[[ $# -ge 1 && -f "$1" ]] || { sed -n '2,8p' "$0"; exit 1; }
REF=$1
ONLY=${ONLY:-eng q4 iq1}
MD=$HOME/App/llama.cpp/models
Q4=${Q4:-$(ls "$MD"/Qwen3.8-Flash-Next-UD-Q4_K_XL/*-00001-of-*.gguf 2>/dev/null | head -1)}
IQ1=${IQ1:-$(ls "$MD"/Qwen3.8-Flash-Next-UD-IQ1_S/*-00001-of-*.gguf 2>/dev/null | head -1)}
if pgrep -x gdec >/dev/null || pgrep -f 'llama-(server|perplexity|cli)' >/dev/null; then
  echo "已有 gdec / llama 进程在跑，先停掉:"; pgrep -af 'gdec|llama-' | grep -v pgrep | cut -c1-150; exit 1
fi
TABLE=()
for it in $ONLY; do
  case $it in
    eng)
      for v in ${ENG_VARIANTS:-prod}; do
        name=${v%%:*}; envs=()
        [[ $v == *:* ]] && IFS=, read -r -a envs <<<"${v#*:}"
        bash tools/kld_engine.sh "$REF" "b_eng_$name" "${envs[@]}"
        TABLE+=("engine_$name=logs/kld_b_eng_$name.log")
      done ;;
    q4)  bash tools/kld_llama.sh cmp "$Q4" "$REF" b_q4;  TABLE+=("llama_UD-Q4_K_XL=logs/kld_b_q4.log") ;;
    iq1) bash tools/kld_llama.sh cmp "$IQ1" "$REF" b_iq1; TABLE+=("llama_UD-IQ1_S=logs/kld_b_iq1.log") ;;
    *) echo "未知项 $it" ;;
  esac
done
echo
python3 tools/kld_table.py "${TABLE[@]}" --unsloth
