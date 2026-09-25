#!/usr/bin/env bash
# #78 Q0: 质量探针——hgn q4cp 路由专家 + GGUF Q8_0 dense（全部 / 按组）的 KLD。
#   bash tools/hq_probe.sh            → logs/hq_probe.out（每行一个配置）
#   BIN=build/gdec-hq  SETS="all gdn qsa ..."  CHUNKS=64
# 组 = GDEC_GGUF_DENSE_FILTER 的 hgn 名子串；未选中的组保持 hgn overlay 4-bit。
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
BIN=${BIN:-build/gdec-hq}
GG=${GG:-models/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf}
KREF=data/kld/bf16_c512.kld
OUT=logs/hq_probe.out
declare -A F=(
  [all]=""
  [gdn]=".linear_attn."
  [qsa]=".self_attn."
  [shexp]=".mlp.shared_expert"
  [hc]="hyper_connection_mixer,output_hc"
  [lmhead]="lm_head"
  [embed]="embed_tokens"
  [ple]=".ple."
  [hcl]="_hyper_connection."
  [xgdn]="!.linear_attn."
  [xqsa]="!.self_attn."
  [xshexp]="!.mlp.shared_expert"
  [xhcl]="!_hyper_connection."
  [xlmhead]="!lm_head"
  [xembed]="!embed_tokens"
)
SETS=${SETS:-"all gdn qsa shexp hc lmhead embed ple"}
for f in "$BIN" "$GG" "$KREF"; do [[ -e $f ]] || { echo "缺文件 $f"; exit 1; }; done
echo "== hq_probe $(date '+%F %T') bin=$BIN" | tee -a "$OUT"
for g in $SETS; do
  svm=$(journalctl -k -b 2>/dev/null | grep -c svm_range_cpu_invalidate_pagetables)
  [[ $svm -eq 0 ]] || { echo "SVM 告警 $svm，停止" | tee -a "$OUT"; exit 1; }
  E=(GDEC_GGUF="$GG" GDEC_GGUF_DENSE=1 GDEC_GGUF_EXPERTS=0)
  [[ -n ${F[$g]} ]] && E+=(GDEC_GGUF_DENSE_FILTER="${F[$g]}")
  r=$(BIN=$BIN CHUNKS=${CHUNKS:-} bash tools/kld_engine.sh "$KREF" hq_$g "${E[@]}" 2>&1)
  k=$(grep -m1 '^Mean    KLD' <<<"$r" | awk '{print $3}')
  t=$(grep -m1 '^Same top p' <<<"$r" | awk '{print $4}')
  p=$(grep -m1 '^Mean PPL(Q)' <<<"$r" | awk '{print $4}')
  w=$(grep -m1 'rc=' <<<"$r" | sed 's/.*\(rc=[0-9]*\) wall=\([0-9]*s\).*/\1 \2/')
  echo "$g	KLD=$k	top1=$t	PPL=$p	$w" | tee -a "$OUT"
done
echo "== done $(date '+%T')" | tee -a "$OUT"
