#!/usr/bin/env bash
# pp_bench.sh — prefill(PP) 性能测试：扫一个或多个上下文长度，输出 tok/s 汇总。
#
# 用法（在仓库任意目录都可执行）:
#   tools/pp_bench.sh 8192 32768                # 测 8K 和 32K
#   tools/pp_bench.sh -r 2 16384 32768          # 每个长度跑 2 次
#   tools/pp_bench.sh 32768 GDEC_QSA_UNION=1    # 叠加 env 开关
#   tools/pp_bench.sh 32768 -GDEC_GDN_FUSED     # 取消某个默认开关
#   BIN=build/gdec_union tools/pp_bench.sh 32768 GDEC_QSA_UNION=1   # 换二进制
#
# 选项:
#   -r N      每个长度重复 N 次（默认 1）
#   -s FILE   token 源文件（默认 data/qsa-oracle/131072.tokens，每行一个 id，
#             按长度截取；即最大可测 131072）
#
# 默认 env = 生产 flags（对齐 start.sh，GDEC_PREFILL_CHUNK=32768）。
# token 文件与日志都在 data/ppbench/（已 gitignore）， tok<长度>.txt 自动生成并复用。
set -u
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="${BIN:-build/gdec}"
TOKDIR="$ROOT/data/ppbench"
SRC="$ROOT/data/qsa-oracle/131072.tokens"
RUNS=1
LENGTHS=()
ENVARGS=()
UNSETS=()

while [ $# -gt 0 ]; do
  case "$1" in
    -r) RUNS="$2"; shift 2 ;;
    -s) SRC="$2"; shift 2 ;;
    -h|--help) sed -n '2,21p' "$0"; exit 0 ;;
    -*) UNSETS+=(-u "${1#-}"); shift ;;
    *[!0-9]*) ENVARGS+=("$1"); shift ;;
    *) LENGTHS+=("$1"); shift ;;
  esac
done
[ ${#LENGTHS[@]} -gt 0 ] || { sed -n '2,21p' "$0" >&2; exit 2; }

if pgrep -af '(^|/)(gdec[^/[:space:]]*|flash_serve)([[:space:]]|$)' >/dev/null; then
  echo "已有引擎在运行，先停掉再测：" >&2
  pgrep -af '(^|/)(gdec[^/[:space:]]*|flash_serve)([[:space:]]|$)' >&2
  exit 1
fi
[ -x "$ROOT/$BIN" ] || { echo "找不到 $ROOT/$BIN —— 先 bash build.sh engine" >&2; exit 1; }
[ -f "$SRC" ] || { echo "找不到 token 源文件 $SRC" >&2; exit 1; }
srclines=$(wc -l < "$SRC")
mkdir -p "$TOKDIR/logs"

declare -a RESULTS
for L in "${LENGTHS[@]}"; do
  TOK="$TOKDIR/tok${L}.txt"
  if [ ! -f "$TOK" ] || [ "$(wc -l < "$TOK")" -lt "$L" ]; then
    [ "$srclines" -ge "$L" ] || { echo "源文件只有 $srclines 行，不够 $L" >&2; exit 1; }
    head -n "$L" "$SRC" > "$TOK.tmp.$$" && mv "$TOK.tmp.$$" "$TOK"
    echo "生成 $TOK（$L 行）"
  fi
  MAXCTX=$((L + 8192))
  for i in $(seq 1 "$RUNS"); do
    LABEL="pp${L}_r${i}"
    LOG="$TOKDIR/logs/${LABEL}.log"
    echo "== $LABEL: ${L} tokens, maxctx=$MAXCTX =="
    ( cd "$ROOT" && env "${UNSETS[@]}" \
        GDEC_QSA_KV_BF16=1 GDEC_QSA_WMMA=1 GDEC_QSA_WMMA_BTV=1 \
        GDEC_MOE_LT=1 GDEC_MOE_LT_BF16=1 GDEC_GR_BF16=1 \
        GDEC_GDN_STREAM=1 GDEC_GDN_WAVE=1 GDEC_NOWARMUP=1 \
        GDEC_PREFILL_CHUNK=32768 GDEC_GEMM_WMMA=1 GDEC_GDN_FUSED=1 \
        GDEC_INDEX_FUSED2=1 GDEC_PP_MOE_OUT=1 GDEC_INDEX_STREAM_SELECT=1 \
        GDEC_KVSNAP=1 GDEC_KVSNAP_MAX_GB=20 \
        GDEC_PROF=1 GDEC_PHASE=1 "${ENVARGS[@]}" \
        bash tools/run_capped.sh 86 -- \
        "$BIN" models/qwen38-flash-next-w4b.hgn models/qwen38-flash-next-w4b.overlay.hgn \
        --tokens-file "$TOK" --gen 1 --maxctx "$MAXCTX" >"$LOG" 2>&1 )
    rc=$?
    RATES=$(grep -oE '= [0-9.]+ tok/s' "$LOG" | awk '{print $2}')
    MEAN=$(echo "$RATES" | awk '{s+=$1; n++} END {if (n) printf "%.1f", s/n; else print "N/A"}')
    if [ "$rc" -ne 0 ] || [ "$MEAN" = "N/A" ]; then
      echo "   rc=$rc 失败，看 $LOG" >&2
      tail -5 "$LOG" >&2
      RESULTS+=("$L|$i|FAILED(rc=$rc)||$LOG")
      continue
    fi
    echo "   平均 $MEAN tok/s（chunk: $(echo $RATES | tr '\n' ' ')）"
    RESULTS+=("$L|$i|$MEAN|$(echo $RATES | tr '\n' ' ')|$LOG")
  done
done

echo
echo "==== 汇总 ===="
printf "%-8s %-5s %-12s %s\n" 长度 次数 平均tok/s 各chunk
for r in "${RESULTS[@]}"; do
  IFS='|' read -r L i M C LOG <<< "$r"
  printf "%-8s %-5s %-12s %s\n" "$L" "$i" "$M" "$C"
done
echo "日志: $TOKDIR/logs/"
