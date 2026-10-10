#!/usr/bin/env bash
# hgn v2（halogen 0.15）与 w4b 同口径性能对比（前台跑，最后一行 PASS/FAIL）。
#   bash tools/bench_v2.sh
#     VARIANTS="w4b v2"   测哪几种、按什么顺序
#     SKIP="kld"          透传给 bench_full.sh（默认跳过 KLD，用 tools/v2_verify.sh 测）
# 每种权重各跑一遍 tools/bench_full.sh（FORMATS=hgn）：prefill 8K/32K/64K、decode@32K、
# MTP 投机、API 端到端与 4 路并发。权重用环境变量覆盖 service.conf，不改配置文件；
# 两种都不挂视觉塔。w4b = 主模型 + overlay + 8-bit MTP；v2 = 主体（自带同一份 MTP）+ n-gram 文件。
# 每种之前把所有 .hgn 逐出 page cache。输出：logs/bench_v2_<种类>/（bench_full 的全部日志）、
# 汇总 logs/bench_v2.md。
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
VARIANTS=${VARIANTS:-w4b v2}
export SKIP=${SKIP:-kld}
D=models
fail=0
evict() {
  python3 - "$@" <<'PY'
import os, sys
for f in sys.argv[1:]:
    try:
        fd = os.open(f, os.O_RDONLY); os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED); os.close(fd)
    except OSError:
        pass
PY
}
for v in $VARIANTS; do
  case $v in
    w4b) export MODEL_FILE=$D/old/qwen38-flash-next-w4b.hgn OVERLAY_FILE=$D/old/qwen38-flash-next-w4b.overlay.hgn \
           MTP_FILE=$D/old/qwen38-flash-next-mtp.hgn VISION_FILE=
         unset NGRAM_FILE ;;
    v2)  export MODEL_FILE=$D/qwen38-flash-next-v2.hgn NGRAM_FILE=$D/qwen38-flash-next-ngram.hgn \
           OVERLAY_FILE= MTP_FILE= VISION_FILE= ;;
    *) echo "VARIANTS 只能是 w4b / v2"; echo FAIL; exit 1 ;;
  esac
  echo "==== $(date +%H:%M:%S) $v：$MODEL_FILE ${NGRAM_FILE:-} ${OVERLAY_FILE:-} ${MTP_FILE:-}"
  evict $D/*.hgn $D/old/*.hgn
  FORMATS=hgn bash tools/bench_full.sh 2>&1 | tee "logs/bench_v2_$v.out"
  [[ $(tail -1 "logs/bench_v2_$v.out") == PASS ]] || { echo "$v：bench_full FAIL"; fail=1; }
  rm -rf "logs/bench_v2_$v"; mkdir -p "logs/bench_v2_$v"
  cp logs/bench_full.md logs/bench_hgn_* "logs/bench_v2_$v/" 2>/dev/null
done

# 合并：每份 bench_full.md 的 hgn 列并排
{
  echo "# hgn v2 vs w4b 性能（$(date '+%Y-%m-%d %H:%M')，BIN=${BIN:-build/qwenox-engine}，无视觉塔，SKIP=$SKIP）"
  echo
  printf '| 项目 |'; for v in $VARIANTS; do printf ' %s |' "$v"; done; echo
  printf '|---|'; for v in $VARIANTS; do printf -- '---|'; done; echo
  first=${VARIANTS%% *}
  grep '^| ' "logs/bench_v2_$first/bench_full.md" | sed -n '2,$p' | cut -d'|' -f2 | while read -r item; do
    printf '| %s |' "$item"
    for v in $VARIANTS; do
      printf ' %s |' "$(grep -F "| $item |" "logs/bench_v2_$v/bench_full.md" | head -1 | cut -d'|' -f3 | sed 's/^ *//; s/ *$//')"
    done
    echo
  done
} | tee logs/bench_v2.md
if (( fail )); then echo FAIL; exit 1; else echo PASS; fi
