#!/usr/bin/env bash
# 128K prefill 基准：hgn 与 GGUF 权重，生产 env（start_hgn.sh / start_gguf.sh --check），
# 另跑一轮 hgn chunk 8192 与旧基线（tools/pp.sh 口径 1345 tok/s）对照。
#   bash tools/pp128k_bench.sh            （约 8-10 分钟，末行 PASS/FAIL）
#     BIN=build/qwenox-epi   换二进制（默认 build/qwenox-engine）
#     TOK=<file>           token 文件（默认 data/qsa-oracle/131072.tokens）
#     SKIP8K=1             不跑 hgn chunk 8192 那一轮
# 每轮之间把另一种格式的权重逐出 page cache（防 amdgpu SVM 死锁），跑完查 journalctl。
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
BIN=${BIN:-build/qwenox-engine}
TOK=${TOK:-data/qsa-oracle/131072.tokens}
T0="$(date '+%Y-%m-%d %H:%M:%S')"
[[ -x $BIN ]] || { echo "没有 $BIN"; echo FAIL; exit 1; }
[[ -f $TOK ]] || { echo "没有 $TOK"; echo FAIL; exit 1; }

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
wfiles() {  # 启动器 --check 里的权重文件
  bash "$1" --check 2>/dev/null | sed -n 's/^CMD //p' | tr ' ' '\n' | grep -E '\.(hgn|gguf)$'
}
svm_check() {
  if journalctl -k --since "$T0" --no-pager 2>/dev/null | grep -q svm_range_cpu_invalidate_pagetables; then
    echo "!! amdgpu SVM 死锁（svm_range_cpu_invalidate_pagetables），中止，需重启"; echo FAIL; exit 1
  fi
}

declare -a ROWS
fail=0
run() {  # 名称 启动器 [K=V...]
  local name=$1 la=$2; shift 2
  local label=pp128k_$name other
  if [[ $la == start_hgn.sh ]]; then other=start_gguf.sh; else other=start_hgn.sh; fi
  mapfile -t OF < <(wfiles "$other")
  (( ${#OF[@]} )) && evict "${OF[@]}"
  echo "== $name：$la $* （$(date +%H:%M:%S)）"
  env LAUNCHER="$la" BIN="$BIN" bash tools/pp_prod.sh "$TOK" "$label" "$@" >"logs/$label.out" 2>&1
  local rc=$?
  svm_check
  local log=logs/$label.log r
  r=$(tr '\r' '\n' <"$log" | grep -ao 'prefill: [0-9]* tokens in [0-9.]* s = [0-9.]* tok/s' |
      awk '{n += $2; s += $5; c++; if (c == 1) f = $8; l = $8; if (c > 1 && $8 < mn || c == 1) mn = $8}
           END {if (n) printf "%d %.1f %.0f %.0f %.0f %.0f %d\n", n, s, n / s, f, l, mn, c}')
  local wall
  wall=$(grep -ao 'wall=[0-9]*s' "logs/$label.out" | tail -1)
  if (( rc )) || [[ -z $r ]]; then
    echo "   失败 rc=$rc，见 $log"; tail -n 5 "logs/$label.out" | sed 's/^/   /'
    ROWS+=("$name|失败||||||"); fail=1; return
  fi
  read -r n s all f l mn c <<<"$r"
  echo "   $n token / $c chunk：$s s，整体 $all tok/s，首 chunk $f，末 chunk $l，最低 $mn（$wall）"
  ROWS+=("$name|$n|$c|$s|$all|$f|$l|$mn")
}

echo "bin=$BIN tok=$TOK ($(wc -w <"$TOK") token)"
run hgn start_hgn.sh
run gguf start_gguf.sh
[[ ${SKIP8K:-0} == 1 ]] || run hgn_c8192 start_hgn.sh QWENOX_PREFILL_CHUNK=8192

echo
printf '%-11s %7s %5s %7s %8s %6s %6s %6s\n' 配置 token chunk 秒 整体tok/s 首块 末块 最低
for r in "${ROWS[@]}"; do IFS='|' read -r a b c d e f g h <<<"$r"
  printf '%-11s %7s %5s %7s %8s %6s %6s %6s\n' "$a" "$b" "$c" "$d" "$e" "$f" "$g" "$h"; done
echo "参考：halogen 0.14.1 128K = 1517 tok/s（86.4 s）；本引擎 09-28 早先 hgn chunk 8192 = 1345（97.4 s）"
(( fail )) && { echo FAIL; exit 1; }
echo PASS
