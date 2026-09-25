#!/usr/bin/env bash
# 一键完整性能测试：hgn（start_hgn.sh）与 GGUF（start_gguf.sh）同口径对比，最后打印汇总表与 PASS / FAIL。
#   bash tools/bench_full.sh
#     FORMATS="gguf hgn"   测哪几种、按什么顺序（只测一种写 FORMATS=gguf）
#     SKIP="kld api"       跳过其中几项：pp spec kld api（decode 跟 32K prefill 一起跑）
#     BIN=build/gdec
# 全部用启动器 --check 给出的生产环境变量与权重参数（service.conf 的 hgn / GGUF 两段）：
#   pp      离线 prefill 8K / 32K / 64K（生产 chunk 16384）：整体 tok/s（总 token / 总时间）与最后一个 chunk 的 tok/s
#   decode  32K prompt 之后逐 token 生成 128（不投机）
#   spec    8K prompt 之后 MTP 投机生成 256（gamma 3）：tok/s 与 commit/round
#   kld     BF16 基准 data/kld/bf16_c512.kld（64×512）：mean KLD、top1 相同率、PPL
#   api     启动器真正起服务（生产配置，只关 KVSNAP 免得 SSD 快照命中）：就绪用时与内存、两条 512 token 生成
#           （中文散文 / 代码，含 MTP 投机接受率）、~8K 与 ~32K prompt 的首 token 延迟与 prefill 速度、
#           4 路并发各生成 256 token 的总吞吐；数字取自 API 日志 / 引擎日志的逐请求统计
# 两种格式之间把另一种的权重逐出 page cache；每步后查 journalctl -k 有无 amdgpu SVM 死锁（有则中止）。
# 输出：logs/bench_<格式>_<项>.{log,out}，汇总 logs/bench_full.md
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
FORMATS=${FORMATS:-gguf hgn}
SKIP=" ${SKIP:-} "
export BIN=${BIN:-build/gdec}
KREF=data/kld/bf16_c512.kld
DOC=$HOME/ppbench/prompt32k.json   # {"text": 约 32K token 的项目文档}
OUT=logs/bench_full.md
API_PORT=$(bash -c 'source service.conf; echo $API_PORT')
mkdir -p logs
declare -A R
fail=0
note() { echo "== $(date +%H:%M:%S) $*"; }
bad() { echo "FAIL: $*"; fail=1; }
want() { [[ $SKIP != *" $1 "* ]]; }
launcher() { if [[ $1 == hgn ]]; then echo start_hgn.sh; else echo start_gguf.sh; fi; }
crashed() { tr '\r' '\n' <"$1" | grep -aqE 'hipError|Segmentation|Aborted|FATAL'; }

GPU_RE='(^|/)(gdec[^/[:space:]]*|flash_serve|serve_api\.py|llama-server|llama-perplexity|llama-cli)([[:space:]]|$)'
if pgrep -af "$GPU_RE" >/dev/null; then
  pgrep -af "$GPU_RE" | cut -c1-160; echo "GPU 上已有引擎或 llama.cpp 在跑，先停掉"; echo FAIL; exit 1
fi

row() { echo "| $1 | ${R[hgn.$2]:-—} | ${R[gguf.$2]:-—} |"; }
summary() {
  {
    echo "# 完整性能测试 $(date '+%Y-%m-%d %H:%M')（BIN=$BIN）"
    echo
    echo "| 项目 | hgn | GGUF |"
    echo "|---|---|---|"
    row "API 启动就绪" ready
    row "就绪后内存（device / RSS GiB）" mem
    row "prefill 8K（tok/s）" pp8k
    row "prefill 32K（整体 / 末 chunk tok/s）" pp32k
    row "prefill 64K（整体 / 末 chunk tok/s）" pp64k
    row "decode @32K，不投机（tok/s）" decode
    row "MTP 投机 8K+256（tok/s，commit/round）" spec
    row "KLD mean / top1 / PPL" kld
    row "API 中文散文 512 tok" api_prose
    row "API 代码 512 tok" api_code
    row "API 8K prompt" api_8k
    row "API 32K prompt" api_32k
    row "API 4 路并发 ×256 tok" api_conc
  } | tee "$OUT"
}

T0=$(date '+%Y-%m-%d %H:%M:%S')
svm_check() {
  if journalctl -k --since "$T0" --no-pager 2>/dev/null | grep -q svm_range_cpu_invalidate_pagetables; then
    bad "内核 amdgpu SVM 死锁（journalctl -k 有 svm_range_cpu_invalidate_pagetables），中止；需重启机器"
    summary; echo FAIL; exit 1
  fi
}
evict() {  # 把文件逐出 page cache（不需要 root；只丢干净页）
  python3 - "$@" <<'PY'
import os, sys
for f in sys.argv[1:]:
    try:
        fd = os.open(f, os.O_RDONLY); os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED); os.close(fd)
    except OSError:
        pass
PY
}

# ---- 离线：prefill / decode / spec / KLD ------------------------------------------
step_pp() {  # 格式 标签 长度 GEN
  local fmt=$1 tag=$2 len=$3 gen=$4 label=bench_$1_$2 rc all='' last='' n='' d
  note "$fmt：prefill $len（GEN=$gen）"
  env LAUNCHER="$(launcher "$fmt")" GEN="$gen" bash tools/pp_prod.sh "$len" "$label" >"logs/$label.out" 2>&1; rc=$?
  svm_check
  local log=logs/$label.log
  if (( rc )) || crashed "$log"; then tail -n 5 "logs/$label.out"; bad "$fmt $tag：rc=$rc（$log）"; return; fi
  read -r all last n < <(tr '\r' '\n' <"$log" | grep -ao 'prefill: [0-9]* tokens in [0-9.]* s = [0-9.]* tok/s' |
    awk '{n += $2; s += $5; l = $8} END {if (n) printf "%.0f %.0f %d\n", n / s, l, n}')
  [[ -n $all ]] || { bad "$fmt $tag：日志里没有 prefill 行（$log）"; return; }
  if [[ $tag == pp8k ]]; then R[$fmt.$tag]=$all; else R[$fmt.$tag]="$all / $last"; fi
  echo "   $n token：整体 $all tok/s，最后一个 chunk $last tok/s"
  if (( gen > 1 )); then
    d=$(tr '\r' '\n' <"$log" | grep -ao 'decode: [0-9]* tokens in [0-9]* ms = [0-9.]* tok/s' | tail -1 | sed 's/.*= \([0-9.]*\) tok.*/\1/')
    if [[ -n $d ]]; then R[$fmt.decode]=$d; echo "   decode $gen token：$d tok/s"; else bad "$fmt：没有 decode 行（$log）"; fi
  fi
}
step_spec() {
  local fmt=$1 label=bench_$1_spec rc v
  note "$fmt：MTP 投机，8K prompt + 256 token（gamma 3）"
  env LAUNCHER="$(launcher "$fmt")" SPEC=256 GAMMA=3 bash tools/pp_prod.sh 8k "$label" >"logs/$label.out" 2>&1; rc=$?
  svm_check
  local log=logs/$label.log
  if (( rc )) || crashed "$log"; then tail -n 5 "logs/$label.out"; bad "$fmt spec：rc=$rc（$log）"; return; fi
  v=$(tr '\r' '\n' <"$log" | grep -ao 'spec: [0-9]* tokens in [0-9.]* s = [0-9.]* tok/s | rounds=[0-9]* commit/round=[0-9.]*' |
    tail -1 | sed 's/.*= \([0-9.]*\) tok\/s.*commit\/round=\([0-9.]*\).*/\1 \2/')
  [[ -n $v ]] || { bad "$fmt spec：日志里没有 spec 行（$log）"; return; }
  R[$fmt.spec]="${v% *}（${v#* }）"; echo "   ${v% *} tok/s，commit/round ${v#* }"
}
step_kld() {
  local fmt=$1 out=logs/bench_$1_kld.out rc v k t p
  note "$fmt：KLD（$KREF）"
  [[ -f $KREF ]] || { bad "找不到 $KREF"; return; }
  env LAUNCHER="$(launcher "$fmt")" bash tools/kld_engine.sh "$KREF" "bench_$fmt" >"$out" 2>&1; rc=$?
  svm_check
  v=$(grep -a '^kld_summary' "$out" | tail -1)
  [[ $rc == 0 && -n $v ]] || { tail -n 5 "$out"; bad "$fmt KLD 失败（$out）"; return; }
  k=$(sed 's/.*mean_kld=\([0-9.]*\).*/\1/' <<<"$v"); t=$(sed 's/.*same_top=\([0-9.]*\).*/\1/' <<<"$v")
  p=$(sed 's/.*ppl_q=\([0-9.]*\).*/\1/' <<<"$v")
  R[$fmt.kld]="$k / $t% / $p"; echo "   mean KLD $k，top1 $t%，PPL $p"
}

# ---- API 端到端 -------------------------------------------------------------------
client() {  # 模式 max_tokens 参数... -> stdout
  python3 - "$API_PORT" "$@" <<'PY'
import json, sys, time, threading, urllib.request
port, mode, max_tokens, args = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4:]
URL = f"http://127.0.0.1:{port}/v1/chat/completions"
def ask(content):
    body = json.dumps({"messages": [{"role": "user", "content": content}],
                       "max_tokens": max_tokens, "temperature": 0}).encode()
    req = urllib.request.Request(URL, body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1200) as r:
        return json.load(r)["usage"]["completion_tokens"]
if mode == "one":
    print(ask(args[0]))
elif mode == "doc":   # doc max <json> <k>：取 text 的最后 1/k
    text = json.load(open(args[0]))["text"]
    k = int(args[1])
    print(ask(text[len(text) - len(text) // k:] + "\n\n请用一句话概括上面这段文档讲了什么。"))
elif mode == "conc":  # conc max 题目...：每个题目一个并发请求
    res, t0 = [], time.time()
    th = [threading.Thread(target=lambda q=q: res.append(ask(q))) for q in args]
    for t in th: t.start()
    for t in th: t.join()
    print(sum(res), f"{time.time() - t0:.2f}", len(res))
PY
}
kv() { grep -o "$1=[0-9]*" <<<"$2" | head -1 | cut -d= -f2; }
api_req() {  # 格式 键 说明 模式 max 参数...：发一条请求，从 API 日志取这条请求的统计
  local fmt=$1 key=$2 what=$3; shift 3
  local got line n p g ttft pf dc acc
  got=$(client "$@" 2>&1) || { bad "$fmt API $what 请求失败：$(tail -n 2 <<<"$got")"; return; }
  sleep 0.5
  line=$(grep -a '^REQ [0-9]* end ' "$RL/api.log" | tail -1)
  p=$(kv prompt "$line"); g=$(kv gen "$line"); ttft=$(kv ttft "$line"); pf=$(kv prefill "$line"); dc=$(kv decode "$line")
  [[ -n $g && -n $dc ]] || { bad "$fmt API $what：API 日志里没有统计"; return; }
  n=$(grep -a '| decode eval = ' "$RL/engine.log" | tail -1 | sed 's/.*req \([0-9]*\) |.*/\1/')
  acc=$(grep -a "req $n | draft acceptance = " "$RL/engine.log" | tail -1 | sed 's/.*acceptance = \([0-9.]*\).*mean len = \([0-9.]*\).*/\1 \2/')
  if [[ $key == api_8k || $key == api_32k ]]; then
    R[$fmt.$key]="$p tok，TTFT $(awk -v t="$ttft" 'BEGIN{printf "%.1f", t / 1000}') s，$(awk -v a="$p" -v b="$pf" 'BEGIN{printf "%.0f", a * 1000 / b}') tok/s"
    [[ $(kv cached "$line") == 0 ]] || R[$fmt.$key]+="（cached $(kv cached "$line")）"
  else
    R[$fmt.$key]="$(awk -v a="$g" -v b="$dc" 'BEGIN{printf "%.1f", a * 1000 / b}') tok/s（$g tok${acc:+，接受率 ${acc% *}，均长 ${acc#* }}）"
  fi
  echo "   $what：${R[$fmt.$key]}"
}
step_api() {
  local fmt=$1 log=logs/bench_$1_api.log pid t got g w n
  note "$fmt：$(launcher "$fmt") 起服务（生产配置，KVSNAP 关）"
  setsid env KVSNAP_MAX_GB=0 bash "$(launcher "$fmt")" >"$log" 2>&1 </dev/null &
  pid=$!
  for ((t = 0; t < 600; t++)); do
    grep -q '服务已就绪' "$log" && break
    kill -0 $pid 2>/dev/null || break
    sleep 1
  done
  RL=$(grep -o 'logs/[0-9]\{8\}-[0-9]\{6\}-[0-9]*' "$log" | head -1)
  if grep -q '服务已就绪' "$log"; then
    R[$fmt.ready]="$t s"
    R[$fmt.mem]=$(grep -a 'memory\[slots\]' "$RL/engine.log" | tail -1 |
      sed 's/.*device \([0-9.]*\) GiB.*process RSS \([0-9.]*\) GiB.*/\1 \/ \2/')
    echo "   就绪 $t s，内存 ${R[$fmt.mem]}，日志 $RL"
    client one 16 '你好' >/dev/null 2>&1   # 预热
    api_req "$fmt" api_prose "中文散文" one 512 '请写一篇约 600 字的中文散文，题目是《秋天的长江》。'
    api_req "$fmt" api_code "代码" one 512 '用 Python 实现一个线程安全的 LRU 缓存类（支持 get/put 和容量上限），并给出 pytest 单元测试。'
    api_req "$fmt" api_8k "8K prompt" doc 64 "$DOC" 4
    api_req "$fmt" api_32k "32K prompt" doc 64 "$DOC" 1
    if got=$(client conc 256 '请介绍一下北京的历史，约 300 字。' '请介绍一下上海的历史，约 300 字。' \
               '请介绍一下西安的历史，约 300 字。' '请介绍一下成都的历史，约 300 字。' 2>&1); then
      read -r g w n <<<"$got"
      R[$fmt.api_conc]="总 $(awk -v a="$g" -v b="$w" 'BEGIN{printf "%.1f", a / b}') tok/s（$n 路共 $g tok / ${w} s）"
      echo "   4 路并发：${R[$fmt.api_conc]}"
    else bad "$fmt API 并发请求失败：$(tail -n 2 <<<"$got")"; fi
  else
    tail -n 20 "$log"; bad "$fmt：服务没有就绪（$log）"
  fi
  kill -TERM $pid 2>/dev/null
  for ((t = 0; t < 60; t++)); do kill -0 $pid 2>/dev/null || break; sleep 1; done
  kill -0 $pid 2>/dev/null && { bad "$fmt：SIGTERM 后启动器 60s 未退出"; kill -KILL -- -$pid 2>/dev/null; }
  sleep 2
  pgrep -af "$GPU_RE" >/dev/null && bad "$fmt：停止后仍有引擎进程：$(pgrep -af "$GPU_RE" | cut -c1-120)"
  svm_check
}

for fmt in $FORMATS; do
  case $fmt in hgn | gguf) ;; *) echo "FORMATS 只能是 hgn / gguf"; echo FAIL; exit 1 ;; esac
  note "==== $fmt ===="
  if [[ $fmt == hgn ]]; then evict models/*.gguf; else evict models/*.hgn; fi
  if want pp; then
    step_pp "$fmt" pp8k 8k 1
    step_pp "$fmt" pp32k 32k 128
    step_pp "$fmt" pp64k 64k 1
  fi
  want spec && step_spec "$fmt"
  want kld && step_kld "$fmt"
  if want api; then
    if [[ -f $DOC ]]; then step_api "$fmt"; else bad "找不到 $DOC，跳过 API"; fi
  fi
done

echo
summary
echo
if (( fail )); then echo FAIL; exit 1; else echo PASS; fi
