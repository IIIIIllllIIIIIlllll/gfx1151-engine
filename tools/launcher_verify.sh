#!/usr/bin/env bash
# 启动器一键验证：start_hgn.sh / start_gguf.sh（Linux）。最后打印 PASS / FAIL。
#   bash tools/launcher_verify.sh
#     GGDIR=<GGUF 目录>   默认 models/（有第 1 个分片时），否则 ~/App/llama.cpp/models/Qwen3.8-Flash-Next-UD-Q4_K_XL
#     BIN=build/gdec      端到端用的引擎二进制（需支持纯 GGUF）
#     E2E=0               只做 --check 类检查，不起服务
# 检查项：
#   1. start_hgn.sh --check：命令行 = gdec w4b overlay mtp --serve ... --vision-tower vision，无 GDEC_GGUF*
#   2. start_gguf.sh --check：命令行 = gdec 第 1 个分片 --serve ... --vision-tower mmproj，
#      ENV 比 hgn 只多 GDEC_GGUF_MTP=<sidecar>；MTP/视觉置空时 GDEC_GGUF_MTP= 且无 --vision-tower
#   3. 缺文件：MODEL_DIR 指向空目录时两个启动器都退出 1 并列出全部缺失文件；少一个分片时只报那一个
#   4. 格式互斥：hgn 启动器拒绝 .gguf、GGUF 启动器拒绝 .hgn；外部残留 GDEC_GGUF* 被清掉；start.sh 只提示
#   5. 端到端（E2E=1）：临时根目录（build/gdec -> BIN，其余软链）里真正启动 hgn / GGUF / GGUF 无 MTP 无视觉，
#      等"服务已就绪"，发一条 chat 请求，停止后引擎进程与端口都已释放（后台进程的 SIGINT 被 bash 忽略，
#      用 SIGTERM，与 Ctrl+C 走同一个 cleanup）
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
ROOT=$PWD
unset MODEL_DIR MODEL_FILE OVERLAY_FILE MTP_FILE VISION_FILE TOKENIZER_DIR GGUF_FILE GGUF_MTP_FILE GGUF_VISION_FILE
for e in $(compgen -e | grep '^GDEC_'); do unset "$e"; done
SHARD=Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf
if [[ -z ${GGDIR:-} ]]; then
  GGDIR=models; [[ -f models/$SHARD ]] || GGDIR=$HOME/App/llama.cpp/models/Qwen3.8-Flash-Next-UD-Q4_K_XL
fi
GG=$GGDIR/$SHARD
MTP=$GGDIR/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf
MMPROJ=$GGDIR/mmproj-BF16.gguf
GGENV=(GGUF_FILE="$GG" GGUF_MTP_FILE="$MTP" GGUF_VISION_FILE="$MMPROJ")
BIN=${BIN:-build/gdec}
W=$(mktemp -d /tmp/launcher_verify.XXXXXX)
fail=0
note() { echo "== $*"; }
bad() { echo "FAIL: $*"; fail=1; }
ok() { echo "OK  $*"; }

for f in models/qwen38-flash-next-w4b.hgn "$GG" "$MTP" "$MMPROJ" models/tokenizer/tokenizer.json "$BIN"; do
  [[ -e "$f" ]] || { echo "缺少 $f"; echo FAIL; exit 1; }
done

check() {  # 标签 启动器 [K=V ...] -> $W/标签.out，返回启动器退出码
  local label=$1 launcher=$2; shift 2
  env "$@" bash "$launcher" --check >"$W/$label.out" 2>&1
}
cmd_of() { sed -n 's/^CMD //p' "$W/$1.out"; }

# ---- 1. hgn --check ----------------------------------------------------------
note "1. start_hgn.sh --check"
if check hgn start_hgn.sh; then
  eval "C=($(cmd_of hgn))"
  want=("$ROOT/build/gdec" ./models/qwen38-flash-next-w4b.hgn ./models/qwen38-flash-next-w4b.overlay.hgn
        ./models/qwen38-flash-next-mtp.hgn --serve --port 8730 --maxctx 262144
        --vision-tower ./models/qwen38-flash-next-vision.hgn)
  [[ "${C[*]}" == "${want[*]}" ]] && ok "CMD ${C[*]:1}" || bad "hgn CMD = ${C[*]}"
  grep -q '^ENV GDEC_GGUF' "$W/hgn.out" && bad "hgn ENV 里有 GDEC_GGUF*" || ok "无 GDEC_GGUF*"
  grep -qx 'ENV GDEC_KV_PAGED=1' "$W/hgn.out" && grep -qx 'ENV GDEC_PREFILL_CHUNK=16384' "$W/hgn.out" &&
    ok "生产 ENV（KV_PAGED、PREFILL_CHUNK …）" || bad "hgn 缺生产 ENV"
else cat "$W/hgn.out"; bad "start_hgn.sh --check 失败"; fi

# ---- 2. gguf --check ---------------------------------------------------------
note "2. start_gguf.sh --check（GGDIR=$GGDIR）"
if check gguf start_gguf.sh "${GGENV[@]}"; then
  eval "C=($(cmd_of gguf))"
  want=("$ROOT/build/gdec" "$GG" --serve --port 8730 --maxctx 262144 --vision-tower "$MMPROJ")
  [[ "${C[*]}" == "${want[*]}" ]] && ok "CMD ${C[*]:1}" || bad "gguf CMD = ${C[*]}"
  d=$(diff <(grep '^ENV' "$W/hgn.out") <(grep '^ENV' "$W/gguf.out") | grep '^[<>]')
  [[ "$d" == "> ENV GDEC_GGUF_MTP=$MTP" ]] && ok "ENV = hgn + GDEC_GGUF_MTP" || bad "ENV 差异：$d"
else cat "$W/gguf.out"; bad "start_gguf.sh --check 失败"; fi
if check gguf_nomtp start_gguf.sh GGUF_FILE="$GG" GGUF_MTP_FILE= GGUF_VISION_FILE=; then
  eval "C=($(cmd_of gguf_nomtp))"
  grep -qx 'ENV GDEC_GGUF_MTP=' "$W/gguf_nomtp.out" && [[ "${C[*]}" != *--vision-tower* ]] &&
    ok "MTP/视觉置空：GDEC_GGUF_MTP= 且无 --vision-tower" || bad "MTP/视觉置空时输出不对"
else cat "$W/gguf_nomtp.out"; bad "GGUF 无 MTP 无视觉 --check 失败"; fi

# ---- 3. 缺文件 ---------------------------------------------------------------
note "3. 缺文件时报错退出"
mkdir -p "$W/empty"
check miss_hgn start_hgn.sh MODEL_DIR="$W/empty"; rc=$?
n=$(grep -c "^  .*：$W/empty/" "$W/miss_hgn.out")
[[ $rc == 1 && $n == 5 ]] && ok "hgn：退出 1，列出 $n 个缺失文件" || { cat "$W/miss_hgn.out"; bad "hgn 缺文件 rc=$rc 列出 $n 个（应为 1 / 5）"; }
sed -n '1,7p' "$W/miss_hgn.out" | sed 's/^/    /'
check miss_gguf start_gguf.sh MODEL_DIR="$W/empty"; rc=$?
n=$(grep -c "^  .*：$W/empty/" "$W/miss_gguf.out")
[[ $rc == 1 && $n == 7 ]] && ok "GGUF：退出 1，列出 $n 个缺失文件" || { cat "$W/miss_gguf.out"; bad "GGUF 缺文件 rc=$rc 列出 $n 个（应为 1 / 7）"; }
sed -n '1,9p' "$W/miss_gguf.out" | sed 's/^/    /'
mkdir -p "$W/part"
for f in "$GGDIR"/Qwen3.8-Flash-Next-UD-Q4_K_XL-0000[124]-of-00004.gguf "$MTP" "$MMPROJ"; do ln -s "$(realpath "$f")" "$W/part/"; done
check miss_shard start_gguf.sh MODEL_DIR="$W/part" TOKENIZER_DIR=./models/tokenizer; rc=$?
[[ $rc == 1 && $(grep -c '^  ' "$W/miss_shard.out") == 1 ]] && grep -q '^  分片 3/4：.*-00003-of-00004.gguf' "$W/miss_shard.out" &&
  ok "少第 3 个分片：只报 分片 3/4" || { cat "$W/miss_shard.out"; bad "缺分片检测不对（rc=$rc）"; }

# ---- 4. 格式互斥 / 残留变量 / start.sh --------------------------------------------
note "4. 格式互斥、残留 GDEC_GGUF*、start.sh 提示"
check x_hgn start_hgn.sh MODEL_FILE="$GG"; rc=$?
[[ $rc == 1 ]] && grep -q 'start_gguf.sh' "$W/x_hgn.out" && ok "hgn 启动器拒绝 .gguf" || { cat "$W/x_hgn.out"; bad "hgn 启动器没拒绝 .gguf"; }
check x_gguf start_gguf.sh GGUF_FILE=models/qwen38-flash-next-w4b.hgn; rc=$?
[[ $rc == 1 ]] && grep -q 'start_hgn.sh' "$W/x_gguf.out" && ok "GGUF 启动器拒绝 .hgn" || { cat "$W/x_gguf.out"; bad "GGUF 启动器没拒绝 .hgn"; }
check stale start_hgn.sh GDEC_GGUF="$GG" GDEC_GGUF_DENSE=1 GDEC_GGUF_PLE=1
grep -q '^ENV GDEC_GGUF' "$W/stale.out" && bad "残留的 GDEC_GGUF* 进了 hgn 的 ENV" || ok "残留 GDEC_GGUF* 被清掉"
bash start.sh --check >"$W/old.out" 2>&1; rc=$?
[[ $rc == 1 ]] && grep -q start_hgn.sh "$W/old.out" && grep -q start_gguf.sh "$W/old.out" &&
  ok "start.sh 退出 1 并指向两个启动器" || { cat "$W/old.out"; bad "start.sh 提示不对"; }

# ---- 5. 端到端 ---------------------------------------------------------------
ENGINE_RE='(^|/)gdec[^/[:space:]]*([[:space:]]|$)'
evict() {  # 把另一种格式的权重逐出 page cache（不需要 root；只丢干净页），减少加载时的回收与碎片
  python3 - "$@" <<'PY'
import os, sys
for f in sys.argv[1:]:
    try:
        fd = os.open(f, os.O_RDONLY); os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED); os.close(fd)
    except OSError:
        pass
PY
}
e2e() {  # 标签 启动器 [K=V ...]
  local label=$1 launcher=$2; shift 2
  local R=$W/root_$label log=$W/e2e_$label.out pid t resp since
  if pgrep -af "$ENGINE_RE" >/dev/null; then
    bad "$label：跳过，上一个引擎进程仍未退出：$(pgrep -af "$ENGINE_RE" | cut -c1-120)"; return
  fi
  if [[ $launcher == start_hgn.sh ]]; then evict "$GGDIR"/*.gguf; else evict models/*.hgn; fi
  since=$(date '+%Y-%m-%d %H:%M:%S')
  mkdir -p "$R/build" "$R/data"
  cp start_hgn.sh start_gguf.sh service.conf "$R/"
  ln -s "$ROOT/tools" "$ROOT/models" "$R/"
  for f in "$ROOT"/build/*; do ln -s "$f" "$R/build/"; done
  rm -f "$R/build/gdec"; ln -s "$(realpath "$BIN")" "$R/build/gdec"
  setsid env "$@" KVSNAP_MAX_GB=0 MAX_CONTEXT=32768 bash "$R/$launcher" >"$log" 2>&1 < /dev/null &
  pid=$!
  for ((t = 0; t < 420; t++)); do
    grep -q '服务已就绪' "$log" && break
    kill -0 $pid 2>/dev/null || break
    sleep 1
  done
  if grep -q '服务已就绪' "$log"; then
    resp=$(curl -s -m 120 http://127.0.0.1:8731/v1/chat/completions -H 'Content-Type: application/json' \
      -d '{"messages":[{"role":"user","content":"1+1 等于几？只回答数字。"}],"max_tokens":64,"temperature":0}')
    if grep -q '"choices"' <<<"$resp"; then
      ok "$label：就绪用时 ${t}s，回复 $(sed -n 's/.*"content":"\([^"]*\)".*/\1/p' <<<"$resp" | tr -d '\n' | cut -c1-60)"
    else bad "$label：chat 请求失败：$(cut -c1-200 <<<"$resp")"; fi
  else
    tail -n 20 "$log"; bad "$label：服务没有就绪（$log）"
  fi
  kill -TERM $pid 2>/dev/null
  for ((t = 0; t < 40; t++)); do kill -0 $pid 2>/dev/null || break; sleep 1; done
  kill -0 $pid 2>/dev/null && { bad "$label：SIGTERM 后启动器 40s 未退出"; kill -KILL -- -$pid 2>/dev/null; }
  sleep 2
  if pgrep -af "$ENGINE_RE" >/dev/null || [[ -n "$(ss -H -ltn 'sport = :8730 or sport = :8731')" ]]; then
    bad "$label：停止后仍有引擎进程或端口占用"; pgrep -af "$ENGINE_RE" | cut -c1-160
  else ok "$label：停止后进程与端口已释放"; fi
  if journalctl -k --since "$since" --no-pager 2>/dev/null | grep -q svm_range_cpu_invalidate_pagetables; then
    bad "$label：内核 amdgpu SVM 死锁（journalctl -k 有 svm_range_cpu_invalidate_pagetables），引擎进程无法结束，需重启机器"
  fi
  grep -E '^(权重|错误)' "$log" | sed 's/^/    /'
  grep -h '^gguf: pure GGUF start' "$R"/logs/*/engine.log 2>/dev/null | cut -c1-160 | sed 's/^/    /'
}
if [[ ${E2E:-1} == 1 ]]; then
  note "5. 端到端（BIN=$BIN，MAX_CONTEXT=32768，端口 8730/8731）"
  if pgrep -af '(^|/)(gdec[^/[:space:]]*|flash_serve|serve_api\.py|llama-server|llama-perplexity|llama-cli)([[:space:]]|$)' >/dev/null; then
    bad "GPU 上已有引擎/llama.cpp 在跑，跳过端到端"
  else
    e2e hgn start_hgn.sh
    e2e gguf start_gguf.sh "${GGENV[@]}"
    e2e gguf_nomtp start_gguf.sh GGUF_FILE="$GG" GGUF_MTP_FILE= GGUF_VISION_FILE=
  fi
fi

echo
echo "输出目录：$W"
if (( fail )); then echo FAIL; exit 1; else echo PASS; fi
