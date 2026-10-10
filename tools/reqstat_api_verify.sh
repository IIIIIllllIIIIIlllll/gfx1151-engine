#!/usr/bin/env bash
# reqstat_api_verify.sh — reqstat 记录 + C++ 查询端点的端到端验证。
#
# 1) 编译并运行 tools/reqstat_read_test（记录器/读取器全链路断言套件）。
# 2) 若本机有 API 二进制（build/qwenox-api 或 build/qwenox-win.exe）和
#    tokenizer（TOKENIZER_DIR，默认 ./models/tokenizer），用测试固件起真实
#    API 进程，curl 校验 GET /reqstat/summary 与 /reqstat/tail；缺任一项
#    则跳过 HTTP 段（SKIP），不影响第 1 段结论。
#
# 用法: bash tools/reqstat_api_verify.sh
# 结尾打印 PASS / FAIL（SKIP 不算失败）。
set -uo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

TMP="$(mktemp -d)"
API_PID=''
cleanup() {
  [[ -z "$API_PID" ]] || { kill "$API_PID" 2>/dev/null; taskkill //PID "$API_PID" //F //T >/dev/null 2>&1; }
  rm -rf "$TMP"
}
trap cleanup EXIT

FAIL=0
say()  { echo "== $*"; }
fail() { echo "FAIL $*"; FAIL=1; }

# ---- 选编译器与 API 二进制 ------------------------------------------------
CXX_BIN=''
for c in "${CXX:-}" c++ g++ clang++; do
  [[ -n "$c" ]] && command -v "$c" >/dev/null 2>&1 && { CXX_BIN="$c"; break; }
done
if [[ -z "$CXX_BIN" ]]; then
  for c in "${THEROCK:-}/lib/llvm/bin/clang++.exe" /c/TheRock/*/lib/llvm/bin/clang++.exe; do
    [[ -x "$c" ]] && { CXX_BIN="$c"; break; }
  done
fi
[[ -n "$CXX_BIN" ]] || { echo "找不到 C++ 编译器（设 CXX=...）" >&2; exit 1; }

API=''
[[ -x build/qwenox-api ]] && API=build/qwenox-api
[[ -x build/qwenox-win.exe ]] && API=build/qwenox-win.exe

# ---- 1) C++ 断言套件 ------------------------------------------------------
say "编译 reqstat_read_test（$CXX_BIN）"
TESTBIN=build/reqstat_read_test
[[ "$OS" == Windows_NT || -n "${WINDIR:-}" ]] && TESTBIN=build/reqstat_read_test.exe
"$CXX_BIN" -O2 -std=c++17 -D_CRT_SECURE_NO_WARNINGS -Isrc/api \
  tools/reqstat_read_test.cpp src/api/reqstat.cpp src/api/reqstat_read.cpp \
  -o "$TESTBIN" || { echo "FAIL 编译失败"; exit 1; }
# Windows 上刚链接出的 exe 偶尔被 Defender/索引器短暂占用，exec 会拿到 127：重试几次。
rc=''
for _ in 1 2 3 4 5; do
  "$TESTBIN" && { rc=0; break; }
  rc=$?
  (( rc == 127 )) || break
  sleep 1
done
[[ "$rc" == 0 ]] || fail "reqstat_read_test"

# ---- 2) HTTP 端到端 --------------------------------------------------------
TOK="${TOKENIZER_DIR:-./models/tokenizer}"
if [[ -z "$API" || ! -r "$TOK/tokenizer.json" ]]; then
  say "SKIP HTTP 段：$([[ -z "$API" ]] && echo '缺 API 二进制（先 bash build.sh api / build_win.sh api）' || echo "缺 tokenizer：$TOK（设 TOKENIZER_DIR=...）")"
else
  say "固件 + 真实 API（$API）"
  "$TESTBIN" --fixtures "$TMP/fix" >/dev/null || fail "写固件失败"
  PORT=18731
  QWENOX_REQSTAT_DIR="$TMP/fix" "$API" --tokenizer "$TOK" --engine 127.0.0.1:9 \
    --host 127.0.0.1 --port "$PORT" >"$TMP/api.log" 2>&1 &
  API_PID=$!
  ok=0
  for _ in $(seq 1 60); do
    grep -q 'listening' "$TMP/api.log" 2>/dev/null && { ok=1; break; }
    kill -0 "$API_PID" 2>/dev/null || break
    sleep 0.5
  done
  if (( ! ok )); then
    tail -n 10 "$TMP/api.log" >&2
    fail "API 启动失败"
  else
    S="$(curl -sf "http://127.0.0.1:$PORT/reqstat/summary")" || fail "GET /reqstat/summary"
    S2="$(curl -sf "http://127.0.0.1:$PORT/reqstat/summary?from=2026-09-29&to=2026-09-29")" || fail "GET summary 带窗口"
    T="$(curl -sf "http://127.0.0.1:$PORT/reqstat/tail?n=3")" || fail "GET /reqstat/tail"
    check() { # check <label> <haystack> <pattern>
      if [[ "$2" == *"$3"* ]]; then echo "ok   $1"; else echo "FAIL $1：缺 $3"; echo "$2" | head -c 400 >&2; FAIL=1; fi
    }
    check "summary 全量 requests=200" "$S" '"requests":200'
    check "summary 输入 token 求和" "$S" '"input_tokens":219900'
    check "summary chain 接受率 0.375" "$S" '"acceptance":0.375'
    check "summary decode tok/s" "$S" '"decode_tok_per_s":10'
    # prefill 速度只算真正处理的 token：219900 - Σ(i%7)=594 -> 219306 / 400s
    check "summary prefill tok/s" "$S" '"prefill_tok_per_s":548.265'
    # 分起草器速度：chain 与 serial 桶都是 50 tok / 5 s = 10 tok/s；
    # serial 桶 proposed=0 也必须出现（它正是拖低总速度的那类）。
    check "summary serial 桶" "$S" '"serial":{"requests":40,"accepted":0,"proposed":0,"acceptance":null'
    n_dps="$(printf '%s' "$S" | grep -o '"decode_tok_per_s":10' | wc -l)"
    [[ "$n_dps" == 3 ]] && echo "ok   分起草器 decode tok/s（总+chain+serial）" || fail "decode_tok_per_s 出现 $n_dps 次（期望 3）"
    check "summary 文件数与坏记录" "$S" '"files_total":2'
    check "summary bad_crc=1" "$S" '"bad_crc":1'
    check "summary 窗口 requests=100" "$S2" '"requests":100'
    check "tail n=3 最新 req_seq=200" "$T" '"req_seq":200'
    n_seq="$(printf '%s' "$T" | grep -o '"req_seq"' | wc -l)"
    [[ "$n_seq" == 3 ]] && echo "ok   tail n=3 恰 3 条" || fail "tail n=3 条数=$n_seq"
    check "tail 接受率字段" "$T" '"acceptance":0.375'
  fi
fi

echo
if (( FAIL )); then echo FAIL; exit 1; fi
echo PASS
