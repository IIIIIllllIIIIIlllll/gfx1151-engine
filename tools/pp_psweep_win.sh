#!/usr/bin/env bash
# pp_psweep_win.sh — M1 小 P 扫描（HANDOFF-PREFILL-ALL.md §2 M1）
# 固定 maxctx 139264，对 P ∈ {256 512 1024 2048 4096 16384}：prompt = 3×P（取
# data/qsa-oracle/131072.tokens 前缀），PREFILL_CHUNK=P，剔除预热 dummy 与第 1 个
# chunk（PLE 冷读），取第 2、3 个 chunk 均值。每档开 QWENOX_KPROF=1 + QWENOX_PROF=1。
# 引擎参数是启动期的，只能一档一进程。v2 模型（MODEL_FILE/NGRAM_FILE 可覆盖）。
# 用法: bash tools/pp_psweep_win.sh [P ...]   （默认六档全跑）
#
# MODE=ttft（§11.2 M1b）：单 chunk TTFT 基线。走 --serve + qwenox-api 同进程多请求：
# 每档每 rep 先发 64 token dummy 暖 PLE（一次性表加载，不计时），再发正式 prompt
# （yarn_ladder filler，rung 唯一化防 TokenCache 前缀命中），只计正式请求墙钟
# （max_tokens=1）。每档 3 次取中位数。引擎开 KPROF，按 kprof 行的 P 匹配正式请求
# 报 qsa:flash 等分段。PREFILL_CHUNK=16384（生产口径），--maxctx 17408。
# 用法: MODE=ttft MODEL_FILE=models/qwen38-flash-next-v2.hgn \
#       NGRAM_FILE=models/qwen38-flash-next-ngram.hgn bash tools/pp_psweep_win.sh
set -u
cd "$(dirname "$0")/.."

TOKENS_SRC=data/qsa-oracle/131072.tokens
MAXCTX=139264
PS="$@"
[ -z "$PS" ] && PS="256 512 1024 2048 4096 16384"

MODEL_ARGS=("${MODEL_FILE:-models/qwen38-flash-next-w4b.hgn}")
if [[ -z "${MODEL_FILE:-}" ]]; then
    MODEL_ARGS+=("models/qwen38-flash-next-w4b.overlay.hgn")
elif [[ -n "${NGRAM_FILE:-}" ]]; then
    MODEL_ARGS+=("$NGRAM_FILE")
fi

# ---- MODE=incr：M1b 增量档（§11.2）：base≈32K 多轮续写 ----
# 真实多轮形态：请求 1 = user(base≈32K) max_tokens=1，取回生成文本 r1；
# 请求 2 = [user(base), assistant(r1), user(新增 P)] —— 请求 1 的完整 prompt+生成
# 是请求 2 的严格前缀 → 引擎 live-mcp cont 路径命中，只 prefill 新增段。
# 每档 rep0 在 base 后紧跟（live 路径），rep1 同 prompt 重发（快照路径参照）。
# 记录请求 2 的 wall / usage.cached_tokens。--maxctx 49152。
# 用法: MODE=incr TAG=w4b INCR_SIZES="256 1024 4096" bash tools/pp_psweep_win.sh
if [[ "${MODE:-}" == "incr" ]]; then
  TTAG="${TAG:-w4b}"
  SIZES="${INCR_SIZES:-256 1024 4096}"
  IBASE="${INCR_BASE:-32768}"
  IMAXCTX=49152
  ENGINE_LOG="logs/incr_engine_${TTAG}.log"
  API_LOG="logs/incr_api_${TTAG}.log"
  mkdir -p logs
  env QWENOX_KPROF=1 QWENOX_PROF=1 \
      QWENOX_QSA_KV_BF16=1 QWENOX_QSA_WMMA=1 QWENOX_QSA_WMMA_BTV=1 \
      QWENOX_MOE_LT=1 QWENOX_MOE_LT_BF16=1 QWENOX_GR_BF16=1 \
      QWENOX_GDN_STREAM=1 QWENOX_GDN_WAVE=1 \
      QWENOX_PREFILL_CHUNK=16384 QWENOX_GEMM_WMMA=1 QWENOX_GDN_FUSED=1 \
      QWENOX_INDEX_FUSED2=1 QWENOX_PP_MOE_OUT=1 QWENOX_INDEX_STREAM_SELECT=1 \
      build/qwenox-engine-win "${MODEL_ARGS[@]}" \
      --serve --host 127.0.0.1 --port 8730 --maxctx $IMAXCTX >"$ENGINE_LOG" 2>&1 &
  EPID=$!
  incr_cleanup() {
    kill $APID $EPID 2>/dev/null
    taskkill //PID $APID //F //T >/dev/null 2>&1
    taskkill //PID $EPID //F //T >/dev/null 2>&1
  }
  APID=''
  trap incr_cleanup EXIT
  begin=$SECONDS
  until grep -q 'serve: listening' "$ENGINE_LOG" 2>/dev/null; do
    kill -0 $EPID 2>/dev/null || { tail -n 15 "$ENGINE_LOG" >&2; exit 1; }
    (( SECONDS - begin < 300 )) || { echo "引擎启动超时" >&2; exit 1; }
    sleep 2
  done
  build/qwenox-win.exe --tokenizer models/tokenizer \
      --engine 127.0.0.1:8730 --host 127.0.0.1 \
      --port 8731 --context $IMAXCTX >"$API_LOG" 2>&1 &
  APID=$!
  begin=$SECONDS
  until netstat -an | grep -E '[:.]8731\s+.*LISTENING' >/dev/null 2>&1; do
    kill -0 $APID 2>/dev/null || { tail -n 15 "$API_LOG" >&2; exit 1; }
    (( SECONDS - begin < 30 )) || { echo "API 启动超时" >&2; exit 1; }
    sleep 1
  done
  SIZES="$SIZES" IBASE="$IBASE" TTAG="$TTAG" python - <<'EOF'
import json, os, statistics, sys, time, urllib.request
sys.path.insert(0, "tools")
from yarn_ladder import build_prompt, chat

base = "http://127.0.0.1:8731"
sizes = [int(x) for x in os.environ["SIZES"].split()]
ibase = int(os.environ["IBASE"])
tag = os.environ["TTAG"]

def chat_msgs(msgs, max_tokens):
    body = {"model": "qwenox", "messages": msgs, "temperature": 0.0,
            "max_tokens": max_tokens, "stream": False}
    req = urllib.request.Request(
        base + "/v1/chat/completions", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=7200) as resp:
        out = json.loads(resp.read())
    return out, time.time() - t0

def usage_of(out):
    u = out.get("usage") or {}
    return (u.get("prompt_tokens") or 0,
            (u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)

warm, _ = build_prompt(f"warm-incr-{tag}", 64, [], "w")
chat(base, warm, 1)

btext, _ = build_prompt(f"incrbase-{tag}", ibase, [], "i")
print("\n==== M1b 增量档（多轮: base≈32K + 新增 P, max_tokens=1）====")
print("\t".join(["P", "rep", "tokens", "cached", "new", "TTFT ms", "new-tok/s"]))
for p in sizes:
    dtext, _ = build_prompt(f"incrdelta-{tag}-{p}", p, [], "d")
    for r in range(2):
        # 请求 1：base（重建 live 前缀），max_tokens=1 拿生成文本
        out1, dt1 = chat_msgs([{"role": "user", "content": btext}], 1)
        pt1, ct1 = usage_of(out1)
        r1 = ((out1.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
        if r == 0:
            print(f"base: tokens={pt1} cached={ct1} wall={dt1*1000:.0f} ms "
                  f"({pt1/dt1:.0f} tok/s) reply={r1!r:.20}", flush=True)
        # 请求 2：多轮续写，新增 P
        msgs = [{"role": "user", "content": btext},
                {"role": "assistant", "content": r1},
                {"role": "user", "content": dtext}]
        out2, dt2 = chat_msgs(msgs, 1)
        pt2, ct2 = usage_of(out2)
        new = pt2 - ct2
        print("\t".join([str(p), str(r), str(pt2), str(ct2), str(new),
                         f"{dt2*1000:.0f}", f"{new/dt2:.0f}"]), flush=True)
EOF
  rc=$?
  exit $rc
fi

# ---- MODE=ttft：M1b 单 chunk TTFT（§11.2）----
if [[ "${MODE:-}" == "ttft" ]]; then
  TTAG="${TAG:-v2}"
  SIZES="${TTFT_SIZES:-512 1024 2048 3000 8192}"
  REPS="${TTFT_REPS:-3}"
  TMAXCTX=17408
  ENGINE_LOG="logs/ttft_engine_${TTAG}.log"
  API_LOG="logs/ttft_api_${TTAG}.log"
  mkdir -p logs
  env QWENOX_KPROF=1 QWENOX_PROF=1 \
      QWENOX_QSA_KV_BF16=1 QWENOX_QSA_WMMA=1 QWENOX_QSA_WMMA_BTV=1 \
      QWENOX_MOE_LT=1 QWENOX_MOE_LT_BF16=1 QWENOX_GR_BF16=1 \
      QWENOX_GDN_STREAM=1 QWENOX_GDN_WAVE=1 \
      QWENOX_PREFILL_CHUNK=16384 QWENOX_GEMM_WMMA=1 QWENOX_GDN_FUSED=1 \
      QWENOX_INDEX_FUSED2=1 QWENOX_PP_MOE_OUT=1 QWENOX_INDEX_STREAM_SELECT=1 \
      build/qwenox-engine-win "${MODEL_ARGS[@]}" \
      --serve --host 127.0.0.1 --port 8730 --maxctx $TMAXCTX >"$ENGINE_LOG" 2>&1 &
  EPID=$!
  ttft_cleanup() {
    kill $APID $EPID 2>/dev/null
    taskkill //PID $APID //F //T >/dev/null 2>&1
    taskkill //PID $EPID //F //T >/dev/null 2>&1
  }
  APID=''
  trap ttft_cleanup EXIT
  begin=$SECONDS
  until grep -q 'serve: listening' "$ENGINE_LOG" 2>/dev/null; do
    kill -0 $EPID 2>/dev/null || { tail -n 15 "$ENGINE_LOG" >&2; exit 1; }
    (( SECONDS - begin < 300 )) || { echo "引擎启动超时" >&2; exit 1; }
    sleep 2
  done
  build/qwenox-win.exe --tokenizer models/tokenizer \
      --engine 127.0.0.1:8730 --host 127.0.0.1 \
      --port 8731 --context $TMAXCTX >"$API_LOG" 2>&1 &
  APID=$!
  begin=$SECONDS
  until netstat -an | grep -E '[:.]8731\s+.*LISTENING' >/dev/null 2>&1; do
    kill -0 $APID 2>/dev/null || { tail -n 15 "$API_LOG" >&2; exit 1; }
    (( SECONDS - begin < 30 )) || { echo "API 启动超时" >&2; exit 1; }
    sleep 1
  done
  SIZES="$SIZES" REPS="$REPS" TTAG="$TTAG" python - <<'EOF'
import os, re, statistics, sys
sys.path.insert(0, "tools")
from yarn_ladder import build_prompt, chat

base = "http://127.0.0.1:8731"
sizes = [int(x) for x in os.environ["SIZES"].split()]
reps = int(os.environ["REPS"])
tag = os.environ["TTAG"]
segs = ["gdn","qsa","moe:up","moe:down","moe:reduce","moe:shared","hc","ht_deq",
        "gdn:scan","qsa:idx","qsa:flash"]

runs = []  # (size, rep, actual_tokens, wall_s, cached)
for p in sizes:
    for r in range(reps):
        warm, _ = build_prompt(f"warm-{tag}-{p}-{r}", 64, [], "w")
        chat(base, warm, 1)
        prompt, _ = build_prompt(f"ttft-{tag}-{p}-{r}", p, [], "t")
        out, dt = chat(base, prompt, 1)
        usage = out.get("usage") or {}
        pt = usage.get("prompt_tokens") or 0
        ct = (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
        runs.append((p, r, pt, dt, ct))
        print(f"P={p} rep{r}: tokens={pt} cached={ct} wall={dt*1000:.0f} ms "
              f"({pt/dt:.0f} tok/s)", flush=True)

# kprof 行按 P 匹配正式请求（dummy P~64 不会落入 size±80 窗口）
kpl = [l for l in open(f"logs/ttft_engine_{tag}.log", encoding="utf-8",
                       errors="replace") if l.startswith("kprof base=")]
def segs_for(p, lo, hi):
    acc, n, segsum = {}, 0, 0.0
    for l in kpl:
        m = re.search(r"kprof base=0 P=(\d+)", l)
        if not m or not (lo <= int(m.group(1)) <= hi):
            continue
        n += 1
        segsum += float(re.search(r"segsum=([0-9.]+) ms", l).group(1))
        for s in segs:
            m2 = re.search(r"(?:^|\| )\[?%s ([0-9.]+)" % re.escape(s), l)
            if m2: acc[s] = acc.get(s, 0.0) + float(m2.group(1))
    if n:
        segsum /= n
        for s in acc: acc[s] /= n
    return n, segsum, acc

print("\n==== M1b TTFT（PLE 暖，max_tokens=1，中位数）====")
hdr = ["P","tokens","TTFT ms","tok/s","segsum","gap%","qsa:flash","qsa:idx",
       "hc","moeΣ","ht_deq","gdn"]
print("\t".join(hdr))
for p in sizes:
    rs = [r for r in runs if r[0] == p]
    med = statistics.median(r[3] for r in rs)
    pt = max(r[2] for r in rs)
    n, segsum, acc = segs_for(p, int(pt*0.9), int(pt*1.1) + 80)
    moe = sum(acc.get(s, 0.0) for s in ("moe:up","moe:down","moe:reduce","moe:shared"))
    gap = (med*1000 - segsum) / (med*1000) * 100 if med else 0.0
    row = [str(p), str(pt), f"{med*1000:.0f}", f"{pt/med:.0f}",
           f"{segsum:.0f}" if n else "-", f"{gap:.0f}" if n else "-",
           f"{acc.get('qsa:flash',0):.0f}" if n else "-",
           f"{acc.get('qsa:idx',0):.0f}" if n else "-",
           f"{acc.get('hc',0):.0f}" if n else "-",
           f"{moe:.0f}" if n else "-",
           f"{acc.get('ht_deq',0):.0f}" if n else "-",
           f"{acc.get('gdn',0):.0f}" if n else "-"]
    print("\t".join(row))
print("gufo 参照: pp2048@d0 = 1628 tok/s, d4K = 1523 tok/s（GUFO-GAP）")
EOF
  rc=$?
  exit $rc
fi

mkdir -p /tmp/psweep logs

for P in $PS; do
  NTOK=$((3 * P))
  TOK=/tmp/psweep_${NTOK}.tokens
  [ -f "$TOK" ] || head -n "$NTOK" "$TOKENS_SRC" > "$TOK"
  LOG="logs/m1_${TAG:-v2}_p${P}.log"
  env QWENOX_KPROF=1 QWENOX_PROF=1 \
      QWENOX_QSA_KV_BF16=1 QWENOX_QSA_WMMA=1 QWENOX_QSA_WMMA_BTV=1 \
      QWENOX_MOE_LT=1 QWENOX_MOE_LT_BF16=1 QWENOX_GR_BF16=1 \
      QWENOX_GDN_STREAM=1 QWENOX_GDN_WAVE=1 \
      QWENOX_PREFILL_CHUNK=$P QWENOX_GEMM_WMMA=1 QWENOX_GDN_FUSED=1 \
      QWENOX_INDEX_FUSED2=1 QWENOX_PP_MOE_OUT=1 QWENOX_INDEX_STREAM_SELECT=1 \
      QWENOX_KVSNAP=1 QWENOX_KVSNAP_MAX_GB=20 \
      build/qwenox-engine-win "${MODEL_ARGS[@]}" \
      --tokens-file "$TOK" --gen 1 --maxctx $MAXCTX >"$LOG" 2>&1
  rc=$?
  echo "== P=$P rc=$rc log=$LOG =="
done

# ---- 汇总（剔除预热行与 chunk 1，取 chunk 2、3 均值）----
python - $PS <<'EOF'
import re, sys, os
TAG = os.environ.get("TAG", "v2")
segs = ["gdn","qsa","moe:up","moe:down","moe:reduce","moe:shared","hc","ht_deq","gdn:scan","qsa:idx","qsa:flash"]

def parse(p):
    log = open(f"logs/m1_{TAG}_p{p}.log", encoding="utf-8", errors="replace").read()
    pre = re.findall(r"prefill: \d+ tokens in ([0-9.]+) s = ([0-9.]+) tok/s", log)
    kp  = re.findall(r"kprof base=(\d+) P=(\d+) segsum=([0-9.]+) ms[^\n]*", log)
    # 带分段的完整行
    kpl = [l for l in log.splitlines() if l.startswith("kprof base=")]
    prof = re.findall(r"prof: gemm_host=([0-9.]+)ms topk_wait=([0-9.]+)ms bucket\+h2d=([0-9.]+)ms\s+ple_host=([0-9.]+)ms ple_wait=([0-9.]+)ms", log)
    # 预热：prefill 行数 = nchunk+1 时第一行是 dummy
    nchunk = 3
    i = 1 if len(pre) == nchunk + 1 else 0
    wall = [float(t) for t,_ in pre[i+1:i+3]]          # chunk 2,3
    toks = [float(v) for _,v in pre[i+1:i+3]]
    ksel = kpl[i+1:i+3] if len(kpl) >= i+3 else kpl[-2:]
    psel = prof[i+1:i+3] if len(prof) >= i+3 else prof[-2:]
    seg_ms = {}
    segsum = 0.0
    for l in ksel:
        m = re.search(r"segsum=([0-9.]+) ms", l); segsum += float(m.group(1))
        for s in segs:
            m = re.search(r"(?:^|\| )\[?%s ([0-9.]+)" % re.escape(s), l)
            if m: seg_ms[s] = seg_ms.get(s, 0.0) + float(m.group(1))
    n = max(len(ksel), 1)
    segsum /= n
    for s in seg_ms: seg_ms[s] /= n
    wall_ms = sum(wall)/len(wall)*1000 if wall else 0.0
    tokps = sum(toks)/len(toks) if toks else 0.0
    ple = sum(float(x[4]) for x in psel)/max(len(psel),1) if psel else 0.0
    gh  = sum(float(x[0]) for x in psel)/max(len(psel),1) if psel else 0.0
    return tokps, wall_ms, segsum, seg_ms, ple, gh

hdr = ["P","tok/s","wall ms","segsum","gap%"] + segs + ["ple_wait","gemm_host"]
print(("\t".join(hdr)))
for p in sys.argv[1:]:
    tokps, wall_ms, segsum, seg_ms, ple, gh = parse(int(p))
    gap = (wall_ms - segsum) / wall_ms * 100 if wall_ms else 0.0
    row = [p, f"{tokps:.1f}", f"{wall_ms:.1f}", f"{segsum:.1f}", f"{gap:.1f}"]
    row += [f"{seg_ms.get(s,0.0):.1f}" for s in segs]
    row += [f"{ple:.1f}", f"{gh:.1f}"]
    print("\t".join(row))
EOF
