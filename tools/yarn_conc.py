#!/usr/bin/env python3
"""Concurrency + KV pool sharing test against a PARALLEL=2 qwenox-api.

Phase 1: send prompts A,B sequentially -> record outputs.
Phase 2: send A,B concurrently -> expect bit-identical outputs (round-robin
         scheduling must not change numerics) and needle hits in both.
Phase 3 (--contend): two ~400K prompts concurrently, pool holds 524288 ->
         expect exactly one clean rejection, the other completes.
"""
import json
import sys
import threading
import time
import urllib.request
import urllib.error

BASE = "http://127.0.0.1:8731"

FILLER = ("Archive entry {tag}-{i:05d}: The harbor office recorded berth "
          "assignments, crane shifts, customs stamps, and tide tables. All "
          "entries were countersigned and filed without incident.\n")
NEEDLE = ("\nMEMO {tag}-{label}: the harbor master confirms the berth code "
          "is {word}. Treat it as confidential.\n\n")
WORDS = {"x": ["ANCHOR-{t}-114", "SEASTAR-{t}-229", "FOG-{t}-881"],
         "y": ["COMPASS-{t}-305", "TIDAL-{t}-662", "BUOY-{t}-097"]}
LABELS = ["alpha", "beta", "gamma"]


def build(tag, target_tokens):
    n_par = max(1, int(target_tokens / 64))
    words = [w.format(t=tag) for w in WORDS[tag]]
    depths = [0.10, 0.50, 0.90]
    at = {int(n_par * d): l for d, l in zip(depths, LABELS)}
    parts = []
    for i in range(n_par):
        if i in at:
            l = at[i]
            parts.append(NEEDLE.format(tag=tag, label=l, word=words[LABELS.index(l)]))
        parts.append(FILLER.format(tag=tag, i=i))
    parts.append("\n\nQuestion: what are the berth codes confirmed by the harbor "
                 "master for alpha, beta, gamma? One per line, codewords only.")
    return "".join(parts), words


def chat(prompt, max_tokens=220):
    body = {"model": "qwenox", "messages": [{"role": "user", "content": prompt}],
            "temperature": 0, "max_tokens": max_tokens, "stream": False}
    req = urllib.request.Request(BASE + "/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=7200) as resp:
            out = json.loads(resp.read())
        u = out["usage"]
        m = out["choices"][0]["message"]
        return {"ok": True, "wall": time.time() - t0, "pt": u["prompt_tokens"],
                "ct": u.get("completion_tokens"),
                "text": (m.get("content") or "") + "\n" + (m.get("reasoning_content") or "")}
    except urllib.error.HTTPError as e:
        return {"ok": False, "wall": time.time() - t0,
                "err": f"HTTP {e.code}: {e.read().decode()[:200]}"}


def hits(words, text):
    return sum(1 for w in words if w in text)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "consistency"
    if mode == "consistency":
        pa, wa = build("x", 40000)
        pb, wb = build("y", 40000)
        print("== phase 1: sequential ==", flush=True)
        ra = chat(pa); rb = chat(pb)
        for name, r, w in [("A", ra, wa), ("B", rb, wb)]:
            print(f"  seq {name}: ok={r['ok']} pt={r.get('pt')} wall={r.get('wall', 0):.1f}s "
                  f"needle={hits(w, r.get('text', ''))}/3", flush=True)
        print("== phase 2: concurrent ==", flush=True)
        results = {}

        def worker(k, p):
            results[k] = chat(p)
        ta = threading.Thread(target=worker, args=("A", pa))
        tb = threading.Thread(target=worker, args=("B", pb))
        t0 = time.time()
        ta.start(); tb.start(); ta.join(); tb.join()
        tot = time.time() - t0
        for name, w in [("A", wa), ("B", wb)]:
            r = results[name]
            print(f"  conc {name}: ok={r['ok']} pt={r.get('pt')} wall={r.get('wall', 0):.1f}s "
                  f"needle={hits(w, r.get('text', ''))}/3", flush=True)
        same_a = ra.get("text") == results["A"].get("text")
        same_b = rb.get("text") == results["B"].get("text")
        print(f"  concurrent wall={tot:.1f}s bit_identical: A={same_a} B={same_b}", flush=True)
        if not same_a:
            print("  seq A tail:", (ra.get('text') or '')[-120:], flush=True)
            print("  conc A tail:", (results['A'].get('text') or '')[-120:], flush=True)
        if not same_b:
            print("  seq B tail:", (rb.get('text') or '')[-120:], flush=True)
            print("  conc B tail:", (results['B'].get('text') or '')[-120:], flush=True)
    elif mode == "contend":
        pc, wc = build("x", 520000)   # ~311K actual tokens
        pd, wd = build("y", 520000)
        print("== contention: 2 x ~311K concurrent, pool=524288 ==", flush=True)
        results = {}

        def worker(k, p):
            results[k] = chat(p, 16)
        ta = threading.Thread(target=worker, args=("C", pc))
        tb = threading.Thread(target=worker, args=("D", pd))
        time.sleep(0)  # keep start order deterministic: C first
        ta.start(); time.sleep(0.3); tb.start()
        ta.join(); tb.join()
        for name in ("C", "D"):
            r = results[name]
            if r["ok"]:
                print(f"  {name}: OK pt={r['pt']} ct={r['ct']} wall={r['wall']:.1f}s", flush=True)
            else:
                print(f"  {name}: REJECTED wall={r['wall']:.1f}s {r['err']}", flush=True)
    else:
        print("unknown mode", flush=True)


if __name__ == "__main__":
    main()
