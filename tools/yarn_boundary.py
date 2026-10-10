#!/usr/bin/env python3
"""Exact-context-boundary tests against a running qwenox-api (completions path).

Token-exact prompt construction: filler paragraphs + " x" padding, verified
with build/tok_cli (same tokenizer as the server). Run on the engine host.

Cases (ctx=524288):
  oversize   : ~560K-token prompt -> expect fast 400, engine unharmed
  exact-full : exactly 524288 tokens -> budget 0, expect clean rejection
  budget-1   : exactly 524287 tokens -> max_tokens=64 -> exactly 1 completion token
  budget-25  : exactly 524263 tokens -> max_tokens=64 -> exactly 25 completion tokens
"""
import json
import subprocess
import sys
import time
import urllib.request
import urllib.error

BASE = "http://127.0.0.1:8731"
CTX = 524288

FILLER = ("Boundary entry {i:06d}: The depot ledger lists routine shipments, "
          "fuel logs, driver rosters, and pallet counts for the week. All "
          "figures matched the quarterly forecast and no incident was filed.\n")


def tok_count(text):
    proc = subprocess.run(["build/tok_cli", "models/tokenizer"],
                          input=json.dumps({"op": "encode", "text": text}) + "\n",
                          capture_output=True, text=True, check=True)
    return len(json.loads(proc.stdout.strip().splitlines()[-1])["ids"])


def build_exact(n):
    """Text whose tokenization is exactly n tokens (verified)."""
    ids_par = tok_count(FILLER.format(i=0))
    k = max(0, (n - 600) // ids_par)
    while True:
        text = "".join(FILLER.format(i=i) for i in range(k))
        c = tok_count(text)
        if c <= n:
            break
        k -= 10
    text += " x" * (n - c)
    got = tok_count(text)
    if got != n:
        raise RuntimeError(f"exact build failed: got {got}, want {n}")
    return text


def completions(prompt, max_tokens):
    body = {"model": "qwenox", "prompt": prompt, "temperature": 0,
            "max_tokens": max_tokens, "stream": False}
    req = urllib.request.Request(BASE + "/v1/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=7200) as resp:
            out = json.loads(resp.read())
        return time.time() - t0, out, None
    except urllib.error.HTTPError as e:
        return time.time() - t0, None, (e.code, e.read().decode()[:400])


def run_case(case):
    if case == "oversize":
        text = "".join(FILLER.format(i=i) for i in range(13000))  # ~572K
        print(f"[oversize] built chars={len(text)}", flush=True)
        dt, out, err = completions(text, 8)
        if err:
            print(f"[oversize] wall={dt:.1f}s HTTP {err[0]}: {err[1]}", flush=True)
        else:
            print(f"[oversize] wall={dt:.1f}s UNEXPECTED SUCCESS "
                  f"prompt_tokens={out['usage']['prompt_tokens']}", flush=True)
        return
    if case == "exact-full":
        text = build_exact(CTX)
    else:
        leave = int(case.split("-")[1])
        text = build_exact(CTX - leave)
    n = tok_count(text)
    print(f"[{case}] verified prompt = {n} tokens", flush=True)
    dt, out, err = completions(text, 64)
    if err:
        print(f"[{case}] wall={dt:.1f}s HTTP {err[0]}: {err[1]}", flush=True)
    else:
        u = out["usage"]
        ch = out["choices"][0]
        print(f"[{case}] wall={dt:.1f}s prompt={u['prompt_tokens']} "
              f"completion={u['completion_tokens']} finish={ch.get('finish_reason')} "
              f"text={(ch.get('text') or '')[:60]!r}", flush=True)


def main():
    cases = sys.argv[1].split(",") if len(sys.argv) > 1 else ["oversize", "exact-full", "budget-1", "budget-25"]
    for case in cases:
        run_case(case)


if __name__ == "__main__":
    main()
