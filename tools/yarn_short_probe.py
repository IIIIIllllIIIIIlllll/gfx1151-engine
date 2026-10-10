#!/usr/bin/env python3
"""Send a few short prompts, print responses (for factor1/factor2 comparison)."""
import json
import sys
import urllib.request

base = "http://127.0.0.1:8731"
prompts = [
    "用三句话介绍杭州。",
    "Write a Python function that checks whether a string is a palindrome. Code only.",
    "What is 17 * 23? Show the calculation briefly.",
]
tag = sys.argv[1] if len(sys.argv) > 1 else "run"
for i, p in enumerate(prompts):
    body = {"model": "qwenox", "messages": [{"role": "user", "content": p}],
            "temperature": 0, "max_tokens": 220}
    req = urllib.request.Request(base + "/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    out = json.loads(urllib.request.urlopen(req, timeout=300).read())
    m = out["choices"][0]["message"]
    print(f"=== [{tag}] prompt{i}: {p[:40]}", flush=True)
    print("content:", (m.get("content") or "")[:400], flush=True)
    print("reasoning:", (m.get("reasoning_content") or "")[:200], flush=True)
