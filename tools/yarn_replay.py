#!/usr/bin/env python3
"""Replay saved prompts and report cache hits (KVSNAP verification)."""
import json
import sys
import time
import urllib.request

base = "http://127.0.0.1:8731"
for name in sys.argv[1:]:
    text = open(f"/tmp/yarn_prompts/prompt_{name}.txt").read()
    body = {"model": "qwenox",
            "messages": [{"role": "user", "content": text}],
            "temperature": 0, "max_tokens": 8}
    req = urllib.request.Request(base + "/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    out = json.loads(urllib.request.urlopen(req, timeout=7200).read())
    dt = time.time() - t0
    u = out["usage"]
    det = u.get("prompt_tokens_details") or {}
    print(f"[{name}] prompt_tokens={u['prompt_tokens']} "
          f"cached={det.get('cached_tokens', 0)} wall={dt:.1f}s", flush=True)
