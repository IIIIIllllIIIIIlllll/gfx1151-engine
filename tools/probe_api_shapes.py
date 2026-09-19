#!/usr/bin/env python3
"""Capture exact response shapes from a reference service (black-box).

These become the spec the front-end must reproduce byte-for-byte.
Writes nothing to the service beyond tiny greedy generations.
"""
import json
import os
import sys
import urllib.error
import urllib.request

BASE = "http://127.0.0.1:8731"
MID = os.environ.get("MODEL_ID", "qwen3.8-flash-next")


def call(method, path, payload=None, raw=False, limit=4000):
    url = BASE + path
    if payload is None:
        data = None
    elif isinstance(payload, (bytes, str)):
        data = payload.encode() if isinstance(payload, str) else payload
    else:
        data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            body = r.read().decode("utf-8", "replace")
            status, hdrs = r.status, dict(r.headers)
    except urllib.error.HTTPError as e:
        body, status, hdrs = e.read().decode("utf-8", "replace"), e.code, dict(e.headers)
    print("=" * 78)
    print(f"{method} {path}")
    if payload is not None:
        shown = payload if isinstance(payload, (bytes, str)) else json.dumps(payload, ensure_ascii=False)
        print("  req:", str(shown)[:300])
    print(f"  status: {status}")
    print(f"  content-type: {hdrs.get('content-type')}")
    print(f"  body ({len(body)} bytes):")
    print(body[:limit])
    return status, body


call("GET", "/v1/models")
call("GET", "/cache")
call("POST", "/v1/completions",
     {"model": MID, "prompt": "The capital of France is", "max_tokens": 6,
      "temperature": 0})
call("POST", "/v1/chat/completions",
     {"model": MID, "messages": [{"role": "user", "content": "Say hi in one word."}],
      "max_tokens": 6, "temperature": 0})
call("POST", "/v1/chat/completions",
     {"model": MID, "messages": [{"role": "user", "content": "hi"}],
      "max_tokens": 8, "temperature": 0, "stream": True})
print("(above streaming body is bounded by the harness print limit)")
call("POST", "/v1/completions",
     {"model": MID, "prompt": "x", "max_tokens": 4, "temperature": 0,
      "logprobs": 3, "echo": True})
call("POST", "/v1/completions", {"model": "wrong-model", "prompt": "x", "max_tokens": 2})
call("POST", "/v1/completions", {"model": MID, "prompt": "x", "max_tokens": 2,
                                 "temperature": -1})
call("POST", "/v1/completions", {"model": MID, "max_tokens": 2})
call("POST", "/v1/chat/completions",
     {"model": MID, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 4,
      "temperature": 0, "drafter": "nope"})
call("POST", "/v1/completions", "{not json", raw=True)
call("POST", "/v1/responses",
     {"model": MID, "input": "Say hi in one word.", "max_output_tokens": 6,
      "temperature": 0})
call("POST", "/v1/chat/completions",
     {"model": MID, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 4,
      "temperature": 0, "stop": ["\n"]})
call("POST", "/v1/completions",
     {"model": MID, "prompt": "hi", "temperature": 1.0, "seed": 1234, "max_tokens": 4,
      "logprobs": 2})
call("POST", "/v1/completions",
     {"model": MID, "prompt": "hi", "max_tokens": 4, "temperature": 0, "stream": True})
call("GET", "/v1/models/extra")
call("POST", "/v1/chat/completions",
     {"model": MID, "messages": [{"role": "user",
                                  "content": [{"type": "image_url",
                                               "image_url": {"url": "http://x/y.png"}}]}],
      "max_tokens": 2, "temperature": 0})
