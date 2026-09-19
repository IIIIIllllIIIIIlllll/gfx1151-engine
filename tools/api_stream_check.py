#!/usr/bin/env python3
"""Per-frame validation of the front-end's SSE streams.

A streaming client parses each `data:` frame independently, so every frame must
be valid UTF-8 containing valid JSON. This catches the class of bug where a
multi-byte character is split across frames (the concatenation looks fine, but
no single frame is parseable) and where the terminator is not exactly `[DONE]`.

Run: .venv-serve/bin/python tools/api_stream_check.py [--base URL]
"""
import argparse
import json
import sys
import urllib.request

PROMPTS = [
    ("ascii", "hi"),
    ("cjk", "用一句话介绍你自己"),
    ("emoji", "用三个 emoji 表达开心"),
    ("code", "写一个 python 的 hello world"),
    ("mixed", "你好 hello 世界 🌏 café ① ﾊﾞ"),
]


def stream(base, path, body):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    raw = b""
    with urllib.request.urlopen(req, timeout=600) as r:
        ct = r.headers.get("content-type", "")
        while True:
            b = r.read(1)
            if not b:
                break
            raw += b
    problems = []
    frames = 0
    done = False
    text = ""
    reasoning = ""
    for i, fr in enumerate(raw.split(b"\n\n")):
        if not fr:
            continue
        if not fr.startswith(b"data: "):
            problems.append(f"frame {i}: no 'data: ' prefix: {fr[:60]!r}")
            continue
        p = fr[6:]
        if p == b"[DONE]":
            done = True
            continue
        try:
            s = p.decode("utf-8")
        except UnicodeDecodeError as e:
            problems.append(f"frame {i}: invalid UTF-8 at byte {e.start}: {p[max(0,e.start-20):e.end+20]!r}")
            continue
        try:
            obj = json.loads(s)
        except Exception as e:  # noqa: BLE001
            problems.append(f"frame {i}: invalid JSON: {e}: {s[:120]!r}")
            continue
        frames += 1
        ch = obj.get("choices") or []
        if ch:
            # chat chunks carry `delta`, completion chunks carry `text`.
            c0 = ch[0]
            txt = c0.get("text")
            if isinstance(txt, str):
                text += txt
            d = c0.get("delta") or {}
            if isinstance(d, dict):
                text += d.get("content") or ""
                reasoning += d.get("reasoning_content") or ""
    if not done:
        problems.append("stream did not end with a bare `data: [DONE]` frame")
    return ct, frames, reasoning, text, problems


def nonstream(base, path, body):
    b = dict(body)
    b.pop("stream", None)
    b.pop("stream_options", None)
    req = urllib.request.Request(base + path, data=json.dumps(b).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        obj = json.loads(r.read().decode("utf-8"))
    ch = obj["choices"][0]
    if path.endswith("completions") and "text" in ch:
        return "", ch["text"]
    m = ch.get("message") or {}
    return m.get("reasoning_content") or "", m.get("content") or ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8731")
    ap.add_argument("--max-tokens", type=int, default=64)
    args = ap.parse_args()

    rc = 0
    for name, prompt in PROMPTS:
        for path, body in (
            ("/v1/chat/completions",
             {"model": "m", "messages": [{"role": "user", "content": prompt}],
              "max_tokens": args.max_tokens, "temperature": 0, "stream": True,
              "stream_options": {"include_usage": True}}),
            ("/v1/completions",
             {"model": "m", "prompt": prompt, "max_tokens": args.max_tokens,
              "temperature": 0, "stream": True}),
        ):
            try:
                ct, frames, sr, st, problems = stream(args.base, path, body)
                nr, nt = nonstream(args.base, path, body)
            except Exception as e:  # noqa: BLE001
                print(f"[FAIL] {name} {path}: request error: {e}")
                rc = 1
                continue
            if (sr, st) != (nr, nt):
                problems.append(
                    f"stream != non-stream: reasoning {'ok' if sr == nr else 'DIFF'}, "
                    f"content {'ok' if st == nt else 'DIFF'}"
                    f" (stream content={st[:40]!r} vs non-stream={nt[:40]!r})")
            status = "ok  " if not problems else "FAIL"
            print(f"[{status}] {name:6s} {path:22s} frames={frames:3d} "
                  f"content={st[:28]!r}")
            for p in problems[:4]:
                print(f"         - {p}")
            if problems:
                rc = 1
    print("RESULT:", "PASS" if rc == 0 else "FAIL")
    return rc


if __name__ == "__main__":
    sys.exit(main())
