#!/usr/bin/env python3
"""YaRN 512K ladder + needle test against a running qwenox-api.

Rungs of increasing prompt length; needle codewords at several depths on the
long rungs. Every rung uses unique filler (rung id in each paragraph) so no
cross-rung prefix cache hits distort prefill timing.

Usage: python3 tools/yarn_ladder.py [--base http://127.0.0.1:8731] [--save-dir /tmp/yarn]
"""
import argparse
import json
import time
import urllib.request
from pathlib import Path

FILLER = ("Archive entry {rung}-{i:05d}: The regional committee reviewed routine "
          "logistics, warehouse inventories, transport schedules, and maintenance "
          "budgets for the quarter. No anomalies were recorded in the minutes; "
          "the audit proceeded without remarks and all documents were filed for "
          "long term reference in the municipal records office.\n")

NEEDLE = ("\nMEMORANDUM {rung}-{label}: The secret codeword for sector {label} "
          "is {word}. All staff must treat this codeword as confidential.\n\n")

QUESTION = ("\n\nQuestion: What are the secret codewords for sectors {labels}? "
            "Answer with the codewords only, one per line, in the order asked.")

SECTOR_WORDS = {
    "alpha": "ZEPHYR-{tag}-417",
    "beta": "OBSIDIAN-{tag}-852",
    "gamma": "LANTERN-{tag}-693",
}


def build_prompt(rung, target_tokens, depths, tag):
    """Return (prompt_text, expected_words) with ~target_tokens of filler."""
    # ~64 tokens per filler paragraph (measured empirically for Qwen BPE).
    n_par = max(1, int(target_tokens / 64))
    sectors = list(SECTOR_WORDS)[: len(depths)]
    words = {s: SECTOR_WORDS[s].format(tag=tag) for s in sectors}
    needle_at = {int(n_par * d): s for d, s in zip(depths, sectors)}
    parts = []
    for i in range(n_par):
        if i in needle_at:
            s = needle_at[i]
            parts.append(NEEDLE.format(rung=rung, label=s, word=words[s]))
        parts.append(FILLER.format(rung=rung, i=i))
    labels = ", ".join(sectors)
    parts.append(QUESTION.format(labels=labels))
    return "".join(parts), [words[s] for s in sectors]


def chat(base, prompt, max_tokens):
    body = {
        "model": "qwenox",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "stream": False,
    }
    req = urllib.request.Request(
        base + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=7200) as resp:
        out = json.loads(resp.read())
    dt = time.time() - t0
    return out, dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8731")
    ap.add_argument("--save-dir", default="/tmp/yarn_prompts")
    ap.add_argument("--only", default="", help="comma list of rung targets")
    args = ap.parse_args()
    save = Path(args.save_dir)
    save.mkdir(parents=True, exist_ok=True)

    # (name, target_tokens, needle_depths or None)
    # measured ~64 tokens per filler paragraph (Qwen BPE, this template)
    rungs = [
        ("32k", 32000, None),
        ("128k", 128000, None),
        ("256k", 256000, None),
        ("300k", 300000, [0.10, 0.50, 0.90]),
        ("400k", 400000, [0.10, 0.50, 0.90]),
        ("512k", 500000, [0.05, 0.50, 0.97]),
    ]
    if args.only:
        keep = set(args.only.split(","))
        rungs = [r for r in rungs if r[0] in keep]

    results = []
    for name, target, depths in rungs:
        prompt, words = build_prompt(name, target, depths or [], name)
        (save / f"prompt_{name}.txt").write_text(prompt, encoding="utf-8")
        try:
            out, dt = chat(args.base, prompt, 220 if depths else 64)
        except Exception as e:  # noqa: BLE001
            print(f"[{name}] REQUEST FAILED: {e}", flush=True)
            results.append((name, target, None, None, dt if 'dt' in dir() else 0, str(e)))
            continue
        usage = out.get("usage") or {}
        pt = usage.get("prompt_tokens")
        ct = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
        msg = out["choices"][0].get("message") or {}
        text = (msg.get("content") or "") + "\n" + (msg.get("reasoning_content") or "")
        comp = usage.get("completion_tokens", 0)
        hits = [w for w in words if w in text]
        pps = pt / dt if pt and dt else 0
        print(f"[{name}] target={target} prompt_tokens={pt} cached={ct} "
              f"wall={dt:.1f}s prefill~{pps:.0f}tok/s completion={comp}tok needle={len(hits)}/{len(words)}",
              flush=True)
        print(f"[{name}] response: {text[:300]!r}", flush=True)
        results.append((name, target, pt, ct, dt, f"needle {len(hits)}/{len(words)}"))

    print("\n==== SUMMARY ====")
    for r in results:
        print(r, flush=True)


if __name__ == "__main__":
    main()
