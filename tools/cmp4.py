#!/usr/bin/env python3
# cmp4.py — gdec vs server gold snapshot (server_greedy24.json, same dir), no docker needed.
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from qwentok import Tokenizer

ROOT = Path(__file__).resolve().parent.parent
GDEC = str(ROOT / "build/gdec")
MODEL_DIR = os.environ.get("MODEL_DIR", "models")
BASE = f"{MODEL_DIR}/qwen38-flash-next-w4b.hgn"
OVL = f"{MODEL_DIR}/qwen38-flash-next-w4b.overlay.hgn"
SNAPSHOT = ROOT / "tools/server_greedy24.json"
GEN = 24


def main(argv=None):
    parser = argparse.ArgumentParser(description="Compare gdec with server snapshots.")
    parser.add_argument("env_extra", nargs="?", help="e.g. GDEC_NOPREFILLBATCH=1")
    args = parser.parse_args(argv)
    env = dict(os.environ)
    if args.env_extra is not None:
        key, sep, value = args.env_extra.partition("=")
        if not sep or not key or "\0" in args.env_extra:
            parser.error("expected NAME=VALUE")
        env[key] = value

    try:
        with open(SNAPSHOT, encoding="utf-8") as f:
            snap = json.load(f)
        if (not isinstance(snap, dict) or not snap or
                any(not isinstance(p, str) or not p or not isinstance(g, str)
                    for p, g in snap.items())):
            raise ValueError("snapshot must be a nonempty prompt-to-text object")
        tk = Tokenizer()
    except (OSError, ValueError, KeyError) as exc:
        print(f"cmp4: setup failed: {exc}", file=sys.stderr)
        return 1

    npass = 0
    for prompt, gold in snap.items():
        stderr = ""
        try:
            ids = tk.encode(prompt)
            p = subprocess.run(
                [GDEC, BASE, OVL, "--tokens", ",".join(map(str, ids)),
                 "--gen", str(GEN)], capture_output=True, text=True,
                encoding="utf-8", timeout=1800, env=env)
            stderr = p.stderr
            if p.returncode != 0:
                raise ValueError(f"gdec exited with status {p.returncode}")
            lines = [line[4:] for line in p.stdout.splitlines() if line.startswith("ids:")]
            if len(lines) != 1:
                raise ValueError(f"expected one ids: line, got {len(lines)}")
            ours = [int(x) for x in lines[0].split()]
            if len(ours) != len(ids) + GEN:
                raise ValueError(f"expected {len(ids) + GEN} token ids, got {len(ours)}")
            if ours[:len(ids)] != ids:
                raise ValueError("output prompt ids do not match input")
            if any(token not in tk.id2tok for token in ours):
                raise ValueError("output contains unknown token ids")
            text = tk.decode(ours[len(ids):])
        except (OSError, subprocess.TimeoutExpired, ValueError, KeyError) as exc:
            print(f"[{prompt!r}] GDEC FAILED: {exc}", flush=True)
            if stderr:
                print(stderr[-2000:], file=sys.stderr, flush=True)
            continue
        ok = text == gold
        npass += ok
        print(f"[{prompt!r}] {'MATCH' if ok else 'DIFF'}", flush=True)
        if not ok:
            print("  gold:", repr(gold))
            print("  ours:", repr(text))
    print(f"{npass}/{len(snap)} byte-identical", flush=True)
    return 0 if npass == len(snap) else 1


if __name__ == "__main__":
    sys.exit(main())
