#!/usr/bin/env python3
"""Delta-debug the tokenizer A/B mismatches down to a minimal window."""
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOK_DIR = os.environ.get("TOK_DIR", os.path.join(ROOT, "models/tokenizer"))
CLI = os.path.join(ROOT, "build/tok_cli")
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.join(ROOT, ".venv-serve", "lib", "python3.13", "site-packages"))

from tokenizers import Tokenizer as HFTokenizer  # noqa: E402
import tok_ab  # noqa: E402


class Cli:
    def __init__(self):
        self.p = subprocess.Popen([CLI, TOK_DIR], stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, text=True,
                                  encoding="utf-8", bufsize=1)

    def encode(self, text):
        self.p.stdin.write(json.dumps({"op": "encode", "text": text,
                                       "offsets": True}, ensure_ascii=False) + "\n")
        self.p.stdin.flush()
        return json.loads(self.p.stdout.readline())


def main():
    ht = HFTokenizer.from_file(os.path.join(TOK_DIR, "tokenizer.json"))
    cli = Cli()
    tok_ab.build_corpus()
    items = tok_ab.corpus

    def ref(t):
        return ht.encode(t, add_special_tokens=False).ids

    def run(t):
        return cli.encode(t)["ids"]

    bad = [(c, t) for c, t in items if run(t) != ref(t)]
    print(f"mismatching: {len(bad)} / {len(items)}")

    def mismatch(t):
        return run(t) != ref(t)

    n_show = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    seen_sig = {}
    shown = 0
    for cat, text in bad:
        t = text
        changed = True
        while changed:
            changed = False
            for i in range(len(t)):
                cand = t[:i] + t[i + 1:]
                if cand and mismatch(cand):
                    t = cand
                    changed = True
                    break
        # trim ends further in pairs
        a, b = run(t), ref(t)
        sig = (tuple(a), tuple(b))
        if sig in seen_sig:
            continue
        seen_sig[sig] = t
        shown += 1
        cps = " ".join(f"U+{ord(c):04X}" for c in t)
        print(f"\n=== case {shown} [{cat}] ===")
        print(f"window : {t!r}")
        print(f"cps    : {cps}")
        print(f"cli    : {a}")
        print(f"ref    : {b}")
        print("tokens:")
        for tag, ids in (("cli", a), ("ref", b)):
            pieces = []
            for i in ids:
                pieces.append(f"{i}:{ht.decode([i], skip_special_tokens=False)!r}")
            print(f"  {tag}: " + " ".join(pieces))
        if shown >= n_show:
            break
    cli.p.stdin.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
