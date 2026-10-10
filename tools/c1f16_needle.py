#!/usr/bin/env python3
"""C1 f16 one-off needle check: 64K prompt, 3 codewords at 10/50/90% depth.
Reuses tools/yarn_ladder.py's prompt builder + chat(); run against a live
qwenox-api (start_win.sh). Usage: python3 tools/c1f16_needle.py [base_url]"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from yarn_ladder import build_prompt, chat  # noqa: E402

base = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8731"
prompt, words = build_prompt("c1f16-64k", 64000, [0.10, 0.50, 0.90], "c1f16")
Path("logs/c1f16_needle_prompt.txt").write_text(prompt, encoding="utf-8")
out, dt = chat(base, prompt, 220)
usage = out.get("usage") or {}
msg = out["choices"][0].get("message") or {}
text = (msg.get("content") or "") + "\n" + (msg.get("reasoning_content") or "")
hits = [w for w in words if w in text]
print(f"prompt_tokens={usage.get('prompt_tokens')} "
      f"cached={(usage.get('prompt_tokens_details') or {}).get('cached_tokens')} "
      f"completion={usage.get('completion_tokens')} wall={dt:.1f}s")
print(f"expected: {words}")
print(f"needle hits: {len(hits)}/{len(words)} {hits}")
print("--- completion ---")
print(text.strip()[:600])
sys.exit(0 if len(hits) == len(words) else 1)
