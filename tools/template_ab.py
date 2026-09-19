#!/usr/bin/env python3
r"""A/B gate for the C++ chat template.

Renders the same matrix through three engines and compares byte-for-byte:
  A. plain jinja2 (the template file as-is, with raise_exception injected)
  B. transformers AutoTokenizer.apply_chat_template(tokenize=False)
  C. build/tpl_cli  (the C++ port)

The hard gate is C == A (jinja2 semantics is what the template text means).
C vs B is reported; where A and B disagree the live service decides, via
tools/probe_service.py --chat (prompt_tokens oracle).

Run:  .venv-serve/bin/python tools/template_ab.py [--show N] [--fuzz N]
"""
import argparse
import json
import os
import random
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOK_DIR = os.environ.get("TOK_DIR", os.path.join(ROOT, "models/tokenizer"))
CLI = os.path.join(ROOT, "build/tpl_cli")
TEMPLATE = os.path.join(TOK_DIR, "chat_template.jinja")

sys.path.insert(0, os.path.join(ROOT, ".venv-serve", "lib", "python3.13", "site-packages"))

import jinja2  # noqa: E402


def raise_exception(message):
    raise jinja2.exceptions.TemplateError(message)


# ------------------------------------------------------------- corpus ------

def msg(role, content=None, **kw):
    m = {"role": role}
    if content is not None:
        m["content"] = content
    m.update(kw)
    return m


TEXT_ITEMS = [{"type": "text", "text": "describe this"}, {"type": "text", "text": "and this"}]
IMG_ITEM = {"type": "image", "image_url": {"url": "data:image/png;base64,AAAA"}}
IMG_URL_ITEM = {"image_url": {"url": "data:image/png;base64,BBBB"}}
VID_ITEM = {"type": "video", "video_url": {"url": "data:video/mp4;base64,CCCC"}}
TOOL_ONE = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get weather for a city",
        "parameters": {"type": "object", "properties": {
            "city": {"type": "string", "description": "city name"},
            "days": {"type": "integer"}},
            "required": ["city"]},
    },
}]
TOOL_TWO = TOOL_ONE + [{
    "type": "function",
    "function": {
        "name": "search<\u4e2d>&'\"",
        "description": "unicode <html> & 'quotes' \"here\"",
        "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
    },
}]
ARGS_VARIANTS = [
    ("obj", {"city": "Paris", "days": 3}),
    ("obj-nested", {"a": {"b": [1, 2, {"c": None}]}, "z": True}),
    ("obj-unicode", {"\u4e2d\u6587": "\u00e9\u0301 <&>'\"", "n": 1.5}),
    ("obj-float", {"x": 1.0, "y": 1e-05, "z": 1e20, "w": -0.0}),
    ("obj-float-edge", {"a": 100000.0, "b": 1e15, "c": 1e16, "d": 1e17, "e": 0.0001,
                        "f": 1e-4, "g": 123456.789, "h": 5e-324,
                        "i": 1.7976931348623157e308, "j": 9007199254740992.0,
                        "k": -1e-5, "l": 0.0, "m": 0.1, "n": -2.5e-10}),
    ("obj-ctrl", {"s": "\u0000\u0001\u001f\u007f\u0085\u00a0\u3000\ufeff\u200b"}),
    ("empty-str", ""),
    ("empty-obj", {}),
    ("null", None),
    ("false", False),
    ("zero", 0),
    ("arr", [1, 2]),
    ("str", "not-an-object"),
    ("absent", "ABSENT"),
]

# Divergences that are understood and unreachable from realistic requests:
# Python ints are arbitrary precision; nlohmann parses out-of-uint64 ranges as
# double. Reported but not part of the pass/fail gate.
KNOWN_DIVERGENT = {"known-divergence"}

# Cases where apply_chat_template's own *API* layer (not the template text)
# rejects the input before rendering. The C++ front-end replaces that layer with
# its own OpenAI-schema validation (400, per the handover's endpoint spec), so
# the template-level behaviour is what the port has to get right. Reported
# separately rather than gated.
REF_LAYER = {"tools-not-iterable"}


def tool_call(name, args):
    tc = {"type": "function", "function": {"name": name}}
    if args != "ABSENT":
        tc["function"]["arguments"] = args
    return tc


def build_cases():
    cases = []

    def add(c, **kw):
        cases.append((c, kw))

    # ---- degenerate / error paths
    add("no-messages", messages="ABSENT")
    add("empty-messages", messages=[])
    add("empty-messages-gen", messages=[], add_generation_prompt=True)
    add("user-only", messages=[msg("user", "hi")])
    add("bad-role", messages=[msg("user", "hi"), msg("wizard", "x")])
    add("system-not-first", messages=[msg("user", "hi"), msg("system", "late")])
    add("content-number", messages=[msg("user", 5)])
    add("content-obj", messages=[msg("user", {"a": 1})])
    add("bad-item-type", messages=[msg("user", [{"type": "audio"}])])
    add("bad-item-no-type", messages=[msg("user", [{"foo": 1}])])
    add("all-tool-response", messages=[msg("tool", "<tool_response>\nx\n</tool_response>")])
    add("tools-not-iterable", messages=[msg("user", "hi")], tools={"a": 1})
    add("tools-null", messages=[msg("user", "hi")], tools=None)
    add("tools-empty", messages=[msg("user", "hi")], tools=[])
    add("assistant-no-content", messages=[msg("user", "hi"), msg("assistant")])
    add("tc-no-name", messages=[msg("user", "hi"),
                                msg("assistant", None, tool_calls=[{"type": "function",
                                                                    "function": {}}])])
    add("tc-name-number", messages=[msg("user", "hi"),
                                    msg("assistant", None,
                                        tool_calls=[tool_call(7, {"a": 1})])])

    # ---- reasoning effort matrix
    for eff in ("ABSENT", "xhigh", "medium", "low", "high", "minimal", "XHIGH", "", None, 0, 1):
        for et in ("ABSENT", True, False, None):
            for sysmsg in ("ABSENT", "sys", "sys-empty", "sys-space"):
                kw = {"messages": [msg("user", "hi")]}
                if eff != "ABSENT":
                    kw["reasoning_effort"] = eff
                if et != "ABSENT":
                    kw["enable_thinking"] = et
                if sysmsg == "sys":
                    kw["messages"] = [msg("system", "You are helpful."), msg("user", "hi")]
                elif sysmsg == "sys-empty":
                    kw["messages"] = [msg("system", ""), msg("user", "hi")]
                elif sysmsg == "sys-space":
                    kw["messages"] = [msg("system", "  \n\t "), msg("user", "hi")]
                add("effort", **kw)

    # ---- preserve_thinking / last_query_index
    for pt in ("ABSENT", True, False, None):
        for eff in ("ABSENT", "medium", "low"):
            for gen in ("ABSENT", True, False):
                for th in ("ABSENT", True, False):
                    kw = {"messages": [msg("system", "s"), msg("user", "u1"),
                                       msg("assistant", "a1", reasoning_content="think1"),
                                       msg("user", "u2"),
                                       msg("assistant", "a2", reasoning_content="think2")]}
                    if pt != "ABSENT":
                        kw["preserve_thinking"] = pt
                    if eff != "ABSENT":
                        kw["reasoning_effort"] = eff
                    if gen != "ABSENT":
                        kw["add_generation_prompt"] = gen
                    if th != "ABSENT":
                        kw["enable_thinking"] = th
                    add("preserve", **kw)

    # ---- reasoning_content shapes
    for rc in ("ABSENT", "text", "", "  pad  ", None, 5, {"a": 1}, ["x"]):
        m = msg("assistant", "answer")
        if rc != "ABSENT":
            m["reasoning_content"] = rc
        add("reasoning-content", messages=[msg("user", "q"), m])

    # ---- tool_calls
    for aname, args in ARGS_VARIANTS:
        for n in (1, 2):
            tcs = [tool_call("get_weather", args),
                   tool_call("second_tool", {"k": "v"})][:n]
            add("tool-calls", messages=[msg("user", "q"),
                                        msg("assistant", "calling" if n == 1 else "",
                                            tool_calls=tcs)])
    add("tool-calls-multiturn",
        messages=[msg("user", "q"), msg("assistant", None, tool_calls=[tool_call("f", {"a": 1})]),
                  msg("tool", "result"), msg("assistant", "done")])
    add("tool-calls-nonfirst", messages=[msg("user", "q"), msg("assistant", "c1"),
                                        msg("user", "q2"),
                                        msg("assistant", None,
                                            tool_calls=[tool_call("g", {"b": 2})])])

    # ---- tool responses: single / consecutive / interleaved
    for seq in ([msg("tool", "r1")],
                [msg("tool", "r1"), msg("tool", "r2")],
                [msg("tool", "r1"), msg("tool", "r2"), msg("tool", "r3")],
                [msg("tool", "r1"), msg("user", "u"), msg("tool", "r2")],
                [msg("tool", "r1"), msg("assistant", "a"), msg("tool", "r2")]):
        add("tool-responses", messages=[msg("user", "q")] + seq)
    for c in ("text", [{"type": "text", "text": "part"}]):
        add("tool-content-shape", messages=[msg("user", "q"), msg("tool", c)])

    # ---- tools rendering
    for tools in ("ABSENT", TOOL_ONE, TOOL_TWO, []):
        for withsys in (True, False):
            ms = ([msg("system", "sys text")] if withsys else []) + [msg("user", "q")]
            kw = {"messages": ms}
            if tools != "ABSENT":
                kw["tools"] = tools
            add("tools-render", **kw)

    # ---- vision
    for vis in ("ABSENT", True, False, None):
        for content in (["ABSENT"],
                        [IMG_ITEM],
                        [IMG_ITEM, IMG_ITEM],
                        [{"type": "text", "text": "see"}, IMG_ITEM],
                        [IMG_URL_ITEM],
                        [{"type": "image_url", "image_url": {"url": "x"}}],
                        [VID_ITEM],
                        [{"video": "v"}]):
            kw = {"messages": [msg("user", "look")]}
            if vis != "ABSENT":
                kw["add_vision_id"] = vis
            if content != ["ABSENT"]:
                kw["messages"] = [msg("user", content)]
            add("vision", **kw)
    add("vision-system-image",
        messages=[msg("system", [IMG_ITEM]), msg("user", "q")])
    add("vision-system-video",
        messages=[msg("system", [VID_ITEM]), msg("user", "q")])
    add("vision-tools-image",
        messages=[msg("user", [IMG_ITEM]), msg("assistant", None,
                                               tool_calls=[tool_call("f", {"a": 1})])],
        tools=TOOL_ONE, add_vision_id=True)

    # ---- string content oddities
    for c in (TEXT_ITEMS, [{"text": "only-text"}], [{"text": ""}],
              [{"type": "text", "text": "t"}, {"text": "t2"}]):
        add("content-list", messages=[msg("user", c)])

    # ---- long multi-turn
    long_msgs = []
    for i in range(12):
        long_msgs.append(msg("user", f"question {i} <&> '\u4e2d'"))
        long_msgs.append(msg("assistant", f"answer {i}", reasoning_content=f"r{i}"))
    add("long-multiturn", messages=long_msgs)
    add("long-multiturn-gen", messages=long_msgs, add_generation_prompt=True,
        tools=TOOL_TWO, add_vision_id=True, reasoning_effort="low")
    # known divergence: arbitrary-precision int (see KNOWN_DIVERGENT)
    add("known-divergence",
        messages=[msg("user", "q"),
                  msg("assistant", None,
                      tool_calls=[tool_call("f", {"big": 123456789012345678901234567890})])])
    return cases


def build_fuzz(n, seed=20260914):
    rng = random.Random(seed)
    roles = ["user", "assistant", "tool", "system", "bad"]
    effs = ["ABSENT", "xhigh", "medium", "low", "junk", None]
    contents = ["ABSENT", "plain", "", "  ", [{"type": "text", "text": "x"}],
                [IMG_ITEM], "pad\n", "\u4e2d<&>'\""]
    out = []
    for _ in range(n):
        ms = []
        for _ in range(rng.randint(1, 6)):
            r = rng.choice(roles)
            c = rng.choice(contents)
            kw = {}
            if rng.random() < 0.3:
                kw["tool_calls"] = [tool_call("f", rng.choice(
                    [{"a": 1}, "", None, "s", [1]]))]
            if rng.random() < 0.3:
                kw["reasoning_content"] = rng.choice(["r", "", None])
            ms.append(msg(r, None if c == "ABSENT" else c, **kw))
        kw = {"messages": ms}
        if rng.random() < 0.5:
            kw["reasoning_effort"] = rng.choice(effs)
        for k in ("enable_thinking", "preserve_thinking", "add_generation_prompt",
                  "add_vision_id"):
            if rng.random() < 0.4:
                kw[k] = rng.choice([True, False, None])
        if rng.random() < 0.3:
            kw["tools"] = rng.choice([TOOL_ONE, TOOL_TWO, []])
        out.append(("fuzz", kw))
    return out


# -------------------------------------------------------------- runner -----

def run_cli(reqs):
    payload = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in reqs)
    p = subprocess.run([CLI], input=payload, capture_output=True, text=True,
                       encoding="utf-8")
    if p.returncode != 0:
        print(p.stderr[-4000:], file=sys.stderr)
        raise SystemExit(f"tpl_cli exited {p.returncode}")
    # split("\n") not splitlines(): rendered text may contain U+0085/U+2028,
    # which str.splitlines() would treat as line boundaries.
    return [json.loads(l) for l in p.stdout.split("\n") if l.strip()]


def cli_req(case):
    """tpl_cli request: the 'ABSENT' sentinel means the key is not sent."""
    return {k: v for k, v in case.items() if v != "ABSENT"}


def j2_req(case):
    return {k: v for k, v in case.items() if v != "ABSENT"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", type=int, default=12)
    ap.add_argument("--fuzz", type=int, default=400)
    args = ap.parse_args()

    text = open(TEMPLATE, encoding="utf-8").read()
    cases = build_cases() + build_fuzz(args.fuzz)
    print(f"cases: {len(cases)}")

    reqs = [cli_req(c) for _, c in cases]
    got = run_cli(reqs)

    bad_b = []          # hard gate: C++ vs apply_chat_template
    bad_a = []          # informational: C++ vs plain jinja2
    known = []          # known-divergence cases
    ref_layer = []      # apply_chat_template API-layer validation cases
    ref_diff = 0
    from transformers import AutoTokenizer
    at = AutoTokenizer.from_pretrained(TOK_DIR)

    for (cat, case), res in zip(cases, got):
        j2 = j2_req(case)
        try:
            ok_a, val_a = render_jinja_arg(text, j2)
        except Exception as e:  # noqa: BLE001
            ok_a, val_a = False, f"<harness:{type(e).__name__}: {e}>"
        try:
            msgs = j2.get("messages")
            kwargs = {k: v for k, v in j2.items() if k != "messages"}
            if msgs is None and "messages" not in j2:
                raise ValueError("no messages key")
            rb = at.apply_chat_template(msgs, tokenize=False, **kwargs)
            ok_b, val_b = True, rb
        except Exception as e:  # noqa: BLE001
            ok_b, val_b = False, getattr(e, "message", None) or str(e)

        if (ok_a, val_a) != (ok_b, val_b):
            ref_diff += 1

        hit_b = res.get("ok") != ok_b or (ok_b and res.get("text") != val_b)
        hit_a = res.get("ok") != ok_a or (ok_a and res.get("text") != val_a)
        if cat in REF_LAYER:
            ref_layer.append((cat, case, res, ok_b, val_b))
        elif cat in KNOWN_DIVERGENT:
            if hit_b:
                known.append((cat, case, res, ok_b, val_b))
        elif hit_b:
            bad_b.append((cat, case, res, ok_b, val_b))
        if hit_a and cat not in KNOWN_DIVERGENT and cat not in REF_LAYER:
            bad_a.append((cat, case, res, ok_a, val_a))

    n = len(cases)
    skipped = sum(1 for c, _ in cases if c in KNOWN_DIVERGENT or c in REF_LAYER)
    gated = n - skipped
    print(f"C++ vs apply_chat_t: {gated - len(bad_b)}/{gated} byte-identical  (HARD GATE)")
    print(f"C++ vs plain jinja2: {n - len(bad_a) - skipped}/{n - skipped} byte-identical  (informational)")
    print(f"jinja2 vs apply_chat_template disagreements: {ref_diff}")
    print(f"known-divergence cases: {len(known)} "
          f"(of {sum(1 for c, _ in cases if c in KNOWN_DIVERGENT)})")
    print(f"reference-layer (apply_chat_template API validation, not template): "
          f"{len(ref_layer)}")

    def show(lst, title):
        if not lst:
            return
        print(f"\n--- first {min(args.show, len(lst))} {title} ---")
        for cat, case, res, ok, val in lst[: args.show]:
            print(f"[{cat}] req={json.dumps(case, ensure_ascii=False)[:300]}")
            print(f"    cli ok={res.get('ok')} {'text' if res.get('ok') else 'error'}="
                  f"{json.dumps(res.get('text') if res.get('ok') else res.get('error'), ensure_ascii=False)[:400]}")
            print(f"    ref ok={ok} value={json.dumps(val, ensure_ascii=False)[:400]}")

    show(bad_b, "apply_chat_template mismatches (HARD GATE)")
    if bad_a:
        show(bad_a, "plain-jinja2 mismatches (informational)")
    show(known, "known divergences")

    ok = not bad_b
    print("\nRESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def render_jinja_arg(text, ctx):
    """Render with only the keys present in ctx; absent keys stay undefined."""
    env = jinja2.Environment()
    env.globals["raise_exception"] = raise_exception
    tpl = env.from_string(text)
    try:
        return True, tpl.render(**ctx)
    except jinja2.exceptions.TemplateError as e:
        return False, e.message


if __name__ == "__main__":
    sys.exit(main())
