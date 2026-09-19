#!/usr/bin/env python3
r"""A/B gate for the C++ tokenizer.

Compares build/tok_cli against the tokenizer the deployment actually uses:
transformers v5 `AutoTokenizer.from_pretrained(<model dir>)`. That is NOT the
same pipeline as the raw `tokenizers.Tokenizer.from_file(tokenizer.json)` --
transformers rebuilds the pre-tokenizer from its class default, keeping `\p{M}`
out of the letter run. tools/probe_service.py settles which is authoritative by
black-box probing the live service's usage.prompt_tokens; it says AutoTokenizer.
The raw pipeline is still measured, and reported as an informational count.

Gates:
  * encode ids must match bit-for-bit on >=5000 examples covering every
    boundary class called out in the brief, plus >=200 fuzz cases;
  * decoder offsets are compared and reported (soft signal);
  * decode() must match for odd/even id sequences under both
    skip_special_tokens values.

Run:  .venv-serve/bin/python tools/tok_ab.py [--limit N] [-v]
Exit code 0 only when the hard gates are 100% clean.
"""
import argparse
import json
import os
import random
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOK_DIR = os.environ.get("TOK_DIR", os.path.join(ROOT, "models/tokenizer"))
CLI = os.path.join(ROOT, "build/tok_cli")

sys.path.insert(0, os.path.join(ROOT, ".venv-serve", "lib", "python3.13", "site-packages"))

# ---------------------------------------------------------------- corpus ----

CJK = "的一是不了人我在有他这为之大来以个中上们到说国和地也子时道出而要于就下得可你年生"
HANGUL = "가나다라마바사아자차카타파하각갂갃간갇갈갉"
KATA = "アイウエオカキクケコサシスセソタチツテト"
HALFWIDTH_KATA = "ｱｲｳｴｵｶｷｸｹｺｻｼｽｾｿﾀﾁﾂﾃﾄ"
ACCENTS = ["e\u0301", "a\u0300", "o\u0308\u0304", "n\u0303", "u\u0327\u0301",
           "A\u030A", "c\u0327", "\u1e0b\u0323", "s\u0323\u0307"]
PRECOMPOSED = "éàöñǹÅçḋṣ"
WS = [" ", "\t", "\n", "\r", "\r\n", "\v", "\f", "\u0085", "\u00a0", "\u1680",
      "\u2000", "\u2001", "\u2002", "\u2003", "\u2004", "\u2005", "\u2006",
      "\u2007", "\u2008", "\u2009", "\u200a", "\u2028", "\u2029", "\u202f",
      "\u205f", "\u3000"]
DIGITS = ["0", "7", "9", "\u0660", "\u06f5", "\u0966", "\uff10", "\u2070",
          "\u00b2", "\u2160", "\u2153", "\u00bd", "\u0e50", "\u17e0"]
MARKS = ["\u0301", "\u0300", "\u0302", "\u0308", "\u030a", "\u0327", "\u0334",
         "\u20d0", "\u20d1", "\ufe00", "\ufe0f", "\u0651", "\u064b", "\u093c"]
EMOJI = ["\U0001f600", "\U0001f469\u200d\U0001f4bb", "\U0001f468\u200d\U0001f469\u200d\U0001f467",
         "\U0001f3f3\ufe0f\u200d\U0001f308", "\U0001f1e8\U0001f1f3", "1\ufe0f\u20e3",
         "\U0001f44d\U0001f3fd", "\u2764\ufe0f", "\u263a\ufe0f", "\U0001f9d1\u200d\U0001f393"]
SPECIAL = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|vision_start|>",
           "<|vision_end|>", "<|image_pad|>", "<|video_pad|>", "<tool_call>",
           "</tool_call>", "<tool_response>", "</tool_response>", "<think>", "</think>"]
NEARMISS = ["<|im_", "<|im_end", "<|image_pa", "<|vision_star", "</tool_cal",
            "<|ends|>", "<tool_calls>", "<|IM_END|>", "<think", "<thinking>"]
CODE = [
    "def f(x):\n\treturn x*2\n",
    "if (a && b) { /* c */ }\n",
    "{\"k\": [1, 2, 3], \"n\": null}\n",
    "for i in range(10):\n    print(i)\r\n",
    "SELECT * FROM t WHERE x='y';\n",
    "#include <vector>\nint main(){}\n",
    "a\tb\tc\n1\t2\t3\n",
    "x = f'{a!r:>{w}}'\n",
]
TEMPLATE_TEXT = ("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                 "<|im_start|>user\nHello there<|im_end|>\n"
                 "<|im_start|>assistant\nHi<|im_end|>\n")

rng = random.Random(20260914)
corpus = []          # list of (category, text)


def add(cat, text):
    corpus.append((cat, text))


def build_corpus():
    # -- contractions / letter prefix rules
    for t in ["'s", "'t", "'re", "'ve", "'m", "'ll", "'d", "'S", "'LL", "'Re",
              "don't", "it's", "we'll", "I'M", "y'all'd've", "l's", "s's",
              "abcdef", " abc", "\tabc", "-abc", " abc def", "x abc", "1abc",
              "a1", "1a", "ab1cd", "\u0301abc", "\u00a0abc", "ABC", "aBc",
              "\u017fabc", "  abc", " \u0301abc"]:
        add("contract/letter", t)
    # -- digits (each digit is its own pre-token in this model)
    for n in range(1, 13):
        add("digits", "123456789012"[:n])
        add("digits", ("9" * n) + "a")
        add("digits", "a" + ("0" * n))
    for d in DIGITS:
        add("digits", d * 3)
        add("digits", "x" + d + "y")
        add("digits", d + "123")
    add("digits", "3.14159")
    add("digits", "1,234,567.89")
    add("digits", "\u0663\u0664\u0665")
    # -- whitespace behaviour
    for w in WS:
        add("ws", w)
        add("ws", w * 2)
        add("ws", w * 3)
        add("ws", "a" + w)
        add("ws", "a" + w + "b")
        add("ws", w + "a")
        add("ws", w + " a")
        add("ws", "a" + w + w + "b")
        add("ws", w + w + "a")
        add("ws", "\n\n" + w)
        add("ws", w + "\n")
    for t in ["  ", "   ", "a  b", "a   b", " \n \n", "\r\n\r\n", "a\r\nb",
              "\t \t", "x \t y", "\n\t\n", " " * 17, ("a " * 8).rstrip()]:
        add("ws", t)
    # -- NFC: combining sequences
    for t in ACCENTS:
        add("nfc", t)
        add("nfc", t * 2)
        add("nfc", "x" + t)
        add("nfc", t + "y")
        add("nfc", "a" + t + "b")
    for t in PRECOMPOSED:
        add("nfc", t)
        add("nfc", t + "\u0301")
    add("nfc", "\u1100\u1161")            # Hangul jamo -> 각
    add("nfc", "\u1100\u1161\u11a8")
    add("nfc", "\uac01")                   # already composed
    add("nfc", "\u1100")                   # lone jamo
    add("nfc", "\u212b")                   # angstrom -> Å via NFC
    add("nfc", "\u2126")                   # ohm -> Ω
    for t in ["\uff76", "\u2460", "\ufb01", "\u33a1", "\u2160", "\u00b5",
              "\u212b\uff76", "x\uff76y"]:
        add("nfc-compat(untouched)", t)
    add("nfc", "e\u0301\u0327")            # ordering: ccc 230 then 202
    add("nfc", "e\u0327\u0301")
    add("nfc", "q\u0307\u0323")
    add("nfc", "\u0f77")                   # singleton-ish canonical decomp
    # -- CJK / Hangul / Kana
    add("cjk", CJK)
    add("cjk", CJK[:1] * 5)
    add("cjk", "hello " + CJK + " world")
    add("cjk", CJK + "，。！？")
    add("cjk", "\u3001\u3002\uff01" + CJK[:3])
    add("kana", KATA)
    add("kana", HALFWIDTH_KATA)
    add("kana", HALFWIDTH_KATA + KATA)
    add("kana", "ｶﾞｷﾞ" + KATA[:3])
    add("hangul", HANGUL)
    add("hangul", "\u1100\u1101\u1102")
    # -- marks / invisible
    for m in MARKS:
        add("marks", m)
        add("marks", "a" + m)
        add("marks", "a" + m + m)
        add("marks", m * 4)
    add("marks", "a\u200db")
    add("marks", "\u200b\u200c\u200d")
    add("marks", "x\ufe0fy")
    add("marks", "ab\u0301\u0302\u0303\u0304")
    # -- emoji
    for e in EMOJI:
        add("emoji", e)
        add("emoji", e * 2)
        add("emoji", "hi " + e + " there")
        add("emoji", e + e)
    # -- special tokens (exact + near miss)
    for s in SPECIAL:
        add("special", s)
        add("special", s * 2)
        add("special", "x" + s)
        add("special", s + "y")
        add("special", "a " + s + " b")
    for s in NEARMISS:
        add("nearmiss", s)
        add("nearmiss", s + "x")
        add("nearmiss", "<|" + s)
    add("special", TEMPLATE_TEXT)
    add("special", TEMPLATE_TEXT.replace("<|im_end|>", "<|im_end"))
    add("special", "<|im_start|>user\n" + "A" * 40 + "<|im_end|>\n<|im_start|>assistant\n")
    # -- code / mixed
    for c in CODE:
        add("code", c)
        add("code", c * 3)
    add("code", "```python\n" + CODE[0] + "```\n")
    add("mixed", "ASCII " + CJK[:5] + " 123 " + ACCENTS[0] + " " + EMOJI[0])
    add("mixed", (TEMPLATE_TEXT + CODE[1] + CJK[:4]) * 2)
    # -- structured extras
    add("misc", "")
    add("misc", " ")
    add("misc", "\u0000")
    add("misc", "a\u0000b")
    add("misc", "\ufffd")
    add("misc", "\ufffd\ufffd")
    add("misc", "\u2028\u2029")
    add("misc", "A" * 500)
    add("misc", "ab" * 400)
    add("misc", " ".join(str(i) for i in range(200)))
    add("misc", "\u00df\u0130\u0131\u1e9e")
    add("misc", "\u0e33\u0eb3")
    add("misc", "中文English混排123")
    # -- mark adjacency (the class that exposed the raw-vs-AutoTokenizer split:
    #    marks are not part of the letter run, so mark+punct+letter re-splits)
    for m in ("\u0308", "\u0301", "\u0327", "\ufe0f", "\u20e3", "\u0651", "\u093c"):
        for p in ("_", ">", ".", "|", "<", "/", "-", "=", "~", "^"):
            for tail in ("s", "vaa", "abc", "i", "A", "x1", "\u4e2d"):
                add("mark-adjacency", m + p + tail)
                add("mark-adjacency", p + m + tail)
                add("mark-adjacency", m + p + tail + m)
                add("mark-adjacency", "a" + m + p + tail)
    # -- random unicode fuzz (>=200 required)
    pool = (CJK + HANGUL + KATA + HALFWIDTH_KATA + PRECOMPOSED + "".join(ACCENTS)
            + "".join(MARKS) + "".join(WS) + "".join(DIGITS) + "".join(EMOJI)
            + "".join(SPECIAL) + "abcXYZ019!?.,;:'\"()[]{}<>-_/\\|@#$%^&*+=~` \t\n\r")
    for _ in range(2600):
        n = rng.randint(1, 40)
        add("fuzz-alphabet", "".join(rng.choice(pool) for _ in range(n)))
    # random assigned codepoints
    ranges = [(0x20, 0x7E), (0xA0, 0x2FF), (0x300, 0x36F), (0x370, 0x3FF),
              (0x400, 0x4FF), (0x590, 0x5FF), (0x600, 0x6FF), (0x900, 0x97F),
              (0xE00, 0xE7F), (0x1100, 0x11FF), (0x1E00, 0x1EFF), (0x2000, 0x206F),
              (0x2070, 0x209F), (0x20A0, 0x20CF), (0x2100, 0x214F), (0x2190, 0x21FF),
              (0x2460, 0x24FF), (0x3000, 0x303F), (0x3040, 0x30FF), (0x3130, 0x318F),
              (0x4E00, 0x4FFF), (0xAC00, 0xD7A3), (0xF900, 0xFAFF), (0xFB00, 0xFB4F),
              (0xFE00, 0xFE0F), (0xFE30, 0xFE4F), (0xFF00, 0xFFEF),
              (0x1F300, 0x1F5FF), (0x1F600, 0x1F64F), (0x1F900, 0x1F9FF),
              (0x20000, 0x2005F)]
    for _ in range(2000):
        n = rng.randint(1, 30)
        s = []
        for _ in range(n):
            lo, hi = rng.choice(ranges)
            s.append(chr(rng.randint(lo, hi)))
        add("fuzz-codepoints", "".join(s))
    # multi-line realistic docs
    for _ in range(120):
        lines = []
        for _ in range(rng.randint(1, 8)):
            lines.append(rng.choice(CODE + [TEMPLATE_TEXT, CJK, " ".join(
                rng.choice(pool) for _ in range(rng.randint(1, 15)))]))
        add("docs", rng.choice(["\n", "\r\n", " "]).join(lines))


# ------------------------------------------------------------------ runner --

def run_cli(reqs):
    payload = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in reqs)
    p = subprocess.run([CLI, TOK_DIR], input=payload, capture_output=True,
                       text=True, encoding="utf-8")
    if p.returncode != 0:
        print(p.stderr[-4000:], file=sys.stderr)
        raise SystemExit(f"tok_cli exited {p.returncode}")
    out = [json.loads(l) for l in p.stdout.split("\n") if l.strip()]
    if len(out) != len(reqs):
        raise SystemExit(f"tok_cli returned {len(out)} lines for {len(reqs)} requests")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--show", type=int, default=10)
    args = ap.parse_args()

    from tokenizers import Tokenizer as HFTokenizer
    from transformers import AutoTokenizer

    raw = HFTokenizer.from_file(os.path.join(TOK_DIR, "tokenizer.json"))
    # The deployment's reference: transformers v5 rebuilds the pre-tokenizer from
    # its own class default (marks out of the letter run), which is what the live
    # Python service does -- confirmed by tools/probe_service.py prompt_tokens.
    ref_tok = AutoTokenizer.from_pretrained(TOK_DIR)
    print(f"reference: {type(ref_tok).__name__} (transformers AutoTokenizer)")

    build_corpus()
    items = corpus
    if args.limit:
        items = items[: args.limit]
    print(f"corpus: {len(items)} encode cases")

    reqs = [{"op": "encode", "text": t, "offsets": True} for _, t in items]
    got = run_cli(reqs)

    bad_ids = []
    bad_off = []
    raw_diff = 0
    cat_ids = {}
    for (cat, text), res in zip(items, got):
        enc = ref_tok(text, add_special_tokens=False, return_offsets_mapping=True)
        ref = list(enc["input_ids"])
        ref_off = [list(o) for o in enc["offset_mapping"]]
        if raw.encode(text, add_special_tokens=False).ids != ref:
            raw_diff += 1
        if "error" in res:
            bad_ids.append((cat, text, f"CLI error: {res['error']}", ref))
            cat_ids[cat] = cat_ids.get(cat, 0) + 1
            continue
        if res["ids"] != ref:
            bad_ids.append((cat, text, res["ids"], ref))
            cat_ids[cat] = cat_ids.get(cat, 0) + 1
        elif res.get("offsets") != ref_off:
            bad_off.append((cat, text, res.get("offsets"), ref_off))

    n = len(items)
    nid = n - len(bad_ids)
    print(f"encode ids : {nid}/{n} match ({100.0*nid/n:.4f}%)")
    if raw_diff:
        print(f"note       : {raw_diff} cases where raw tokenizer.json "
              f"disagrees with the AutoTokenizer reference")
    if bad_off:
        print(f"offsets    : {n - len(bad_off)}/{n} match "
              f"(soft; {len(bad_off)} differ)")
    else:
        print(f"offsets    : {n}/{n} match")
    if cat_ids:
        print("mismatch by category:", json.dumps(cat_ids, sort_keys=True))

    def show(lst, title, k):
        if not lst:
            return
        print(f"\n--- first {min(k, len(lst))} {title} ---")
        for cat, text, a, b in lst[:k]:
            print(f"[{cat}] {text!r}")
            print(f"    cli={a}")
            print(f"    ref={b}")
    show(bad_ids, "id mismatches", args.show)
    if args.verbose:
        show(bad_off, "offset mismatches", args.show)

    # ------------------------------------------------------------- decode --
    vocab = ref_tok.get_vocab()
    ids_pool = sorted(set(vocab.values()))
    max_id = max(ids_pool)
    specials = set(getattr(ref_tok, "added_tokens_decoder", {}).keys())
    dreqs = []
    dmeta = []
    declist = []
    for k in range(1, 13):
        for parity in (0, 1):
            seq = [i for i in ids_pool if i % 2 == parity][: k]
            if len(seq) == k:
                declist.append(seq)
    for _ in range(400):
        k = rng.randint(1, 16)
        declist.append([rng.choice(ids_pool) for _ in range(k)])
    for _ in range(200):
        k = rng.randint(1, 8)
        mixed = []
        for _ in range(k):
            mixed.append(rng.choice(list(specials)) if specials and rng.random() < 0.4
                         else rng.choice(ids_pool))
        declist.append(mixed)
    for _ in range(60):
        declist.append([rng.randint(0, max_id) for _ in range(rng.randint(1, 8))])

    for seq in declist:
        for skip in (True, False):
            dreqs.append({"op": "decode", "ids": seq, "skip_special_tokens": skip})
            dmeta.append((seq, skip))
    dgot = run_cli(dreqs)
    bad_dec = []
    for (seq, skip), res in zip(dmeta, dgot):
        try:
            ref = ref_tok.decode(seq, skip_special_tokens=skip)
        except Exception:  # noqa: BLE001
            continue
        if "error" in res:
            bad_dec.append((seq, skip, f"CLI error: {res['error']}", ref))
        elif res["text"] != ref:
            bad_dec.append((seq, skip, res["text"], ref))
    tot = len(dmeta)
    print(f"\ndecode     : {tot - len(bad_dec)}/{tot} match "
          f"({100.0*(tot-len(bad_dec))/tot:.4f}%)")
    if bad_dec:
        print(f"--- first {args.show} decode mismatches ---")
        for seq, skip, a, b in bad_dec[: args.show]:
            print(f"skip={skip} ids={seq}\n    cli={a!r}\n    ref={b!r}")

    ok = not bad_ids and not bad_dec
    print("\nRESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
