#!/usr/bin/env python3
# qwentok.py — minimal Qwen2 BPE tokenizer (stdlib only): encode + decode.
# Replicates tokenizer.json: NFC -> split regex -> ByteLevel -> BPE merges,
# plus literal added-token matching.
import json, os, unicodedata

DIR = os.environ.get("TOK_DIR", "models/tokenizer")

def _bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + \
         list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))

_B2U = _bytes_to_unicode()
_U2B = {v: k for k, v in _B2U.items()}

def _cls(ch):
    c = unicodedata.category(ch)
    if c[0] == "L": return "L"
    if c[0] == "M": return "M"
    if c[0] == "N": return "N"
    if ch in " \t\n\r\x0b\x0c": return "S"
    return "O"  # other (punct, symbols...)

_CONTRACTIONS = ("'s", "'t", "'re", "'ve", "'m", "'ll", "'d")

def _split(text):
    # (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}|
    #  ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
    # emulated with backtracking (leftmost-first) semantics per alternative
    out, i, n = [], 0, len(text)
    while i < n:
        # 1. contractions
        low = text[i:i + 4].lower()
        m = None
        for c in _CONTRACTIONS:
            if low.startswith(c):
                m = text[i:i + len(c)]
                break
        # 2. [^\r\nLN]?[LM]+
        if m is None:
            j = i
            if _cls(text[j]) not in "LN" and text[j] not in "\r\n" and \
               j + 1 < n and _cls(text[j + 1]) in "LM":
                j += 1
            if _cls(text[j]) in "LM":
                while j < n and _cls(text[j]) in "LM":
                    j += 1
                m = text[i:j]
        # 3. \p{N} (single char)
        if m is None and _cls(text[i]) == "N":
            m = text[i:i + 1]
        # 4. ' '?[^\sLMN]+[\r\n]*
        if m is None:
            j = i
            if text[j] == " " and j + 1 < n and _cls(text[j + 1]) == "O":
                j += 1
            if _cls(text[j]) == "O":
                while j < n and _cls(text[j]) == "O":
                    j += 1
                while j < n and text[j] in "\r\n":
                    j += 1
                m = text[i:j]
        # 5-7. whitespace: \s*[\r\n]+ | \s+(?!\S) | \s+
        if m is None:
            j = i
            while j < n and _cls(text[j]) == "S":
                j += 1
            run = text[i:j]
            if "\r" in run or "\n" in run:
                # through the last \r\n in the run
                k = max(run.rfind("\r"), run.rfind("\n"))
                m = run[:k + 1]
            elif j < n and len(run) >= 2:
                m = run[:-1]  # \s+(?!\S): leave last space for the next token
            else:
                m = run
        out.append(m)
        i += len(m)
    return out

class Tokenizer:
    def __init__(self, dir=DIR):
        tj = json.load(open(dir + "/tokenizer.json"))
        self.vocab = tj["model"]["vocab"]
        self.id2tok = {v: k for k, v in self.vocab.items()}
        self.ranks = {}
        for i, mg in enumerate(tj["model"]["merges"]):
            a, b = mg.split(" ")
            self.ranks[(a, b)] = i
        self.special = {}
        for t in tj["added_tokens"]:
            self.special[t["content"]] = t["id"]
        self.id2tok.update({v: k for k, v in self.special.items()})

    def _bpe(self, token):
        # token: byte-level unicode string
        seq = list(token)
        if len(seq) == 1:
            return seq
        while True:
            best, bi = None, -1
            for i in range(len(seq) - 1):
                r = self.ranks.get((seq[i], seq[i + 1]))
                if r is not None and (best is None or r < best):
                    best, bi = r, i
            if bi < 0:
                return seq
            seq[bi:bi + 2] = [seq[bi] + seq[bi + 1]]

    def encode(self, text):
        text = unicodedata.normalize("NFC", text)
        ids = []
        i, n = 0, len(text)
        specials = sorted(self.special, key=len, reverse=True)
        while i < n:
            hit = None
            for s in specials:
                if text.startswith(s, i):
                    hit = s
                    break
            if hit:
                ids.append(self.special[hit])
                i += len(hit)
                continue
            # take plain run until next special
            j = i
            while j < n and not any(text.startswith(s, j) for s in specials):
                j += 1
            for ptok in _split(text[i:j]):
                bl = "".join(_B2U[b] for b in ptok.encode("utf-8"))
                for t in self._bpe(bl):
                    ids.append(self.vocab[t])
            i = j
        return ids

    def decode(self, ids):
        out = bytearray()
        for i in ids:
            t = self.id2tok[i]
            if t in self.special or (t.startswith("<|") and t.endswith("|>")):
                out += t.encode("utf-8")
            else:
                out += bytes(_U2B[c] for c in t)
        return out.decode("utf-8", errors="replace")

if __name__ == "__main__":
    import sys
    tk = Tokenizer()
    if sys.argv[1] == "enc":
        print(tk.encode(sys.argv[2]))
    elif sys.argv[1] == "dec":
        print(repr(tk.decode([int(x) for x in sys.argv[2].split(",")])))
