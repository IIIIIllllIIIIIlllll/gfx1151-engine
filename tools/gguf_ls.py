#!/usr/bin/env python3
"""Pure-python GGUF header lister (no numpy).
  gguf_ls.py FILE.gguf [--kv] [--tensors] [--summary] [--grep REGEX]
Multi-shard files: pass the first shard; the others (-0000N-of-0000M) are read too.
"""
import re, struct, sys, os, glob, collections

GT = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 9: "Q8_1",
      10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS",
      17: "IQ2_XS", 18: "IQ3_XXS", 19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S",
      23: "IQ4_XS", 24: "I8", 25: "I16", 26: "I32", 27: "I64", 28: "F64", 29: "IQ1_M",
      30: "BF16", 34: "TQ1_0", 35: "TQ2_0", 39: "MXFP4"}
# (block elements, block bytes)
BS = {"F32": (1, 4), "F16": (1, 2), "BF16": (1, 2), "Q4_0": (32, 18), "Q4_1": (32, 20),
      "Q5_0": (32, 22), "Q5_1": (32, 24), "Q8_0": (32, 34), "Q2_K": (256, 84), "Q3_K": (256, 110),
      "Q4_K": (256, 144), "Q5_K": (256, 176), "Q6_K": (256, 210), "IQ4_NL": (32, 18),
      "IQ4_XS": (256, 136), "IQ3_XXS": (256, 98), "IQ3_S": (256, 110), "IQ2_XXS": (256, 66),
      "IQ2_XS": (256, 74), "IQ2_S": (256, 82), "IQ1_S": (256, 50), "IQ1_M": (256, 56),
      "I8": (1, 1), "I16": (1, 2), "I32": (1, 4), "MXFP4": (32, 17)}


class R:
    def __init__(s, f): s.f = f
    def u(s, fmt): n = struct.calcsize(fmt); return struct.unpack("<" + fmt, s.f.read(n))[0]
    def str(s): n = s.u("Q"); return s.f.read(n).decode("utf-8", "replace")
    def val(s, t):
        if t == 8: return s.str()
        if t == 9:
            at = s.u("I"); n = s.u("Q")
            if at in (8, 9) or n > 64:  # skip big arrays (tokens etc.)
                out = [s.val(at) for _ in range(n)]
                return f"<array {n} x t{at}>" if n > 64 else out
            return [s.val(at) for _ in range(n)]
        return s.u({0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q",
                    11: "q", 12: "d"}[t])


def read(path):
    with open(path, "rb") as f:
        r = R(f)
        assert f.read(4) == b"GGUF", path
        ver = r.u("I"); nt = r.u("Q"); nkv = r.u("Q")
        kv = {}
        for _ in range(nkv):
            k = r.str(); t = r.u("I"); kv[k] = r.val(t)
        align = kv.get("general.alignment", 32)
        ts = []
        for _ in range(nt):
            name = r.str(); nd = r.u("I"); dims = [r.u("Q") for _ in range(nd)]
            typ = GT.get(r.u("I"), "?"); off = r.u("Q")
            ts.append((name, dims, typ, off))
        base = (f.tell() + align - 1) // align * align
    return ver, kv, ts, base


def main():
    a = sys.argv[1:]
    if not a: print(__doc__); sys.exit(1)
    path = a[0]
    m = re.match(r"(.*)-(\d{5})-of-(\d{5})\.gguf$", path)
    files = sorted(glob.glob(m.group(1) + "-*-of-" + m.group(3) + ".gguf")) if m else [path]
    rx = re.compile(a[a.index("--grep") + 1]) if "--grep" in a else None
    allt = []
    for p in files:
        ver, kv, ts, base = read(p)
        if "--kv" in a and p == files[0]:
            for k, v in kv.items():
                if "tokenizer.ggml.tokens" in k or "merges" in k or "token_type" in k: continue
                print(f"KV {k} = {v}")
        allt += [(n, d, t, o, os.path.basename(p)) for n, d, t, o in ts]
    tot = collections.Counter(); cnt = collections.Counter()
    for n, d, t, o, fn in allt:
        ne = 1
        for x in d: ne *= x
        be, bb = BS.get(t, (1, 0))
        nb = ne // be * bb
        # group key: strip block index
        g = re.sub(r"^blk\.\d+\.", "blk.N.", n)
        tot[(g, t)] += nb; cnt[(g, t)] += 1
        if "--tensors" in a and (rx is None or rx.search(n)):
            print(f"T {n:48s} {t:7s} {'x'.join(map(str, d)):>22s} {nb/2**20:10.2f} MiB")
    if "--summary" in a or "--tensors" not in a:
        byt = collections.Counter()
        for (g, t), b in sorted(tot.items()):
            if rx is None or rx.search(g):
                print(f"S {g:48s} {t:7s} n={cnt[(g, t)]:3d} {b/2**30:8.3f} GiB")
            byt[t] += b
        print("BYTYPE", {t: round(b / 2**30, 2) for t, b in byt.items()},
              "total", round(sum(byt.values()) / 2**30, 2), "GiB, tensors", len(allt))


if __name__ == "__main__":
    main()
