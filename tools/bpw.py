#!/usr/bin/env python3
"""bpw.py — bits per weight of hgn and GGUF models, by category (headers only,
a few seconds, read-only).

  python3 tools/bpw.py                         # default set (see DEFAULTS)
  python3 tools/bpw.py A.hgn,A.overlay.hgn  X-00001-of-00004.gguf  M-mtp.hgn

Each argument is one model. hgn: comma-separated files loaded in order, later
files replace tensors of the same name (base, then overlay — as the engine
does). GGUF: give the first shard, the others are found automatically.

bpw = stored bytes * 8 / number of weights. Weights are counted from the
logical tensor shapes (hgn dims, GGUF ne), so both formats are counted the
same way; per-category parameter counts are compared across models to prove
the same weights are being compared. mtp.* tensors inside a main model (hgn
base keeps a 4-bit fallback draft head) are left out; pass the MTP sidecar
as its own argument to measure it. i64 metadata tensors are not weights.
GGUF byte sizes come from the ggml type table and are checked against the
tensor offsets in the file.
"""

import glob
import os
import re
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from gguf_mini import GGUF  # noqa: E402

HGN_DT = {0: "bf16", 4: "i64", 5: "q4cp", 7: "q8g64", 8: "q8g32", 10: "fp8"}
# ggml type id -> (name, block elements, block bytes)
GGML = {0: ("F32", 1, 4), 1: ("F16", 1, 2), 2: ("Q4_0", 32, 18), 3: ("Q4_1", 32, 20),
        6: ("Q5_0", 32, 22), 7: ("Q5_1", 32, 24), 8: ("Q8_0", 32, 34), 9: ("Q8_1", 32, 36),
        10: ("Q2_K", 256, 84), 11: ("Q3_K", 256, 110), 12: ("Q4_K", 256, 144),
        13: ("Q5_K", 256, 176), 14: ("Q6_K", 256, 210), 15: ("Q8_K", 256, 292),
        16: ("IQ2_XXS", 256, 66), 17: ("IQ2_XS", 256, 74), 18: ("IQ3_XXS", 256, 98),
        19: ("IQ1_S", 256, 50), 20: ("IQ4_NL", 32, 18), 21: ("IQ3_S", 256, 110),
        22: ("IQ2_S", 256, 82), 23: ("IQ4_XS", 256, 136), 24: ("I8", 1, 1),
        25: ("I16", 1, 2), 26: ("I32", 1, 4), 27: ("I64", 1, 8), 28: ("F64", 1, 8),
        29: ("IQ1_M", 256, 56), 30: ("BF16", 1, 2)}

CATS = ["routed experts", "dense linear", "embed + lm_head", "PLE n-gram table",
        "norm / 1-D / conv"]


def category(name, ndim):
    if "mlp.experts." in name or "_exps." in name:
        return CATS[0]
    if "ple.ngram_embedding" in name or name.startswith("per_layer_token_embd"):
        return CATS[3]
    if name.split(".")[0] in ("embed_tokens", "lm_head", "token_embd", "output") and \
            name.endswith(".weight") and name.count(".") == 1:
        return CATS[2]
    if ndim == 1 or "conv" in name.lower() or "norm" in name or name.endswith("ssm_a") \
            or name.endswith(".bias"):
        return CATS[4]
    return CATS[1]


def numel(dims):
    n = 1
    for x in dims:
        n *= int(x)
    return n


def read_hgn(path):
    out = {}
    with open(path, "rb") as f:
        h = f.read(104)
        magic, _v, cnt, _r, ro, _do, _fs = struct.unpack_from("<4sIIIQQQ", h, 0)
        f.seek(ro)
        for _ in range(cnt):
            r = f.read(160)
            n = r[:96].split(b"\0")[0].decode()
            dt, nd = struct.unpack_from("<II", r, 96)
            dims = struct.unpack_from("<4Q", r, 104)[:nd]
            _o, sz, _x = struct.unpack_from("<QQQ", r, 136)
            out[n] = (HGN_DT.get(dt, f"dt{dt}"), dims, sz)
    return out


def load_hgn(files):
    t = {}
    for p in files:
        t.update(read_hgn(p))  # later files replace by name (overlay)
    return {n: v for n, v in t.items() if v[0] != "i64"}, sum(os.path.getsize(p) for p in files)


def shards(first):
    m = re.match(r"^(.*-)00001-of-(\d{5})\.gguf$", first)
    if not m:
        return [first]
    return [f"{m.group(1)}{i:05d}-of-{m.group(2)}.gguf" for i in range(1, int(m.group(2)) + 1)]


def load_gguf(first):
    t, disk, bad = {}, 0, []
    for p in shards(first):
        g = GGUF(p)
        fsz = os.path.getsize(p)
        disk += fsz
        al = int(g.kv.get("general.alignment", 32))
        offs = sorted((o, n) for n, (_d, _t, o) in g.tensors.items())
        nxt = {n: (offs[i + 1][0] if i + 1 < len(offs) else fsz) for i, (_o, n) in enumerate(offs)}
        for n, (dims, ty, o) in g.tensors.items():
            name, be, bb = GGML[ty]
            ne = numel(dims)
            sz = ne // be * bb
            gap = nxt[n] - o - sz
            if ne % be or gap < 0 or (nxt[n] < fsz and gap >= al):
                bad.append(n)
            t[n] = (name, tuple(dims), sz)
        g.mm.close()
        g.f.close()
    if bad:
        print(f"  WARNING: {len(bad)} GGUF tensor sizes disagree with offsets: {bad[:3]}")
    return t, disk


def summarize(label, tens, disk):
    has_main = any(not n.startswith("mtp.") for n in tens)
    rows = {c: [0, 0, {}] for c in CATS}
    skipped = 0
    for n, (dt, dims, sz) in tens.items():
        if has_main and n.startswith("mtp."):
            skipped += 1
            continue
        r = rows[category(n, len(dims))]
        ne = numel(dims)
        r[0] += ne
        r[1] += sz
        d = r[2].setdefault(dt, [0, 0])
        d[0] += ne
        d[1] += sz
    print(f"\n== {label}")
    print(f"   {'category':<20} {'params':>9} {'GiB':>8} {'bpw':>7}   types (share of params: bpw)")
    tp = tb = 0
    for c in CATS:
        ne, sz, d = rows[c]
        if not ne:
            continue
        tp += ne
        tb += sz
        ty = ", ".join(f"{k} {100 * v[0] / ne:.0f}%: {8 * v[1] / v[0]:.2f}"
                       for k, v in sorted(d.items(), key=lambda kv: -kv[1][0]))
        print(f"   {c:<20} {ne / 1e9:8.3f}B {sz / 2**30:8.2f} {8 * sz / ne:7.3f}   {ty}")
    pp, pb = rows[CATS[3]][0], rows[CATS[3]][1]
    print(f"   {'TOTAL':<20} {tp / 1e9:8.3f}B {tb / 2**30:8.2f} {8 * tb / tp:7.3f}")
    if pp and tp > pp:
        print(f"   {'TOTAL without PLE':<20} {(tp - pp) / 1e9:8.3f}B {(tb - pb) / 2**30:8.2f} "
              f"{8 * (tb - pb) / (tp - pp):7.3f}   (the part that lives in GPU memory)")
    print(f"   files on disk {disk / 2**30:.2f} GiB"
          + (f"; {skipped} mtp.* tensors not counted (fallback draft head)" if skipped else "")
          + (f"; {(disk - tb) / 2**30:.2f} GiB of it replaced by overlay / headers" if disk - tb > 2**28 else ""))
    return {c: rows[c][0] for c in CATS}


def main():
    root = os.path.dirname(HERE)
    hq = os.path.expanduser("~/Models/hq")
    DEFAULTS = [
        ("hgn production (w4b + overlay)",
         [f"{root}/models/qwen38-flash-next-w4b.hgn", f"{root}/models/qwen38-flash-next-w4b.overlay.hgn"]),
        ("hgn HQ (w4b-imat + overlay-q8)",
         [f"{hq}/qwen38-flash-next-w4b-imat.hgn", f"{hq}/qwen38-flash-next-w4b.overlay-q8.hgn"]),
        ("GGUF UD-Q4_K_XL", [f"{root}/models/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf"]),
    ]
    if len(sys.argv) > 1:
        models = [(a, a.split(",")) for a in sys.argv[1:]]
    else:
        models = [(l, f) for l, f in DEFAULTS if all(os.path.exists(p) for p in f)]
        for l, f in DEFAULTS:
            if (l, f) not in models:
                print(f"skip {l}: missing {[p for p in f if not os.path.exists(p)]}")
    counts = []
    for label, files in models:
        for p in files:
            if not os.path.exists(p):
                raise SystemExit(f"missing {p}")
        if files[0].endswith(".gguf"):
            tens, disk = load_gguf(files[0])
        else:
            tens, disk = load_hgn(files)
        counts.append((label, summarize(label, tens, disk)))
    if len(counts) > 1:
        print("\n== same weights? parameter count per category vs the first model")
        ref_l, ref = counts[0]
        ok = True
        for label, c in counts[1:]:
            for k in CATS:
                if ref[k] or c[k]:
                    d = (c[k] - ref[k]) / max(ref[k], 1)
                    flag = "" if abs(d) < 0.01 else "   <-- differs"
                    ok &= abs(d) < 0.01 or k == CATS[4]
                    print(f"   {label[:28]:<28} {k:<20} {c[k] / 1e9:8.3f}B vs {ref[k] / 1e9:8.3f}B "
                          f"({100 * d:+.2f}%){flag}")
        print("   " + ("OK: same weights (norm/1-D may differ by layout)" if ok
                      else "NOTE: some categories differ, bpw per category still valid"))


if __name__ == "__main__":
    main()
