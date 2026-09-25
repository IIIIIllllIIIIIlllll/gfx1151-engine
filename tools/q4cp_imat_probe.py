"""Probe: old q4cp (w4b.hgn) vs imatrix-weighted q4cp on a few experts.
  PYTHONPATH=~/Workspace/pylib:tools python3 tools/q4cp_imat_probe.py MODEL_DIR BASE.hgn IMATRIX
Prints per tensor: old blob == naive replica?, imatrix-weighted rel error
old / new(no imat) / new(imat), plain rel-L2, time."""
import sys
import time

import numpy as np

from flashnext2hgn import STFile, scan_model_dir  # noqa: F401
from gguf_mini import GGUF
from hgn_hq import read_hgn
from q4cp_imat import dequant, naive_groups, quant_groups

model_dir, base, imf = sys.argv[1:4]
table = scan_model_dir(model_dir)
old = read_hgn(base)
im = GGUF(imf)
fo = open(base, "rb")


def imat(layer, kind):
    s = im.array(f"blk.{layer}.ffn_{kind}_exps.weight.in_sum2").astype(np.float32)
    c = im.array(f"blk.{layer}.ffn_{kind}_exps.weight.counts").astype(np.float32).reshape(-1, 1)
    v = s / np.maximum(c, 1)
    good = c.ravel() > 0
    if not good.all():
        v[~good] = v[good].mean(axis=0)
    return v, int((~good).sum())


for layer in (0, 23, 47):
    for t, kind in (("gate_up_proj", "gate"), ("down_proj", "down")):
        n = f"layers.{layer}.mlp.experts.{t}.weight"
        st, dt, shape = table["model.language_model." + n[:-len(".weight")]]
        E, R, C = shape
        dto, dims, off, size = old[n]
        fo.seek(off)
        cb = np.frombuffer(fo.read(64), np.float32).copy()
        stride = ((C // 32 * 2) + 15) & ~15
        iv, nzero = imat(layer, kind)
        _dt, _sh, raw = st.raw("model.language_model." + n[:-len(".weight")])
        u16 = np.frombuffer(raw, np.uint16)
        res = []
        for e in (0, 1, 100, 311, 511):
            w = (u16[e * R * C:(e + 1) * R * C].astype(np.uint32) << 16).view(np.float32).reshape(-1, 32)
            qw = np.broadcast_to(iv[e].reshape(1, C), (R, C)).reshape(-1, 32)
            # old blob rows for expert e
            fo.seek(off + 64 + e * R * C // 2)
            oc = np.frombuffer(fo.read(R * C // 2), np.uint8).reshape(R, C // 2)
            fo.seek(off + 64 + E * R * C // 2 + e * R * stride)
            os_ = np.frombuffer(fo.read(R * stride), np.uint8).reshape(R, stride)
            os16 = os_[:, : C // 32 * 2].copy().view(np.float16).reshape(-1)
            onib = np.empty((R, C), np.uint8)
            onib[:, 0::2] = oc & 15
            onib[:, 1::2] = oc >> 4
            onib = onib.reshape(-1, 32)
            nn, ns = naive_groups(w, cb)
            same = np.array_equal(nn, onib) and np.array_equal(ns.view(np.uint16), os16.view(np.uint16))
            t1 = time.time()
            n0, s0 = quant_groups(w, None, cb)
            n1, s1 = quant_groups(w, qw, cb)
            tq = time.time() - t1
            def werr(d):
                return float(((d - w) ** 2 * qw).sum() / ((w ** 2) * qw).sum())
            def perr(d):
                return float(((d - w) ** 2).sum() / (w ** 2).sum())
            do, d0, d1 = dequant(onib, os16, cb), dequant(n0, s0, cb), dequant(n1, s1, cb)
            res.append((same, werr(do), werr(d0), werr(d1), perr(do), perr(d1), tq))
        r = np.array([[float(v) for v in row] for row in res])
        print(f"{n} {shape} zero-count experts={nzero} replica={all(row[0] for row in res)}  "
              f"w-err old {r[:,1].mean():.4e} new-noimat {r[:,2].mean():.4e} new-imat {r[:,3].mean():.4e} "
              f"(x{r[:,1].mean()/r[:,3].mean():.2f})  plain old {r[:,4].mean():.4e} new {r[:,5].mean():.4e}  "
              f"{r[:,6].mean():.2f}s/expert", flush=True)
