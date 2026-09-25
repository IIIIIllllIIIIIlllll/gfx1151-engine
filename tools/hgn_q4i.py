#!/usr/bin/env python3
"""hgn_q4i.py — rebuild the w4b base .hgn with imatrix-weighted q4cp routed
experts (same format, same sizes, same codebooks; tools/q4cp_imat.py).

Every tensor except the trunk routed experts (layers.N.mlp.experts.*) is
copied byte-for-byte from the old base (PLE fp8 table, embed, norms, the
q4cp dense fallbacks, the MTP q4cp fallbacks). The trunk experts are
requantized from the original BF16 safetensors with per-group scale search
weighted by a llama.cpp imatrix (GGUF format, e.g. imatrix_unsloth; tensors
blk.N.ffn_{gate,down}_exps.weight.{in_sum2,counts}, per expert).

  PYTHONPATH=~/Workspace/pylib python3 tools/hgn_q4i.py MODEL_DIR \
      --old-base models/qwen38-flash-next-w4b.hgn \
      --imatrix ~/Models/BF16/.../imatrix_unsloth.gguf_file \
      --out ~/Models/hq/qwen38-flash-next-w4b-imat.hgn [--jobs 28] [--layers 0,1]

--layers limits the requant to some layers (others copied) for quick tests.
The output is written to <out>.part and renamed at the end; it never writes
into a directory named models/.
"""

import argparse
import multiprocessing as mp
import os
import re
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from flashnext2hgn import DT_Q4CP, HgnWriter, STFile, human, q4cp_size  # noqa: E402
from gguf_mini import GGUF  # noqa: E402
from hgn_hq import read_hgn  # noqa: E402
import q4cp_imat  # noqa: E402

EXP_RE = re.compile(r"^layers\.(\d+)\.mlp\.experts\.(gate_up_proj|down_proj)\.weight$")
CHUNK = 256 << 20
DONTNEED = getattr(os, "POSIX_FADV_DONTNEED", None)

# ---- worker ------------------------------------------------------------------
_W = {}


def _winit(model_dir, imatrix):
    import json
    idx = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))
    _W["map"] = idx["weight_map"]
    _W["dir"] = model_dir
    _W["st"] = {}
    _W["im"] = GGUF(imatrix)
    _W["imc"] = {}


def _imat(layer, kind):
    key = (layer, kind)
    if key not in _W["imc"]:
        im = _W["im"]
        s = im.array(f"blk.{layer}.ffn_{kind}_exps.weight.in_sum2").astype(np.float32)
        c = im.array(f"blk.{layer}.ffn_{kind}_exps.weight.counts").astype(np.float32).reshape(-1)
        v = s / np.maximum(c, 1)[:, None]
        good = c > 0
        if not good.all():
            v[~good] = v[good].mean(axis=0)
        _W["imc"] = {key: v}  # one layer at a time is enough
    return _W["imc"][key]


def _wjob(args):
    src, layer, kind, e0, e1, R, C, cb = args
    fn = _W["map"][src]
    st = _W["st"].get(fn)
    if st is None:
        st = _W["st"][fn] = STFile(os.path.join(_W["dir"], fn))
    _dt, _sh, raw = st.raw(src)
    u16 = np.frombuffer(raw, np.uint16)
    iv = _imat(layer, kind)
    codes, scales = [], []
    for e in range(e0, e1):
        w = (u16[e * R * C:(e + 1) * R * C].astype(np.uint32) << 16).view(np.float32)
        g = w.reshape(-1, 32)
        qw = np.broadcast_to(iv[e].reshape(1, C), (R, C)).reshape(-1, 32)
        nib, s16 = q4cp_imat.quant_groups(g, qw, cb)
        c, s = q4cp_imat.pack(nib, s16, R, C)
        codes.append(c)
        scales.append(s)
    if DONTNEED is not None:
        a = st.hdr[src]["data_offsets"][0] + st.base
        os.posix_fadvise(st.fd, a + e0 * R * C * 2, (e1 - e0) * R * C * 2, DONTNEED)
    return b"".join(codes), b"".join(scales)


# ---- main --------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model_dir")
    ap.add_argument("--old-base", required=True)
    ap.add_argument("--imatrix", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 4) - 4))
    ap.add_argument("--layers", default="", help="comma list of layers to requant (default all)")
    ap.add_argument("--per-job", type=int, default=4, help="experts per worker task")
    a = ap.parse_args()

    out = os.path.abspath(os.path.expanduser(a.out))
    if os.path.basename(os.path.dirname(out)) == "models":
        raise SystemExit("refusing to write into a models/ directory; pick another path")
    t_start = time.time()
    import json
    wmap = json.load(open(os.path.join(a.model_dir, "model.safetensors.index.json")))["weight_map"]
    old = read_hgn(a.old_base)
    im = GGUF(a.imatrix)
    only = {int(x) for x in a.layers.split(",") if x}

    names = sorted(old, key=lambda n: old[n][2])  # file order
    todo = {}
    for n in names:
        m = EXP_RE.match(n)
        if not m:
            continue
        layer = int(m.group(1))
        if only and layer not in only:
            continue
        dt, dims, off, size = old[n]
        if dt != DT_Q4CP or len(dims) != 3:
            raise SystemExit(f"{n}: unexpected dtype {dt} dims {dims}")
        src = "model.language_model." + n[: -len(".weight")]
        if src not in wmap:
            raise SystemExit(f"{n}: {src} not in safetensors index")
        kind = "gate" if m.group(2) == "gate_up_proj" else "down"
        for suf in ("in_sum2", "counts"):
            if f"blk.{layer}.ffn_{kind}_exps.weight.{suf}" not in im.tensors:
                raise SystemExit(f"imatrix lacks blk.{layer}.ffn_{kind}_exps.weight.{suf}")
        E, R, C = dims
        if q4cp_size(E * R, C)[0] != size:
            raise SystemExit(f"{n}: size mismatch vs q4cp_size")
        todo[n] = (src, layer, kind, E, R, C)
    print(f"old base {len(old)} tensors; requant {len(todo)} expert tensors "
          f"({human(sum(old[n][3] for n in todo))}), copy {len(old) - len(todo)} "
          f"({human(sum(old[n][3] for n in old if n not in todo))}); jobs {a.jobs}", flush=True)

    os.makedirs(os.path.dirname(out), exist_ok=True)
    tmp = out + ".part"
    # identical layout: header + records copied verbatim, every blob at its old offset
    fo = open(a.old_base, "rb")
    first = min(old[n][2] for n in names)
    fout = open(tmp, "wb")
    fout.write(fo.read(first))
    pool = mp.Pool(a.jobs, initializer=_winit, initargs=(a.model_dir, a.imatrix))
    werr = []
    done = 0
    total = sum(old[n][3] for n in names)

    def flush_out():
        fout.flush()
        if DONTNEED is not None:
            os.fdatasync(fout.fileno())
            os.posix_fadvise(fout.fileno(), 0, 0, DONTNEED)

    for i, n in enumerate(names):
        dt, dims, off, size = old[n]
        t0 = time.time()
        if fout.tell() < off:
            fout.write(b"\0" * (off - fout.tell()))
        start = fout.tell()
        assert start == off, f"{n}: at {start}, old offset {off}"
        if n not in todo:
            left, p = size, off
            while left:
                k = min(CHUNK, left)
                fo.seek(p)
                buf = fo.read(k)
                assert len(buf) == k
                fout.write(buf)
                if DONTNEED is not None:
                    os.posix_fadvise(fo.fileno(), p, k, DONTNEED)
                p += k
                left -= k
                if k == CHUNK:
                    flush_out()
            kind = "copy"
        else:
            src, layer, kname, E, R, C = todo[n]
            fo.seek(off)
            cbb = fo.read(64)
            cb = np.frombuffer(cbb, np.float32).copy()
            fout.write(cbb)
            jobs = [(src, layer, kname, e, min(E, e + a.per_job), R, C, cb)
                    for e in range(0, E, a.per_job)]
            scales = []
            for c, s in pool.imap(_wjob, jobs):
                fout.write(c)
                scales.append(s)
            for s in scales:
                fout.write(s)
            del scales
            kind = f"q4i L{layer}"
            # spot check expert 0: weighted error new vs old
            if layer in (0, 23, 47) or only:
                fout.flush()
                werr.append((n, spot_check(a.model_dir, wmap, src, im, layer, kname, old[n],
                                           fo, tmp, start, R, C, E)))
                print(f"    check {n}: w-err old {werr[-1][1][0]:.4e} new {werr[-1][1][1]:.4e}",
                      flush=True)
            flush_out()
        wrote = fout.tell() - start
        assert wrote == size, f"{n}: wrote {wrote} != {size}"
        done += size
        el = time.time() - t0
        if kind != "copy" or size > (1 << 30) or i % 200 == 0:
            eta = (time.time() - t_start) / done * (total - done) / 60
            print(f"  [{i + 1}/{len(names)}] {kind} {n} {human(size)} {el:.1f}s "
                  f"({100 * done / total:.0f}%, eta {eta:.0f} min)", flush=True)
    pool.close()
    pool.join()
    oldsize = os.path.getsize(a.old_base)
    if fout.tell() < oldsize:
        fout.write(b"\0" * (oldsize - fout.tell()))
    flush_out()
    fout.close()
    if os.path.getsize(tmp) != os.path.getsize(a.old_base):
        raise SystemExit(f"FAIL: size {os.path.getsize(tmp)} != old {os.path.getsize(a.old_base)}")
    bad = [n for n, (eo, en) in werr if not en < eo]
    if bad:
        raise SystemExit(f"FAIL: weighted error did not improve on {bad}")
    os.replace(tmp, out)
    print(f"wrote {out} ({human(os.path.getsize(out))}) in "
          f"{(time.time() - t_start) / 60:.1f} min", flush=True)


def spot_check(model_dir, wmap, src, im, layer, kind, rec, fo, newpath, newoff, R, C, E):
    """Weighted rel error (imatrix) of expert 0 and E-1: old blob vs new blob."""
    _dt, _dims, off, _size = rec
    st = STFile(os.path.join(model_dir, wmap[src]))
    u16 = np.frombuffer(st.raw(src)[2], np.uint16)
    s = im.array(f"blk.{layer}.ffn_{kind}_exps.weight.in_sum2").astype(np.float32)
    c = im.array(f"blk.{layer}.ffn_{kind}_exps.weight.counts").astype(np.float32).reshape(-1)
    stride = ((C // 32 * 2) + 15) & ~15
    fo.seek(off)
    cb = np.frombuffer(fo.read(64), np.float32).copy()
    res = [0.0, 0.0]
    fn = open(newpath, "rb")
    for e in (0, E - 1):
        x = (u16[e * R * C:(e + 1) * R * C].astype(np.uint32) << 16).view(np.float32).reshape(R, C)
        qw = (s[e] / max(c[e], 1)).reshape(1, C) if c[e] > 0 else np.ones((1, C), np.float32)
        for k, (f, base) in enumerate(((fo, off), (fn, newoff))):
            f.seek(base + 64 + e * R * C // 2)
            cd = np.frombuffer(f.read(R * C // 2), np.uint8).reshape(R, C // 2)
            f.seek(base + 64 + E * R * C // 2 + e * R * stride)
            sc = np.frombuffer(f.read(R * stride), np.uint8).reshape(R, stride)
            sc = sc[:, : C // 32 * 2].copy().view(np.float16).astype(np.float32)
            nib = np.empty((R, C), np.uint8)
            nib[:, 0::2] = cd & 15
            nib[:, 1::2] = cd >> 4
            d = (cb[nib].reshape(R, C // 32, 32) * sc[:, :, None]).reshape(R, C)
            res[k] += float(((d - x) ** 2 * qw).sum() / ((x * x) * qw).sum())
    fn.close()
    return res[0] / 2, res[1] / 2


if __name__ == "__main__":
    main()
