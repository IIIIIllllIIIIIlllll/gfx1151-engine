#!/usr/bin/env python3
"""hgn_hq.py — build a high-quality dense overlay (.hgn) for the production
qwen38-flash-next-w4b.hgn base, straight from the original BF16 safetensors.

The production overlay (w4b.overlay.hgn) holds the dense/attention/HC/shared
expert/lm_head matrices as 4-bit q4cp. KLD bisect (tools/hq_probe.sh,
2026-09-26) showed those 4-bit dense weights carry ~85% of the hgn quality
gap: q4cp experts + 8-bit dense = KLD 0.063 vs 0.163. This tool rewrites
every tensor of the old overlay (plus optionally embed_tokens) as

  dtype 8  q8g32 planar: [rows*cols int8][rows*cols/32 fp16 scales]
           (= llama.cpp Q8_0 math: d = amax/127, q = round(x/d))
  dtype 0  bf16 passthrough for the few-row matrices the engine keeps in
           bf16 on the GGUF path too (in_proj_a/b, shared_expert_gate,
           block_inject_weight, indexer.index_qk_proj)

Tensors matching --q4-keep substrings are copied verbatim from the old
overlay instead (mixed precision). The output replaces the old overlay:

  gdec  w4b.hgn  <out>.hgn  [mtp.hgn]  ...

Usage:
  PYTHONPATH=~/Workspace/pylib python3 tools/hgn_hq.py MODEL_DIR \
      --old-overlay models/qwen38-flash-next-w4b.overlay.hgn \
      --out ~/Models/hq/qwen38-flash-next-w4b.overlay-q8.hgn \
      [--embed] [--q4-keep sub1,sub2] [--check N] [--dry-run]
"""

import argparse
import os
import struct
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from flashnext2hgn import (  # noqa: E402
    DT_BF16, DT_Q8G32, HDR, REC, HgnWriter, check_config, dequant_q8g32, human,
    map_lm_name, quant_q8g32, scan_model_dir)
from flashnext2hgn import OVL_BF16 as BF16_SMALL  # noqa: E402

import json  # noqa: E402


def read_hgn(path):
    """name -> (dtype, dims, offset, size) for an hgn file."""
    with open(path, "rb") as f:
        h = f.read(HDR)
        mag, _ver, cnt, _r, ro, _do, fs = struct.unpack_from("<4sIIIQQQ", h, 0)
        if mag != b"HGN1":
            raise SystemExit(f"{path}: bad magic")
        if fs != os.path.getsize(path):
            raise SystemExit(f"{path}: size mismatch")
        f.seek(ro)
        out = {}
        for _ in range(cnt):
            r = f.read(REC)
            name = r[:96].split(b"\0")[0].decode()
            dt, nd = struct.unpack_from("<II", r, 96)
            dims = struct.unpack_from("<4Q", r, 104)[:nd]
            off, size, _x = struct.unpack_from("<QQQ", r, 136)
            out[name] = (dt, tuple(dims), off, size)
    return out


def dequant_q4cp(blob, R, C):
    cb = np.frombuffer(blob, np.float32, 16)
    stride = ((C // 32 * 2) + 15) & ~15
    codes = np.frombuffer(blob, np.uint8, R * C // 2, 64).reshape(R, C // 2)
    sc = np.frombuffer(blob, np.uint8, R * stride, 64 + R * C // 2).reshape(R, stride)
    sc = sc[:, : C // 32 * 2].copy().view(np.float16).astype(np.float32)
    nib = np.empty((R, C), np.uint8)
    nib[:, 0::2] = codes & 15
    nib[:, 1::2] = codes >> 4
    return (cb[nib].reshape(R, C // 32, 32) * sc[:, :, None]).reshape(R, C)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model_dir")
    ap.add_argument("--old-overlay", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--embed", action="store_true", help="also embed_tokens -> q8g32")
    ap.add_argument("--q4-keep", default="", help="comma list of name substrings kept 4-bit")
    ap.add_argument("--check", type=int, default=0,
                    help="report rel-L2 error (q8 vs bf16 and old q4 vs bf16) on every N-th tensor")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    out = os.path.abspath(os.path.expanduser(a.out))
    if "/models/" in out.replace("\\", "/") + "/" and os.path.basename(
            os.path.dirname(out)) == "models":
        raise SystemExit("refusing to write into a models/ directory; pick another path")
    t_start = time.time()
    check_config(json.load(open(os.path.join(a.model_dir, "config.json"))))
    table = scan_model_dir(a.model_dir)
    src = {}
    for sname, (st, dt, shape) in table.items():
        m = map_lm_name(sname)
        if m is not None:
            src[m] = (st, dt, shape, sname)
    old = read_hgn(a.old_overlay)
    keep = [s for s in a.q4_keep.split(",") if s]

    names = sorted(old)
    if a.embed and "embed_tokens.weight" not in old:
        names.append("embed_tokens.weight")
    plan = []  # (name, kind, dims, size)
    for n in names:
        if any(s in n for s in keep):
            if n not in old:
                raise SystemExit(f"--q4-keep {n}: not in the old overlay")
            dt, dims, off, size = old[n]
            plan.append((n, ("copy", dt), dims, size))
            continue
        if n not in src:
            raise SystemExit(f"{n}: not in the safetensors")
        _st, sdt, shape, _s = src[n]
        if sdt != "BF16":
            raise SystemExit(f"{n}: source dtype {sdt}, expected BF16")
        shape = tuple(shape)
        if n in old and old[n][1] != shape:
            raise SystemExit(f"{n}: shape {shape} != old overlay {old[n][1]}")
        R, C = int(np.prod(shape[:-1])), shape[-1]
        if any(n.endswith(s) for s in BF16_SMALL):
            plan.append((n, ("bf16", DT_BF16), shape, R * C * 2))
        else:
            if C % 32:
                raise SystemExit(f"{n}: cols {C} not a multiple of 32")
            plan.append((n, ("q8", DT_Q8G32), shape, R * C + R * C // 32 * 2))

    from collections import Counter
    cnt, byt = Counter(), Counter()
    for _n, (k, _d), _dims, size in plan:
        cnt[k] += 1
        byt[k] += size
    print(f"plan: {len(plan)} tensors " +
          ", ".join(f"{k} {cnt[k]} ({human(byt[k])})" for k in sorted(cnt)) +
          f"; total {human(sum(byt.values()))}", flush=True)
    if a.dry_run:
        return

    os.makedirs(os.path.dirname(out), exist_ok=True)
    tmp = out + ".part"
    w = HgnWriter(tmp, os.path.basename(out).replace(".hgn", "")[:63])
    for n, (_k, dt), dims, size in plan:
        w.plan(n, dt, dims, size)
    w.begin()
    fo = open(a.old_overlay, "rb")
    worst = []
    for i, (n, (k, dt), dims, size) in enumerate(plan):
        t0 = time.time()
        R, C = int(np.prod(dims[:-1])), dims[-1]
        if k == "copy":
            fo.seek(old[n][2])
            blob = fo.read(size)
        else:
            st, _sdt, _shape, sname = src[n]
            if k == "bf16":
                blob = bytes(st.raw(sname)[2])
            else:
                x = st.f32(sname).reshape(R, C)
                blob = quant_q8g32(x)
                if a.check and i % a.check == 0:
                    ref = x.astype(np.float64)
                    nrm = np.linalg.norm(ref) or 1.0
                    e8 = np.linalg.norm(dequant_q8g32(blob, R, C) - ref) / nrm
                    msg = f"q8 rel {e8:.2e}"
                    if n in old and old[n][0] == 5:
                        fo.seek(old[n][2])
                        ob = fo.read(old[n][3])
                        e4 = np.linalg.norm(dequant_q4cp(ob, R, C) - ref) / nrm
                        msg += f"  old q4cp rel {e4:.2e}"
                    worst.append((e8, n))
                    print(f"    check {n}: {msg}", flush=True)
                del x
        w.write(blob)
        print(f"  [{i + 1}/{len(plan)}] {k:4s} {n} {dims} {human(size)} "
              f"{time.time() - t0:.1f}s", flush=True)
        del blob
    w.finish()
    os.replace(tmp, out)
    if worst:
        e, n = max(worst)
        print(f"check: worst q8 rel-L2 {e:.2e} ({n})")
        if e > 1e-2:
            raise SystemExit("FAIL: q8g32 error too large")
    print(f"wrote {out} ({human(os.path.getsize(out))}) in "
          f"{(time.time() - t_start) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
