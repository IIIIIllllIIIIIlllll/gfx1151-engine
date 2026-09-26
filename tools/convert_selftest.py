#!/usr/bin/env python3
"""convert_selftest.py — fast synthetic check of tools/flashnext2hgn.py
(no GPU, no real model, ~1 min). Builds a tiny fake Qwen3.8-Flash-Next
safetensors dir (valid config.json, small tensors, big enough that the
codebook row sampling draws from the rng) and checks:

  1. --classic output is byte-identical to a reference converter
     (--ref OLD_flashnext2hgn.py, e.g. `git show e2f4d9f:tools/flashnext2hgn.py`)
  2. HQ without imatrix: same layout/sizes as classic, only the routed
     experts differ (universal codebook; --expert-codebook trained: trained,
     sorted, max|cb| = 1); the weighted scale search beats absmax (universal:
     same codebook; trained: the classic file)
  3. HQ with a legacy imatrix.dat and with a GGUF imatrix: identical to each
     other, differ from the x^2 run, one never-routed expert uses the mean
  4. <name>.overlay.hgn: right tensor set (non-expert, non-mtp q4cp tensors
     + embed last), q8g32/bf16 dtypes, dequant error small; --only-overlay
     writes the same bytes; start.sh loads base, overlay, mtp in that order
  5. --jobs 1 == --jobs 3 (pool path deterministic)

Prints PASS/FAIL per check and exits non-zero on any failure.
"""

import argparse
import importlib.util
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import flashnext2hgn as F  # noqa: E402

LM = "model.language_model."
E, RG, C, RD = 8, 4200, 64, 64  # gate_up [E, RG, C]; down [E, C, 32]


def bf16(x):
    u = np.asarray(x, np.float32).view(np.uint32)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


def write_st(path, tensors):
    hdr, off, blobs = {}, 0, []
    for n, (dt, arr) in tensors.items():
        b = arr.tobytes()
        hdr[n] = {"dtype": dt, "shape": list(arr.shape), "data_offsets": [off, off + len(b)]}
        off += len(b)
        blobs.append(b)
    h = json.dumps(hdr).encode()
    h += b" " * ((-len(h)) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(h)) + h)
        for b in blobs:
            f.write(b)


def fake_model(d, rng):
    os.makedirs(d, exist_ok=True)
    json.dump({"text_config": F.EXPECTED_CONFIG}, open(os.path.join(d, "config.json"), "w"))
    t1, t2 = {}, {}

    def w(shape, s=0.02, heavy=False):
        x = rng.standard_t(3, shape) * s if heavy else rng.normal(0, s, shape)
        return ("BF16", bf16(x).reshape(shape))

    t1[LM + "embed_tokens.weight"] = w((40000, C))
    t1[LM + "layers.0.self_attn.q_proj.weight"] = w((C, C))
    t1[LM + "layers.0.self_attn.q_norm.weight"] = w((C,))
    t1[LM + "layers.0.linear_attn.in_proj_a.weight"] = w((16, C))
    t1[LM + "layers.0.mlp.shared_expert.up_proj.weight"] = w((96, C))
    t1[LM + "layers.0.mlp.gate.weight"] = w((E, C))
    t2[LM + "layers.0.mlp.experts.gate_up_proj"] = w((E, RG, C), heavy=True)
    t2[LM + "layers.0.mlp.experts.down_proj"] = w((E, C, 32), heavy=True)
    t2[LM + "layers.47.mlp.experts.gate_up_proj"] = w((E, RG, C), heavy=True)  # real experts are heavy-tailed
    t2["lm_head.weight"] = w((300, C))
    t2["mtp.layers.0.mlp.experts.gate_up_proj"] = w((E, 64, C), heavy=True)
    t2["mtp.layers.0.self_attn.q_proj.weight"] = w((C, C))
    write_st(os.path.join(d, "model-00001-of-00002.safetensors"), t1)
    write_st(os.path.join(d, "model-00002-of-00002.safetensors"), t2)
    return {"gate": {0: C, 47: C}, "down": {0: 32}}


def imatrices(d, dims, rng):
    """Same per-expert statistics as legacy .dat and as GGUF; expert 3 of
    layer 0 never routed (counts 0 / values 0)."""
    ents = {}
    for kind, layers in dims.items():
        for L, c in layers.items():
            # powers of two: sum/count (GGUF) and mean*ncall/ncall (legacy)
            # then give bit-identical means
            cnt = (2.0 ** rng.integers(6, 10, E)).astype(np.float32)
            cnt[3] = 0
            mean = (rng.gamma(0.7, 1.0, (E, c)) * (1 + 10 * (rng.random(c) < 0.03))).astype(np.float32)
            mean[3] = 0
            ents[f"blk.{L}.ffn_{kind}_exps.weight"] = (cnt, mean)
    # legacy: values = mean * ncall (llama.cpp save_imatrix_legacy), ncall = max count
    leg = os.path.join(d, "imatrix.dat")
    with open(leg, "wb") as f:
        f.write(struct.pack("<i", len(ents)))
        for n, (cnt, mean) in ents.items():
            nc = int(cnt.max())
            f.write(struct.pack("<i", len(n)) + n.encode() + struct.pack("<ii", nc, mean.size))
            f.write((mean * nc).astype(np.float32).tobytes())
    # GGUF v3: tensors <n>.in_sum2 [E, c] and <n>.counts [E, 1], F32
    gg = os.path.join(d, "imatrix.gguf")
    tens = []
    for n, (cnt, mean) in ents.items():
        tens.append((n + ".in_sum2", (mean.shape[1], E), (mean * cnt[:, None]).astype(np.float32)))
        tens.append((n + ".counts", (1, E), cnt.reshape(E, 1)))

    def s(x):
        b = x.encode()
        return struct.pack("<Q", len(b)) + b
    head = b"GGUF" + struct.pack("<IQQ", 3, len(tens), 1)
    head += s("general.type") + struct.pack("<I", 8) + s("imatrix")
    off, data = 0, b""
    for n, ne, a in tens:
        head += s(n) + struct.pack("<I", len(ne)) + b"".join(struct.pack("<Q", x) for x in ne)
        head += struct.pack("<IQ", 0, off)
        b = a.tobytes()
        b += b"\0" * ((-len(b)) % 32)
        data += b
        off += len(b)
    head += b"\0" * ((-len(head)) % 32)
    open(gg, "wb").write(head + data)
    return leg, gg


def run(args, log):
    cmd = [sys.executable] + args
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    open(log, "a").write("$ " + " ".join(cmd) + "\n" + r.stdout + "\n")
    if r.returncode:
        print(r.stdout[-2000:])
        raise SystemExit(f"FAIL: command failed ({r.returncode}): {' '.join(args[:3])}")
    return r.stdout


def recs(path):
    with open(path, "rb") as f:
        h = f.read(F.HDR)
        _m, _v, cnt, _r, ro, _do, fs = struct.unpack_from("<4sIIIQQQ", h, 0)
        f.seek(ro)
        out = {}
        for _ in range(cnt):
            r = f.read(F.REC)
            n = r[:96].split(b"\0")[0].decode()
            dt, nd = struct.unpack_from("<II", r, 96)
            dims = struct.unpack_from("<4Q", r, 104)[:nd]
            o, sz, _x = struct.unpack_from("<QQQ", r, 136)
            out[n] = (dt, tuple(dims), o, sz)
    return out


def blob(path, rec):
    with open(path, "rb") as f:
        f.seek(rec[2])
        return f.read(rec[3])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", help="reference (old) flashnext2hgn.py for the --classic check")
    ap.add_argument("--keep", action="store_true")
    a = ap.parse_args()
    fails = []

    def check(ok, msg):
        print(("PASS " if ok else "FAIL ") + msg, flush=True)
        if not ok:
            fails.append(msg)

    tmp = tempfile.mkdtemp(prefix="convst_")
    log = os.path.join(tmp, "log.txt")
    rng = np.random.default_rng(7)
    md = os.path.join(tmp, "model")
    dims = fake_model(md, rng)
    leg, gg = imatrices(tmp, dims, rng)
    eb = os.path.join(tmp, "bin")
    os.makedirs(eb)
    for b in ("gdec", "gdec-api"):
        open(os.path.join(eb, b), "w").close()
    conv = os.path.join(HERE, "flashnext2hgn.py")
    com = [md, "--name", "m", "--skip-mtp-sidecar", "--engine-bin", eb]

    def out(tag):
        return os.path.join(tmp, tag)

    run([conv] + com + ["--out", out("classic"), "--classic"], log)
    if a.ref:
        run([a.ref] + com + ["--out", out("ref")], log)
        same = all(open(os.path.join(out("classic"), f), "rb").read() ==
                   open(os.path.join(out("ref"), f), "rb").read() for f in ("m.hgn", "start.sh"))
        check(same, "1 --classic byte-identical to the reference converter (m.hgn, start.sh)")
    else:
        print("SKIP 1 --classic vs reference (no --ref given)")

    run([conv] + com + ["--out", out("x2"), "--jobs", "3"], log)
    run([conv] + com + ["--out", out("x2j1"), "--jobs", "1"], log)
    run([conv] + com + ["--out", out("tr"), "--jobs", "3", "--expert-codebook", "trained"], log)
    run([conv] + com + ["--out", out("leg"), "--imatrix", leg, "--jobs", "3"], log)
    st = run([conv] + com + ["--out", out("gg"), "--imatrix", gg, "--jobs", "2"], log)
    run([conv, md, "--name", "m", "--out", out("ovl"), "--only-overlay"], log)
    rd = lambda t, f="m.hgn": open(os.path.join(out(t), f), "rb").read()  # noqa: E731

    # 2. layout / codebooks / only experts differ / weighted error lower
    pc, px = os.path.join(out("classic"), "m.hgn"), os.path.join(out("x2"), "m.hgn")
    rc, rx = recs(pc), recs(px)
    check(rc == rx and os.path.getsize(pc) == os.path.getsize(px),
          "2a HQ base: same records (names/dtypes/dims/offsets/sizes) as classic")
    exp = [n for n in rc if F.EXP_RE.match(n)]
    diff = sorted(n for n in rc if blob(pc, rc[n]) != blob(px, rx[n]))
    check(diff == sorted(exp), f"2b only the {len(exp)} routed-expert tensors differ ({diff})")
    cbc = {n: np.frombuffer(blob(pc, rc[n])[:64], np.float32) for n in exp}
    check(all(blob(px, rx[n])[:64] == F.UNIVERSAL_EXPERT_CB.tobytes() for n in exp) and
          np.all(np.diff(F.UNIVERSAL_EXPERT_CB) > 0) and
          np.array_equal(F.UNIVERSAL_EXPERT_CB, -F.UNIVERSAL_EXPERT_CB[::-1]),
          "2c expert codebooks = UNIVERSAL_EXPERT_CB (sorted, symmetric)")
    pt = os.path.join(out("tr"), "m.hgn")
    rt = recs(pt)
    cbt = {n: np.frombuffer(blob(pt, rt[n])[:64], np.float32) for n in exp}
    trained = [n for n in exp if not np.array_equal(cbt[n], cbc[n])]
    diff_t = sorted(n for n in rc if blob(pc, rc[n]) != blob(pt, rt[n]))
    check(rt == rc and diff_t == sorted(exp) and len(trained) >= len(exp) - 1 and
          all(np.all(np.diff(c) > 0) and abs(np.abs(c).max() - 1) < 1e-6 for c in cbt.values()),
          f"2e --expert-codebook trained: same layout, codebooks trained ({len(trained)}/{len(exp)}), "
          "sorted, max|cb| = 1")
    tab = F.scan_model_dir(md)
    ok = True
    for n in exp:
        src = next(s for s in tab if F.map_lm_name(s) == n)
        stf = tab[src][0]
        Ee, R, Cc = rc[n][1]
        eu, en = F.expert_spot_check(stf, src, px, rx[n][2], Ee, R, Cc, None)
        eo, et = F.expert_spot_check(stf, src, pt, rt[n][2], Ee, R, Cc, None, ref_cb=cbc[n])
        _eo, ec = F.expert_spot_check(stf, src, pc, rc[n][2], Ee, R, Cc, None)
        ok &= en < eu and et < eo and abs(ec - eo) <= 1e-6 * eo
        print(f"     {n}: x^2-weighted err universal absmax {eu:.3e} -> HQ {en:.3e}; "
              f"classic {eo:.3e} (file {ec:.3e}) -> trained {et:.3e}")
    check(ok, "2d weighted scale search beats absmax on every expert tensor (universal: same "
          "codebook; trained: vs the classic file)")

    # 3. imatrix paths
    check(rd("leg") == rd("gg"), "3a legacy .dat imatrix == GGUF imatrix (same statistics)")
    check(rd("leg") != rd("x2"), "3b imatrix run differs from the x^2 run")
    pg, rg = os.path.join(out("gg"), "m.hgn"), recs(os.path.join(out("gg"), "m.hgn"))
    mtp = [n for n in exp if n.startswith("mtp.")]
    check(all(blob(pg, rg[n]) == blob(px, rx[n]) for n in mtp),
          "3c MTP experts (no blk.48 in imatrix) fall back to x^2 weighting")
    im = F.Imatrix(gg)
    v = im.experts(0, "gate", E, C)
    check(v is not None and np.allclose(v[3], np.delete(v, 3, 0).mean(0)),
          "3d never-routed expert gets the layer-mean importance")
    check("imatrix-weighted" in st and "q4i+imat" in st, "3e log reports imatrix weighting")

    # 4. overlay
    po = os.path.join(out("x2"), "m.overlay.hgn")
    ro = recs(po)
    want = F.overlay_names({F.map_lm_name(s): 0 for s in tab if F.map_lm_name(s)})
    with open(po, "rb") as f:
        f.seek(F.HDR)
        order = [f.read(F.REC)[:96].split(b"\0")[0].decode() for _ in ro]
    check(order == want and order[-1] == "embed_tokens.weight" and
          not any(F.EXP_RE.match(n) or n.startswith("mtp.") for n in order),
          f"4a overlay tensor set/order ({len(order)}: {order})")
    ok = True
    for n in order:
        dt, dd, _o, _s = ro[n]
        src = next(s for s in tab if F.map_lm_name(s) == n)
        x = tab[src][0].f32(src).reshape(-1, dd[-1])
        if any(n.endswith(s) for s in F.OVL_BF16):
            ok &= dt == F.DT_BF16 and blob(po, ro[n]) == bytes(tab[src][0].raw(src)[2])
        else:
            e = np.linalg.norm(F.dequant_q8g32(blob(po, ro[n]), *x.shape) - x) / np.linalg.norm(x)
            ok &= dt == F.DT_Q8G32 and e < 1e-2
    check(ok, "4b overlay dtypes (q8g32 / bf16 passthrough) and q8 error < 1e-2")
    check(rd("ovl", "m.overlay.hgn") == rd("x2", "m.overlay.hgn") == rd("gg", "m.overlay.hgn"),
          "4c --only-overlay and imatrix runs write the same overlay bytes")
    s = open(os.path.join(out("x2"), "start.sh")).read()
    i0, i1 = s.find('"$HERE/m.hgn"'), s.find('"$HERE/m.overlay.hgn"')
    check(0 <= i0 < i1, "4d start.sh loads m.hgn then m.overlay.hgn")
    check("m.overlay.hgn" not in open(os.path.join(out("classic"), "start.sh")).read() and
          not os.path.exists(os.path.join(out("classic"), "m.overlay.hgn")),
          "4e --classic writes no overlay")

    # 5. determinism across job counts
    check(rd("x2") == rd("x2j1"), "5 --jobs 1 == --jobs 3")

    if not a.keep:
        shutil.rmtree(tmp)
    else:
        print(f"kept {tmp}")
    print("CONVERT_SELFTEST " + ("PASS" if not fails else f"FAIL ({len(fails)})"))
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
