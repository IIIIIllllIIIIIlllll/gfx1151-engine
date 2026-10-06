# -*- coding: utf-8 -*-
"""v2_encode.py — encode every v2-format tensor from BF16 source weights.

Reads the official v2 table for the target tensor set (names/stores/shapes/
variants), then produces, under WORK:
  symbols/<name>.bin        store-2 payloads, byte-copied from official v2
  parts/q6g64/<name>.bin    q6g64 encodes from BF16 (numpy)
  parts/ht/<name>.bin       ht encodes from BF16 (rotate in numpy + htenc C++)
  parts/i4r/<name>.<e0>-<e1>.bin   i4r encodes, 64 experts per part
  manifest.json             per-tensor sizes + self-check stats

Usage:
  PYTHONPATH=~/Workspace/pylib python3 v2_encode.py symbols|q6g64|ht|i4r [--workers N]
  PYTHONPATH=~/Workspace/pylib python3 v2_encode.py all
"""
import argparse
import json
import os
import subprocess
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hgnc
import v2enc

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
V2 = os.path.join(REPO, "models/qwen38-flash-next-v2.hgn")
BF16 = os.environ.get("V2REP_BF16", "/home/mark/Models/Qwen3.8-Flash-Next")
WORK = os.environ.get("V2REP_WORK", "/home/mark/Models/v2repro")
HTENC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "htenc")
CHUNK = 64  # experts per i4r part

_ST = None
_TABLE = None


def table():
    global _TABLE
    if _TABLE is None:
        _, _, _, ents = hgnc.read(V2)
        _TABLE = {e.name: e for e in ents}
    return _TABLE


def bf16():
    global _ST
    if _ST is None:
        _ST = hgnc.SafeTensors(BF16)
    return _ST


def bf16_name(hgn_name):
    n = hgn_name[:-7] if hgn_name.endswith(".weight") else hgn_name
    if n.startswith("mtp.") or n == "lm_head":
        cands = (n + ".weight", n)
    else:
        # experts are stored without the .weight suffix in HF safetensors
        cands = (hgnc.LM_PREFIX + n + ".weight", hgnc.LM_PREFIX + n)
    for c in cands:
        if c in bf16().loc:
            return c
    raise KeyError(hgn_name)


def sym_payload(name):
    e = table()[name]
    return bytes(hgnc.payload(V2, e))


def sym_f32(name):
    return np.frombuffer(sym_payload(name), "<f2").astype(np.float32)


def ht_sym_names(name):
    """ht tensor 'a.b.weight' -> ('a.b.suh', 'a.b.svh')."""
    base = name[:-7] if name.endswith(".weight") else name
    return base + ".suh", base + ".svh"


def i4r_sym_names(name):
    """expert tensor -> (input-side, output-side) symbol names (official naming)."""
    base = name[: -len(".weight")]
    if base.endswith("down_proj"):
        return base + ".suh", base + ".svh"
    assert base.endswith("gate_up_proj")
    return base[: -len("gate_up_proj")] + "su", base + ".svh"


# ------------------------------------------------------------------ symbols
def do_symbols():
    out = os.path.join(WORK, "symbols")
    os.makedirs(out, exist_ok=True)
    n = 0
    for name, e in table().items():
        if e.store != 2:
            continue
        fp = os.path.join(out, name + ".bin")
        if not (os.path.isfile(fp) and os.path.getsize(fp) == e.size):
            with open(fp + ".tmp", "wb") as f:
                f.write(sym_payload(name))
            os.replace(fp + ".tmp", fp)
        n += 1
    print(f"symbols: {n} extracted -> {out}")


# ------------------------------------------------------------------- q6g64
def do_q6g64():
    out = os.path.join(WORK, "parts", "q6g64")
    os.makedirs(out, exist_ok=True)
    todo = [(n, e) for n, e in table().items() if e.store == 24]
    print(f"q6g64: {len(todo)} tensors")
    for i, (name, e) in enumerate(sorted(todo)):
        fp = os.path.join(out, name + ".bin")
        if os.path.isfile(fp) and os.path.getsize(fp) == e.size:
            continue
        R, C = e.shape
        W = bf16().f32(bf16_name(name))
        blob = v2enc.q6g64_encode(W)
        assert len(blob) == e.size, (name, len(blob), e.size)
        with open(fp + ".tmp", "wb") as f:
            f.write(blob)
        os.replace(fp + ".tmp", fp)
        Wh = v2enc.q6g64_decode(blob, R, C)
        rl = float(np.linalg.norm(Wh - W) / np.linalg.norm(W))
        print(f"  [{i+1}/{len(todo)}] {name} rel-L2 {rl:.5f}", flush=True)


# ---------------------------------------------------------------------- ht
def do_ht():
    out = os.path.join(WORK, "parts", "ht")
    tdir = os.path.join(WORK, "targets")
    os.makedirs(out, exist_ok=True)
    os.makedirs(tdir, exist_ok=True)
    v2enc.HT_LUT.astype("<f4").tofile(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "ht_lut.f32"))
    todo = [(n, e) for n, e in table().items() if e.store == 16]
    todo.sort(key=lambda x: -x[1].size)  # big tensors first
    print(f"ht: {len(todo)} tensors")
    for i, (name, e) in enumerate(todo):
        fp = os.path.join(out, name + ".bin")
        if os.path.isfile(fp) and os.path.getsize(fp) == e.size:
            continue
        t0 = time.time()
        O, K = e.shape
        su_n, sv_n = ht_sym_names(name)
        su, sv = sym_f32(su_n), sym_f32(sv_n)
        W = bf16().f32(bf16_name(name)).reshape(O, K)
        G = v2enc.rotate(W, 1.0 / su, 1.0 / sv)
        # dead rows (svh==0 -> 1/sv=inf, W row all-zero): G row is irrelevant
        # (decode multiplies by svh=0), but nan/inf would poison the whole
        # ht block's beam costs — clamp to 0
        G[~np.isfinite(G)] = 0
        tg = os.path.join(tdir, "cur.f32")
        G.astype("<f4").tofile(tg)
        del W, G
        r = subprocess.run([HTENC, tg, str(O), str(K), fp + ".tmp",
                            os.environ.get("HTENC_BEAM", "128"),
                            os.environ.get("HTENC_SWEEPS", "6")],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"htenc failed on {name}: {r.stderr}")
        os.replace(fp + ".tmp", fp)
        print(f"  [{i+1}/{len(todo)}] {name} {O}x{K} {time.time()-t0:.0f}s | {r.stderr.strip()}",
              flush=True)


# --------------------------------------------------------------------- i4r
def i4r_worker(task):
    name, e0, e1 = task
    e = table()[name]
    E, O, K = e.shape
    su_n, sv_n = i4r_sym_names(name)
    su, sv = sym_f32(su_n), sym_f32(sv_n)
    W = bf16().f32(bf16_name(name), e0, e1)
    X = v2enc.rotate(W, 1.0 / su, 1.0 / sv)
    X[~np.isfinite(X)] = 0  # same dead-row guard as do_ht
    pk, sc = v2enc.i4r_encode_rot(X)
    blob = pk.tobytes() + sc.astype("<f2").tobytes()
    n = e1 - e0
    expect = n * O * K // 2 + n * O * (K // 128) * 2
    assert len(blob) == expect, (name, len(blob), expect)
    # roundtrip self-check on the first 2 experts of the part
    Xh = v2enc.i4r_decode_rot(pk[:2], sc[:2])
    Wh = v2enc.unrotate(Xh, su, sv)
    rl = float(np.linalg.norm(Wh - W[:2]) / np.linalg.norm(W[:2]))
    fp = os.path.join(WORK, "parts", "i4r", f"{name}.{e0}-{e1}.bin")
    with open(fp + ".tmp", "wb") as f:
        f.write(blob)
    os.replace(fp + ".tmp", fp)
    return dict(tensor=name, e0=e0, e1=e1, bytes=len(blob), rel_l2=rl)


def do_i4r(workers):
    os.makedirs(os.path.join(WORK, "parts", "i4r"), exist_ok=True)
    tasks = []
    for name, e in sorted(table().items()):
        if e.store != 23:
            continue
        E = e.shape[0]
        for e0 in range(0, E, CHUNK):
            e1 = min(e0 + CHUNK, E)
            fp = os.path.join(WORK, "parts", "i4r", f"{name}.{e0}-{e1}.bin")
            n = e1 - e0
            O, K = e.shape[1], e.shape[2]
            expect = n * O * K // 2 + n * O * (K // 128) * 2
            if os.path.isfile(fp) and os.path.getsize(fp) == expect:
                continue
            tasks.append((name, e0, e1))
    print(f"i4r: {len(tasks)} tasks remaining, {workers} workers", flush=True)
    t0 = time.time()
    results = []
    with Pool(workers) as pool:
        for i, r in enumerate(pool.imap_unordered(i4r_worker, tasks)):
            results.append(r)
            if (i + 1) % 16 == 0 or i + 1 == len(tasks):
                rels = [x["rel_l2"] for x in results]
                el = time.time() - t0
                print(f"  [{i+1}/{len(tasks)}] {el:.0f}s mean rel-L2 "
                      f"{np.mean(rels):.5f} max {max(rels):.5f}", flush=True)
    man = dict(tasks=len(results), seconds=time.time() - t0,
               mean_rel_l2=float(np.mean([r["rel_l2"] for r in results])) if results else None,
               parts=results)
    with open(os.path.join(WORK, "manifest_i4r.json"), "w") as f:
        json.dump(man, f, indent=1)
    print(f"i4r DONE: {len(results)} parts, {man['seconds']:.0f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["symbols", "q6g64", "ht", "i4r", "all"])
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    os.makedirs(WORK, exist_ok=True)
    if args.stage in ("symbols", "all"):
        do_symbols()
    if args.stage in ("q6g64", "all"):
        do_q6g64()
    if args.stage in ("ht", "all"):
        do_ht()
    if args.stage in ("i4r", "all"):
        do_i4r(args.workers)


if __name__ == "__main__":
    main()
