# -*- coding: utf-8 -*-
"""v2_assemble.py — assemble the reproduced v2 main + ngram hgn files.

Target tensor set/order/stores/variants come from the official v2 table.
Payload sources:
  store 16 (ht)    -> WORK/parts/ht/<name>.bin          (encoded from BF16)
  store 24 (q6g64) -> WORK/parts/q6g64/<name>.bin       (encoded from BF16)
  store 23 (i4r)   -> WORK/parts/i4r/<name>.<e0>-<e1>.bin chunks, re-segmented
                      to [all codes][all scales]
  store 2  (f16)   -> WORK/symbols/<name>.bin           (== official bytes)
  store 0/4/5      -> byte copy from the w4b base (checksum must match official)
  store 7  (q8g64) -> byte copy from the mtp sidecar    (checksum must match)
ngram file: single fp8g tensor copied from w4b (checksum must match the
official ngram file's).

Usage: v2_assemble.py [--dry-run]
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hgnc

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
V2 = os.path.join(REPO, "models/qwen38-flash-next-v2.hgn")
NG = os.path.join(REPO, "models/qwen38-flash-next-ngram.hgn")
W4B = os.path.join(REPO, "models/old/qwen38-flash-next-w4b.hgn")
MTP = os.path.join(REPO, "models/qwen38-flash-next-mtp.hgn")
WORK = os.environ.get("V2REP_WORK", "/home/mark/Models/v2repro")
OUT = os.path.join(WORK, "out")
IDENT_MAIN = "qwen3.8-flash-next-v2-repro"
IDENT_NGRAM = "qwen3.8-flash-next-ngram-repro"
CHUNK = 64
BUF = 16 << 20


def copy_payload(fdst, src_path, off, size):
    with open(src_path, "rb") as f:
        f.seek(off)
        left = size
        while left:
            data = f.read(min(BUF, left))
            if not data:
                raise IOError(f"short read {src_path}")
            fdst.write(data)
            left -= len(data)


def xor_fold_payload(path, off, size):
    """u32 XOR-fold over exactly [off, off+size)."""
    main = size // 4 * 4
    total = np.uint32(0)
    done = 0
    with open(path, "rb") as f:
        f.seek(off)
        while done < main:
            n = min(BUF, main - done)
            a = np.frombuffer(f.read(n), dtype="<u4")
            total = np.bitwise_xor(total, np.bitwise_xor.reduce(a), dtype=np.uint32)
            done += n
        tail = f.read(size - main)
    if tail:
        t = np.frombuffer(tail + b"\0" * (4 - len(tail)), dtype="<u4")
        total = np.bitwise_xor(total, t[0], dtype=np.uint32)
    return int(total)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    ver, ident, _, v2ents = hgnc.read(V2)
    _, _, _, ngents = hgnc.read(NG)
    _, _, _, w4ents = hgnc.read(W4B)
    _, _, _,mtpents = hgnc.read(MTP)
    w4 = {e.name: e for e in w4ents}
    mtp = {e.name: e for e in mtpents}
    assert len(ngents) == 1

    n = len(v2ents)
    doff = hgnc.align64(hgnc.HDR + hgnc.ENT * n)
    pos = doff
    plan = []
    stats = {"ht": 0, "q6g64": 0, "i4r": 0, "sym": 0, "copy_w4b": 0, "copy_mtp": 0}
    for e in v2ents:
        ne = hgnc.Entry()
        ne.name, ne.store, ne.variant, ne.shape = e.name, e.store, e.variant, e.shape
        ne.offset = pos
        if e.store == 16:
            src = os.path.join(WORK, "parts", "ht", e.name + ".bin")
            stats["ht"] += 1
        elif e.store == 24:
            src = os.path.join(WORK, "parts", "q6g64", e.name + ".bin")
            stats["q6g64"] += 1
        elif e.store == 23:
            src = None  # chunk list resolved at write time
            stats["i4r"] += 1
        elif e.store == 2:
            src = os.path.join(WORK, "symbols", e.name + ".bin")
            stats["sym"] += 1
        elif e.store in (0, 4, 5):
            w = w4[e.name]
            assert w.checksum == e.checksum and w.size == e.size, f"w4b mismatch {e.name}"
            src = (W4B, w.offset)
            stats["copy_w4b"] += 1
        elif e.store == 7:
            m = mtp[e.name]
            assert m.checksum == e.checksum and m.size == e.size, f"mtp mismatch {e.name}"
            assert e.name.startswith("mtp."), e.name
            src = (MTP, m.offset)
            stats["copy_mtp"] += 1
        else:
            raise ValueError(f"unexpected store {e.store} for {e.name}")
        ne.size = e.size
        ne.checksum = e.checksum if isinstance(src, tuple) else 0
        plan.append((ne, src))
        pos = hgnc.align64(pos + e.size)
    fsize = pos
    print(f"main: {n} tensors, doff {doff}, size {fsize/2**30:.2f} GiB; {stats}")

    ng_e = ngents[0]
    ng_w = w4[ng_e.name]
    assert ng_w.checksum == ng_e.checksum and ng_w.size == ng_e.size, "w4b ple != official ngram"
    ng_doff = hgnc.align64(hgnc.HDR + hgnc.ENT)
    ng_size = hgnc.align64(ng_doff + ng_e.size)
    print(f"ngram: 1 tensor, size {ng_size/2**30:.2f} GiB (from w4b)")

    if args.dry_run:
        return

    # ---- main file
    os.makedirs(OUT, exist_ok=True)
    dst = os.path.join(OUT, "qwen38-flash-next-v2.hgn")
    if os.path.exists(dst):
        raise SystemExit(f"dst exists: {dst} (delete first)")
    free = __import__("shutil").disk_usage(OUT).free
    if free < fsize + ng_size + (1 << 30):
        raise SystemExit(f"insufficient space: free {free/2**30:.0f} GiB")
    t0 = time.time()
    tmp = dst + ".tmp"
    with open(tmp, "wb") as fdst:
        fdst.seek(doff)  # front hole is filled by header+records at the end
        cur = doff
        for i, (ne, src) in enumerate(plan):
            if ne.offset > cur:
                fdst.write(b"\0" * (ne.offset - cur))
                cur = ne.offset
            if ne.store == 23:
                E, O, K = ne.shape
                cc = CHUNK * O * K // 2
                cs = CHUNK * O * (K // 128) * 2
                blobs = []
                for e0 in range(0, E, CHUNK):
                    fp = os.path.join(WORK, "parts", "i4r", f"{ne.name}.{e0}-{min(e0+CHUNK, E)}.bin")
                    data = open(fp, "rb").read()
                    assert len(data) == cc + cs, (fp, len(data))
                    blobs.append(data)
                payload = b"".join(b[:cc] for b in blobs) + b"".join(b[cc:] for b in blobs)
                assert len(payload) == ne.size
                ne.checksum = hgnc.xor_fold(payload)
                fdst.write(payload)
                cur += len(payload)
            elif ne.store in (16, 24, 2):
                sz = os.path.getsize(src)
                assert sz == ne.size, (src, sz, ne.size)
                ne.checksum = xor_fold_payload(src, 0, sz)
                copy_payload(fdst, src, 0, sz)
                cur += sz
            else:
                soff, sfile = src[1], src[0]
                copy_payload(fdst, sfile, soff, ne.size)
                cur += ne.size
            if (i + 1) % 200 == 0:
                print(f"  [{i+1}/{n}] {cur/2**30:.1f} GiB {time.time()-t0:.0f}s", flush=True)
        fdst.truncate(cur)
        hgnc.write_header(fdst, ver, n, doff, cur, IDENT_MAIN)
        fdst.write(b"".join(ne.pack() for ne, _ in plan))
        pad = doff - fdst.tell()
        assert pad >= 0
        fdst.write(b"\0" * pad)
        fdst.flush()
        os.fsync(fdst.fileno())
    os.replace(tmp, dst)
    print(f"main done: {dst} {os.path.getsize(dst)/2**30:.2f} GiB, {time.time()-t0:.0f}s")

    # ---- ngram file
    t0 = time.time()
    ndst = os.path.join(OUT, "qwen38-flash-next-ngram.hgn")
    tmp = ndst + ".tmp"
    ne = hgnc.Entry()
    ne.name, ne.store, ne.variant, ne.shape = ng_e.name, ng_e.store, ng_e.variant, ng_e.shape
    ne.offset, ne.size, ne.checksum = ng_doff, ng_e.size, ng_e.checksum
    with open(tmp, "wb") as fdst:
        fdst.write(b"\0" * ng_doff)
        copy_payload(fdst, W4B, ng_w.offset, ng_w.size)
        fdst.truncate(ng_doff + ng_e.size)
        hgnc.write_header(fdst, 2, 1, ng_doff, ng_doff + ng_e.size, IDENT_NGRAM)
        fdst.write(ne.pack())
        pad = ng_doff - fdst.tell()
        assert pad >= 0
        fdst.write(b"\0" * pad)
        fdst.flush()
        os.fsync(fdst.fileno())
    os.replace(tmp, ndst)
    print(f"ngram done: {ndst} {os.path.getsize(ndst)/2**30:.2f} GiB, {time.time()-t0:.0f}s")

    with open(os.path.join(WORK, "assemble_manifest.json"), "w") as f:
        json.dump({"main": dst, "ngram": ndst, "stats": stats}, f, indent=1)


if __name__ == "__main__":
    main()
