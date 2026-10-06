# -*- coding: utf-8 -*-
"""v2_verify.py — offline verification of the reproduced v2 files.

1) both containers readable; tensor set/store/variant/shape == official v2
2) all 818 symbol payloads byte-identical to official
3) sampled decode vs BF16: ours rel-L2 vs official rel-L2 (must be <= off*1.25)
4) copied tensors (store 0/4/5/7 + ngram fp8): xor-fold checksum matches the
   official table entry (i.e. payload bytes equal)

Usage: PYTHONPATH=~/Workspace/pylib python3 v2_verify.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hgnc
import v2enc

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
V2 = os.path.join(REPO, "models/qwen38-flash-next-v2.hgn")
NG = os.path.join(REPO, "models/qwen38-flash-next-ngram.hgn")
BF16 = os.environ.get("V2REP_BF16", "/home/mark/Models/Qwen3.8-Flash-Next")
WORK = os.environ.get("V2REP_WORK", "/home/mark/Models/v2repro")
NEW = os.path.join(WORK, "out", "qwen38-flash-next-v2.hgn")
NEWNG = os.path.join(WORK, "out", "qwen38-flash-next-ngram.hgn")

fails = []


def check(ok, msg):
    print(("  OK  " if ok else "  FAIL") + " " + msg, flush=True)
    if not ok:
        fails.append(msg)


def rel(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def main():
    # 1) containers
    _, ident_o, _, oents = hgnc.read(V2)
    _, ident_n, _, nents = hgnc.read(NEW)
    print(f"official: {len(oents)} tensors ident={ident_o!r}")
    print(f"ours    : {len(nents)} tensors ident={ident_n!r}")
    o_by = {e.name: e for e in oents}
    n_by = {e.name: e for e in nents}
    check(set(o_by) == set(n_by), f"tensor set equal ({len(o_by)} vs {len(n_by)})")
    bad = [nm for nm in o_by if nm in n_by and
           (o_by[nm].store, o_by[nm].variant, o_by[nm].shape, o_by[nm].size) !=
           (n_by[nm].store, n_by[nm].variant, n_by[nm].shape, n_by[nm].size)]
    check(not bad, f"store/variant/shape/size match ({len(bad)} bad)")
    for nm in bad[:5]:
        print("    ", nm, o_by[nm].store, o_by[nm].shape, "->", n_by[nm].store, n_by[nm].shape)

    # 2) symbols byte-identical
    nsym = 0
    symbad = 0
    for nm, e in o_by.items():
        if e.store != 2:
            continue
        a = bytes(hgnc.payload(V2, e))
        b = bytes(hgnc.payload(NEW, n_by[nm]))
        nsym += 1
        if a != b:
            symbad += 1
    check(symbad == 0, f"symbols byte-identical ({nsym - symbad}/{nsym})")

    # 3) sampled decode vs BF16
    st = hgnc.SafeTensors(BF16)

    def sym_of(path, by, n):
        return np.frombuffer(bytes(hgnc.payload(path, by[n])), "<f2").astype(np.float32)

    def bf16_name(hgn_name):
        n = hgn_name[:-7] if hgn_name.endswith(".weight") else hgn_name
        if n.startswith("mtp.") or n == "lm_head":
            cands = (n + ".weight", n)
        else:
            cands = (hgnc.LM_PREFIX + n + ".weight", hgnc.LM_PREFIX + n)
        for c in cands:
            if c in st.loc:
                return c
        raise KeyError(hgn_name)

    rng = np.random.RandomState(0)
    ht_names = sorted(n for n, e in o_by.items() if e.store == 16)
    q6_names = sorted(n for n, e in o_by.items() if e.store == 24)
    i4_names = sorted(n for n, e in o_by.items() if e.store == 23)
    ht_s = [ht_names[i] for i in rng.choice(len(ht_names), 6, replace=False)]
    q6_s = [q6_names[i] for i in rng.choice(len(q6_names), 6, replace=False)]
    i4_s = [i4_names[i] for i in rng.choice(len(i4_names), 4, replace=False)]
    if "layers.0.linear_attn.in_proj_qkv.weight" in o_by:
        ht_s[0] = "layers.0.linear_attn.in_proj_qkv.weight"

    for nm in ht_s:
        e = o_by[nm]
        O, K = e.shape
        su_n, sv_n = nm[:-7] + ".suh", nm[:-7] + ".svh"
        su, sv = sym_of(V2, o_by, su_n), sym_of(V2, o_by, sv_n)
        W = st.f32(bf16_name(nm)).reshape(O, K)
        ro = rel(v2enc.unrotate(v2enc.ht_decode(hgnc.payload(V2, e), O, K), su, sv), W)
        rn = rel(v2enc.unrotate(v2enc.ht_decode(hgnc.payload(NEW, n_by[nm]), O, K), su, sv), W)
        check(rn <= ro * 1.25 + 1e-3, f"ht {nm}: ours {rn:.5f} vs official {ro:.5f}")

    for nm in q6_s:
        e = o_by[nm]
        R, C = e.shape
        W = st.f32(bf16_name(nm)).reshape(R, C)
        ro = rel(v2enc.q6g64_decode(hgnc.payload(V2, e), R, C), W)
        rn = rel(v2enc.q6g64_decode(hgnc.payload(NEW, n_by[nm]), R, C), W)
        check(rn <= ro * 1.25 + 1e-3, f"q6g64 {nm}: ours {rn:.5f} vs official {ro:.5f}")

    for nm in i4_s:
        e = o_by[nm]
        E, O, K = e.shape
        base = nm[:-7]
        if base.endswith("down_proj"):
            su_n, sv_n = base + ".suh", base + ".svh"
        else:
            su_n, sv_n = base[: -len("gate_up_proj")] + "su", base + ".svh"
        su, sv = sym_of(V2, o_by, su_n), sym_of(V2, o_by, sv_n)
        W = st.f32(bf16_name(nm), 0, 2)
        for tag, path, ent in (("off", V2, e), ("our", NEW, n_by[nm])):
            raw = hgnc.payload(path, ent)
            nc = E * O * K // 2
            codes = np.frombuffer(bytes(raw[: 2 * O * K // 2]), np.uint8).reshape(2, O, K // 2)
            scales = np.frombuffer(bytes(raw[nc: nc + 2 * O * (K // 128) * 2]), "<f2").reshape(2, O, K // 128)
            Xh = v2enc.i4r_decode_rot(codes, scales)
            r = rel(v2enc.unrotate(Xh, su, sv), W)
            if tag == "off":
                ro = r
            else:
                rn = r
        check(rn <= ro * 1.25 + 1e-3, f"i4r {nm}: ours {rn:.5f} vs official {ro:.5f}")

    # 4) copied tensors: checksum in our table must equal official's, and the
    # payload must fold to it (byte equality is implied by assemble's source
    # checksum assertion; here we re-fold a sample from the NEW file).
    copied = [nm for nm, e in o_by.items() if e.store in (0, 4, 5, 7)]
    sample = [copied[i] for i in rng.choice(len(copied), min(12, len(copied)), replace=False)]
    for nm in sample:
        e = n_by[nm]
        ck = 0
        buf = hgnc.payload(NEW, e)
        ck = hgnc.xor_fold(bytes(buf))
        check(ck == o_by[nm].checksum and e.checksum == o_by[nm].checksum,
              f"copy {nm}: checksum {ck:#x} == official {o_by[nm].checksum:#x}")

    # ngram file
    _, _, _, ng_o = hgnc.read(NG)
    _, _, _, ng_n = hgnc.read(NEWNG)
    eo, en = ng_o[0], ng_n[0]
    print(f"ngram: ours {en.size/2**30:.2f} GiB checksum {en.checksum:#x}")
    check(en.name == eo.name and en.store == eo.store and en.size == eo.size and
          en.checksum == eo.checksum, "ngram entry matches official")

    print("\n" + ("ALL CHECKS PASS" if not fails else f"{len(fails)} FAILURES"))
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
