# -*- coding: utf-8 -*-
"""Decode-check my v2enc numpy reference against the official v2 file + BF16 ground
truth. Prints rel-L2 / Pearson per sampled tensor; also dumps symbol statistics.

Usage: PYTHONPATH=~/Workspace/pylib python3 v2_spotcheck.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hgnc
import v2enc

V2 = "models/qwen38-flash-next-v2.hgn"
BF16 = "/home/mark/Models/Qwen3.8-Flash-Next"


def rel(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def pearson(a, b):
    a = a.ravel().astype(np.float64)
    b = b.ravel().astype(np.float64)
    return float(np.corrcoef(a, b)[0, 1])


def main():
    _, _, _, ents = hgnc.read(V2)
    by = {e.name: e for e in ents}
    st = hgnc.SafeTensors(BF16)

    def sym(name):
        e = by[name]
        assert e.store == 2
        return np.frombuffer(bytes(hgnc.payload(V2, e)), "<f2").astype(np.float32)

    # -- symbol statistics (suh should be ±1; dense svh = ±row std, experts ±1)
    for nm in ("layers.0.linear_attn.in_proj_qkv.suh", "layers.0.linear_attn.in_proj_qkv.svh",
               "layers.0.mlp.experts.down_proj.suh", "layers.0.mlp.experts.down_proj.svh",
               "layers.0.mlp.experts.su", "layers.0.mlp.experts.gate_up_proj.svh"):
        v = sym(nm)
        u = np.unique(np.abs(v))
        print(f"sym {nm}: n={v.size} |v| uniq={u.size} min={u.min():.6f} max={u.max():.6f}")

    # -- ht: layers.0.linear_attn.in_proj_qkv.weight [10240, 2560]
    nm = "layers.0.linear_attn.in_proj_qkv.weight"
    e = by[nm]
    O, K = e.shape
    G = v2enc.ht_decode(hgnc.payload(V2, e), O, K)
    W = st.f32("model.language_model." + nm)
    suh, svh = sym(nm[:-7] + ".suh"), sym(nm[:-7] + ".svh")  # ht 符号名不带 .weight
    Wh = v2enc.unrotate(G, suh, svh)
    print(f"ht {nm}: rel-L2 {rel(Wh, W):.6f} pearson {pearson(Wh, W):.6f}")

    # -- i4r: layers.0 experts 0..1, both kinds
    for kind, K in (("down_proj", 640), ("gate_up_proj", 2560)):
        nm = f"layers.0.mlp.experts.{kind}.weight"
        e = by[nm]
        E, O, K2 = e.shape
        assert K2 == K
        raw = hgnc.payload(V2, e)
        nc = E * O * K // 2
        codes = np.frombuffer(bytes(raw[: 2 * O * K // 2]), np.uint8).reshape(2, O, K // 2)
        soff = nc
        scales = np.frombuffer(bytes(raw[soff: soff + 2 * O * (K // 128) * 2]), "<f2").reshape(2, O, K // 128)
        su = sym(f"layers.0.mlp.experts.{kind}.suh" if kind == "down_proj" else "layers.0.mlp.experts.su")
        sv = sym(f"layers.0.mlp.experts.{kind}.svh")
        Xh = v2enc.i4r_decode_rot(codes, scales)
        Wh = v2enc.unrotate(Xh, su, sv)
        W = st.f32(f"model.language_model.layers.0.mlp.experts.{kind}", 0, 2)
        print(f"i4r {nm}: rel-L2 {rel(Wh, W):.6f} pearson {pearson(Wh, W):.6f}")

    # -- q6g64: layers.0.attn_hyper_connection.input_mix_weight_down [320, 10240]
    nm = "layers.0.attn_hyper_connection.input_mix_weight_down.weight"
    e = by[nm]
    R, C = e.shape
    Wh = v2enc.q6g64_decode(hgnc.payload(V2, e), R, C)
    W = st.f32("model.language_model." + nm)
    print(f"q6g64 {nm}: rel-L2 {rel(Wh, W):.6f} pearson {pearson(Wh, W):.6f}")


if __name__ == "__main__":
    main()
