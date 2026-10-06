# -*- coding: utf-8 -*-
"""One-tensor ht encoder quality test: encode layer-0 in_proj_qkv from BF16,
decode back, rel-L2 vs BF16 ground truth; compare with the official payload."""
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hgnc
import v2enc

V2 = "models/qwen38-flash-next-v2.hgn"
BF16 = "/home/mark/Models/Qwen3.8-Flash-Next"
TMP = "/home/mark/Models/v2repro/ht_test"
HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    os.makedirs(TMP, exist_ok=True)
    v2enc.HT_LUT.astype("<f4").tofile(os.path.join(HERE, "ht_lut.f32"))

    _, _, _, ents = hgnc.read(V2)
    by = {e.name: e for e in ents}
    nm = "layers.0.linear_attn.in_proj_qkv.weight"
    e = by[nm]
    O, K = e.shape

    def sym(n):
        return np.frombuffer(bytes(hgnc.payload(V2, by[n])), "<f2").astype(np.float32)

    suh, svh = sym(nm[:-7] + ".suh"), sym(nm[:-7] + ".svh")
    st = hgnc.SafeTensors(BF16)
    W = st.f32("model.language_model." + nm)
    G = v2enc.rotate(W, 1.0 / suh, 1.0 / svh)
    tg = os.path.join(TMP, "t.f32")
    co = os.path.join(TMP, "c.bin")
    G.astype("<f4").tofile(tg)
    passes = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    r = subprocess.run([os.path.join(HERE, "htenc"), tg, str(O), str(K), co, str(passes)],
                       capture_output=True, text=True)
    print(r.stderr.strip())
    codes = np.fromfile(co, np.uint8)
    Gh = v2enc.ht_decode(codes, O, K)
    Wh = v2enc.unrotate(Gh, suh, svh)
    rl = float(np.linalg.norm(Wh - W) / np.linalg.norm(W))
    print(f"ours  : rel-L2 {rl:.6f}")
    Gref = v2enc.ht_decode(hgnc.payload(V2, e), O, K)
    Wref = v2enc.unrotate(Gref, suh, svh)
    print(f"officl: rel-L2 {float(np.linalg.norm(Wref - W) / np.linalg.norm(W)):.6f}")


if __name__ == "__main__":
    main()
