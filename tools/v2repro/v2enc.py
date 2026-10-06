"""v2 store math: rotation, ht(16) / i4r(23) / q6g64(24) decode, i4r/q6g64 encode.

Format source: HGN-V2.md (reverse-engineered, numpy-verified) and
src/gpu/parts/28_kernels_hgn_v2.inc (engine decode kernels).

Rotation (ht and i4r share it):
  W = diag(svh) . H . G . H . diag(suh) / 128      (H = Sylvester-128, H.H = 128 I)
  G = Hn . diag(svh)^-1 . W . diag(suh)^-1 . Hn    (Hn = H / sqrt(128), block-diag)
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _had(n):
    H = np.array([[1.0]], np.float32)
    while H.shape[0] < n:
        H = np.block([[H, H], [H, -H]])
    return (H / np.sqrt(n)).astype(np.float32)


H128 = _had(128)


def rotate(W, su, sv):
    """W [..., O, K] -> rotated domain G. su = 1/suh (K,), sv = 1/svh (O,)."""
    sh = W.shape
    E, O, K = W.reshape(-1, *sh[-2:]).shape
    X = W.reshape(E, O, K) * sv[None, :, None] * su[None, None, :]
    X = (X.reshape(E, O, K // 128, 128) @ H128.T).reshape(E, O // 128, 128, K)
    return np.matmul(H128, X).reshape(*sh)


def unrotate(X, su, sv):
    """Rotated G [..., O, K] -> W, same su/sv convention as rotate()."""
    sh = X.shape
    E, O, K = X.reshape(-1, *sh[-2:]).shape
    Y = X.reshape(E, O, K)
    Y = np.matmul(H128.T, Y.reshape(E, O // 128, 128, K)).reshape(E, O, K // 128, 128)
    Y = (Y @ H128).reshape(E, O, K)
    return (Y * sv[None, :, None] * su[None, None, :]).reshape(*sh)


# ------------------------------------------------------------------ ht (16)
# code -> value: t = w16 * 0x83dcd12d; s = bytesum(t); X = f16(fma(1024+s, c, d))
# c = 1.732421875/256, d = -10.3828125; f32 fused fma then f16, as the kernel.
def ht_lut():
    w = np.arange(65536, dtype=np.uint64)
    t = ((w * np.uint64(0x83DCD12D)) & np.uint64(0xFFFFFFFF)).astype(np.uint32)
    s = ((t & 0xFF) + ((t >> 8) & 0xFF) + ((t >> 16) & 0xFF) + (t >> 24)).astype(np.int64)
    c = np.float32(1.732421875 / 256.0)
    d = np.float32(-10.3828125)
    out = np.empty(65536, np.float16)
    for i in range(65536):  # exact f32 fma via double (LUT built once)
        out[i] = np.float16(np.float32(math.fma(float(1024 + s[i]), float(c), float(d))))
    return out


HT_LUT = ht_lut()


def ht_decode(codes, O, K):
    """codes: uint8 [O*K/2] official tile layout -> rotated G [O, K] f32."""
    ntx = K // 128
    t = np.frombuffer(codes, np.uint8).reshape(O // 128, ntx, 8, 8, 128)  # ty,tx,g,q,byte
    w = t.view("<u4")  # [ty,tx,g,q,32] word L
    prev = np.roll(w, 1, axis=-1)
    V = w.astype(np.uint64) | (prev.astype(np.uint64) << np.uint64(32))
    e = np.arange(8, dtype=np.uint64)
    w16 = ((V[..., None] >> (np.uint64(28) - np.uint64(4) * e)) & np.uint64(0xFFFF)).astype(np.int64)
    X = HT_LUT[w16].astype(np.float32)  # [ty,tx,g,q,L,e]
    # scatter: r = ty*128 + q*16 + L%16 ; c = tx*128 + g*16 + (L//16)*8 + e
    ty, tx, g, q, L, ee = np.indices(X.shape, sparse=True)
    r = ty * 128 + q * 16 + L % 16
    c = tx * 128 + g * 16 + (L // 16) * 8 + ee
    rr, cc = np.broadcast_arrays(r, c)
    G = np.empty((O, K), np.float32)
    G[rr.ravel(), cc.ravel()] = X.ravel()
    return G


# ----------------------------------------------------------------- i4r (23)
def i4r_split(buf, E, O, K):
    b = np.frombuffer(buf, np.uint8)
    nc = E * O * K // 2
    return b[:nc].reshape(E, O, K // 2), b[nc:].view("<f2").reshape(E, O, K // 128)


def i4r_decode_rot(codes, scales):
    E, O, K2 = codes.shape
    q = np.stack([codes & 15, codes >> 4], -1).reshape(E, O, K2 * 2).astype(np.float32) - 8
    return (q.reshape(E, O, -1, 128) * scales.astype(np.float32)[..., None]).reshape(E, O, K2 * 2)


def f16(a):
    return np.asarray(a, np.float32).astype(np.float16)


def i4r_encode_rot(X, mults=(1.0, 0.97, 0.94, 0.91, 0.88, 0.85)):
    """Rotated X [..., K] (K%128==0). Signed scale s = -x_ext/8 * m so the group
    extreme maps to code 0; pick m with least squared error.
    -> (packed codes uint8 [..., K/2], scales f16 [..., K/128])"""
    sh = X.shape
    ng = sh[-1] // 128
    g = X.reshape(-1, ng, 128)
    idx = np.abs(g).argmax(-1)
    ext = np.take_along_axis(g, idx[..., None], -1)[..., 0]
    best_e = best_q = best_s = None
    for m in mults:
        s = f16(-ext / 8 * m)
        sf = s.astype(np.float32)
        q = np.clip(np.round(g / np.where(sf == 0, 1, sf)[..., None]) + 8, 0, 15)
        e = (((q - 8) * sf[..., None] - g) ** 2).sum(-1)
        if best_e is None:
            best_e, best_q, best_s = e, q, s
        else:
            w = e < best_e
            best_e = np.where(w, e, best_e)
            best_q = np.where(w[..., None], q, best_q)
            best_s = np.where(w, s, best_s)
    q = best_q.reshape(*sh).astype(np.uint8)
    return (q[..., 0::2] | (q[..., 1::2] << 4)), best_s.reshape(*sh[:-1], sh[-1] // 128)


# --------------------------------------------------------------- q6g64 (24)
def q6g64_stride(C):
    return (13 * C // 16 + 15) & ~15


def q6g64_decode(buf, R, C):
    st = q6g64_stride(C)
    b = np.frombuffer(buf, np.uint8).reshape(R, st)
    lo = b[:, : C // 2]
    hi = b[:, C // 2: 3 * C // 4]
    sb = b[:, 3 * C // 4: 3 * C // 4 + C // 64 * 4].copy().view("<f2").reshape(R, C // 64, 2)
    q4 = np.stack([lo & 15, lo >> 4], -1).reshape(R, C)
    q2 = (hi[:, :, None] >> (2 * np.arange(4)[None, None, :])).reshape(R, C // 4, 4)
    q2 = (q2 & 3).reshape(R, C)
    code = (q4 | (q2 << 4)).astype(np.float32).reshape(R, C // 64, 64)
    return (code * sb[..., 0:1].astype(np.float32) + sb[..., 1:2].astype(np.float32)).reshape(R, C)


def q6g64_encode(W, mults=(1.0, 0.97, 0.94, 0.91, 0.88, 0.85)):
    """W [R, C] f32 (C%64==0) -> payload bytes. Per 64-col group: f16 scale+bias,
    W = code*s + b; scale candidates from (hi-lo)/63 * m, least squared error."""
    R, C = W.shape
    g = W.reshape(R, C // 64, 64)
    lo, hi = g.min(-1), g.max(-1)
    best_e = best_q = best_s = best_b = None
    for m in mults:
        s = f16((hi - lo) / 63 * m)
        b = f16(lo)
        sf = s.astype(np.float32)
        bf = b.astype(np.float32)
        q = np.clip(np.round((g - bf[..., None]) / np.where(sf == 0, 1, sf)[..., None]), 0, 63)
        e = ((q * sf[..., None] + bf[..., None] - g) ** 2).sum(-1)
        if best_e is None:
            best_e, best_q, best_s, best_b = e, q, s, b
        else:
            w = e < best_e
            best_e = np.where(w, e, best_e)
            best_q = np.where(w[..., None], q, best_q)
            best_s = np.where(w, s, best_s)
            best_b = np.where(w, b, best_b)
    q = best_q.astype(np.uint8).reshape(R, C)
    lo4 = (q[:, 0::2] & 15) | ((q[:, 1::2] & 15) << 4)
    q2 = (q.reshape(R, C // 4, 4) >> 4).astype(np.uint8)
    hi2 = (q2 << (2 * np.arange(4, dtype=np.uint8))).sum(-1).astype(np.uint8)
    rec = np.stack([best_s, best_b], -1).view(np.uint8).reshape(R, C // 64 * 4)
    st = q6g64_stride(C)
    out = np.zeros((R, st), np.uint8)
    out[:, : C // 2] = lo4
    out[:, C // 2: 3 * C // 4] = hi2
    out[:, 3 * C // 4: 3 * C // 4 + C // 64 * 4] = rec
    return out.tobytes()
