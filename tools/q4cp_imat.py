"""q4cp_imat.py — imatrix-weighted q4cp quantizer (same on-disk format as
flashnext2hgn.quant_q4cp: per-tensor 16-entry fp32 codebook, 4-bit codes,
one fp16 scale per 32 columns).

The original quantizer uses scale = group absmax and nearest codes. This one
searches the scale per group, minimizing the weighted squared error

    sum_j  wt_j (x_j - s * cb[q_j])^2,   wt_j = imat_j * sqrt(sigma2 + x_j^2)

(llama.cpp quantize_row_iq4_nl_impl weighting; sigma2 = 2 * mean(x^2) of the
group; imat_j = mean squared input activation of column j from an imatrix).
For each trial scale the codes are the nearest codebook entries, then the
least-squares scale for those codes is taken; the best trial wins, followed
by one reassign + refit round and a final reassign with the fp16-rounded
scale. Without an imatrix wt_j = x_j^2 (still better than absmax).
"""

import numpy as np

TRIALS = np.linspace(0.82, 1.18, 13).astype(np.float32)  # includes 1.0


def nearest(cb, x):
    """Nearest codebook index (ties -> lower), via the 15 midpoints."""
    mid = ((cb[:-1] + cb[1:]) * 0.5).astype(np.float32)
    return np.searchsorted(mid, x).astype(np.uint8)


def nearest_ref(cb, x):
    """flashnext2hgn's exact tie rule (used only by naive_groups)."""
    j = np.clip(np.searchsorted(cb, x), 1, 15)
    return np.where(x - cb[j - 1] <= cb[j] - x, j - 1, j).astype(np.uint8)


def quant_groups(g, qw, cb, trials=TRIALS):
    """g [N,32] f32 values, qw [N,32] f32 importance (or None), cb [16] sorted.
    Returns (nib [N,32] uint8, scale fp16 [N])."""
    g = g.astype(np.float32, copy=False)
    amax = np.abs(g).max(axis=1)
    g2 = g * g
    if qw is None:
        wt = g2
    else:
        sigma2 = 2.0 * g2.mean(axis=1, keepdims=True)
        wt = qw * np.sqrt(sigma2 + g2)
    wg = wt * g
    zero = amax == 0
    safe = np.where(zero, 1.0, amax).astype(np.float32)
    best_score = np.full(g.shape[0], -1.0, np.float32)
    best_s = safe.copy()
    for f in trials:
        q = cb[nearest(cb, g / (safe * f)[:, None])]
        sqx = (wg * q).sum(axis=1)
        sq2 = (wt * q * q).sum(axis=1)
        ok = sq2 > 0
        sc = np.where(ok, sqx * sqx / np.where(ok, sq2, 1.0), 0.0)
        upd = sc > best_score
        best_score = np.where(upd, sc, best_score)
        best_s = np.where(upd & ok, sqx / np.where(ok, sq2, 1.0), best_s)
    # one refit round at the chosen scale
    s = np.where(best_s > 0, best_s, safe)
    q = cb[nearest(cb, g / s[:, None])]
    sqx = (wg * q).sum(axis=1)
    sq2 = (wt * q * q).sum(axis=1)
    ok = sq2 > 0
    sc = np.where(ok, sqx * sqx / np.where(ok, sq2, 1.0), 0.0)
    s = np.where(ok & (sc >= best_score), sqx / np.where(ok, sq2, 1.0), s)
    s16 = np.where(zero, 0.0, s).astype(np.float16)
    s32 = s16.astype(np.float32)
    nib = nearest(cb, g / np.where(s32 == 0, 1.0, s32)[:, None])
    return nib, s16


def naive_groups(g, cb):
    """Exact replica of flashnext2hgn.quant_q4cp's per-group step."""
    a16 = np.abs(g).max(axis=1).astype(np.float16)
    s = a16.astype(np.float32)
    s = np.where(s == 0, 1.0, s)
    return nearest_ref(cb, (g / s[:, None]).astype(np.float32)), a16


def pack(nib, s16, R, C):
    """nib [R*C/32, 32], s16 [R*C/32] -> (codes bytes, scale rows bytes)."""
    G = C // 32
    stride = ((G * 2) + 15) & ~15
    n = nib.reshape(R, C)
    codes = (n[:, 0::2] | (n[:, 1::2] << 4)).astype(np.uint8)
    sc = np.zeros((R, stride), np.uint8)
    sc[:, : G * 2] = s16.reshape(R, G).view(np.uint8).reshape(R, G * 2)
    return codes.tobytes(), sc.tobytes()


def dequant(nib, s16, cb):
    return cb[nib] * s16.astype(np.float32)[:, None]
