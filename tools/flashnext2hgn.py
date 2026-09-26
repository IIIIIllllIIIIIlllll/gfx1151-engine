#!/usr/bin/env python3
"""flashnext2hgn.py — convert a Qwen3.8-Flash-Next (qwen4_exp) HF safetensors
model into the .hgn checkpoint format consumed by gdec (gfx1151).

Self-contained: stdlib + numpy only (no torch, no safetensors); uses the
sibling modules q4cp_imat.py and (for GGUF imatrix files) gguf_mini.py.

Usage:
  flashnext2hgn.py MODEL_DIR --out OUTDIR [--name NAME] [--imatrix FILE]
                 [--jobs N] [--skip-overlay | --only-overlay] [--overlay-skip S]
                 [--expert-codebook universal|trained]
                 [--skip-vision] [--skip-ple] [--skip-mtp-sidecar] [--classic]
  flashnext2hgn.py --quant-selftest

Outputs (into OUTDIR):
  <name>.hgn          base checkpoint (q4cp linears, bf16 norms, fp8 PLE table)
  <name>.overlay.hgn  8-bit dense overlay (q8g32 attention/GDN/HC/shared
                      expert/lm_head/embed); load it after the base
  <name>-mtp.hgn      MTP sidecar (q8g64 linears; optional at engine launch)
  <name>-vision.hgn   vision tower (all bf16; optional --vision-tower arg)
  tokenizer/          tokenizer files copied from MODEL_DIR
  start.sh            engine + API launcher: gdec base overlay mtp ...

Only this exact architecture shape is supported (the engine hardcodes it):
48 layers (GDN + every-4th QSA), hidden 2560, 512 experts, 24 heads,
vocab 248320, MTP 1 layer, vision depth 27. config.json is validated.

Quantization (default, "HQ"; see HGN-HQ.md):
  routed experts  q4cp (groups of 32 columns, fp16 scales): the scale of
                  every group is searched to minimize the weighted error,
                  weights = imatrix * sqrt(sigma^2 + x^2) with --imatrix,
                  else x^2 (tools/q4cp_imat.py); codebook = the fixed
                  universal expert codebook (UNIVERSAL_EXPERT_CB), or with
                  --expert-codebook trained one trained per tensor for the
                  weighted error (weighted Lloyd + alternating scale search /
                  least-squares refit on a 1M-value sample)
  dense           q8g32 in the overlay (llama.cpp Q8_0 math); the base keeps
                  a q4cp copy so it still runs without the overlay
  MTP sidecar     q8g64 per-64-column affine uint8
  PLE table       fp8 e4m3 with one global scale
--classic is the original data-free converter (absmax q4cp everywhere, no
overlay), bit-identical to earlier releases.

Layout reference (must match src/hgn.h):
  header 104B: magic "HGN1" | u32 version | u32 count | u32 reserved |
               u64 records_off | u64 data_off | u64 file_size | name[64]
  record 160B: name[96] | u32 dtype | u32 ndims | u64 dims[4] |
               u64 data_offset | u64 data_size | u64 extra
  dtype 0 bf16 | 4 i64 | 5 q4cp | 7 q8g64 | 10 fp8-e4m3(+trailing f32 scale)
  Blobs are 64-byte aligned.
"""

import argparse
import json
import math
import mmap
import os
import re
import shutil
import struct
import sys
import time
import zlib

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import q4cp_imat  # noqa: E402  (weighted q4cp group quantizer, numpy only)

DT_BF16, DT_I64, DT_Q4CP, DT_Q8G64, DT_Q8G32, DT_FP8 = 0, 4, 5, 7, 8, 10
HDR, REC = 104, 160
DONTNEED = getattr(os, "POSIX_FADV_DONTNEED", None)

# ----------------------------------------------------------------------------
# safetensors reading (zero-copy over mmap)
# ----------------------------------------------------------------------------

ST_ITEMSIZE = {"BF16": 2, "F16": 2, "F32": 4, "I64": 8, "I32": 4}


class STFile:
    def __init__(self, path):
        self.path = path
        self.fd = os.open(path, os.O_RDONLY)
        self.mm = mmap.mmap(self.fd, 0, access=mmap.ACCESS_READ)
        (hlen,) = struct.unpack_from("<Q", self.mm, 0)
        self.hdr = json.loads(self.mm[8 : 8 + hlen])
        self.base = 8 + hlen

    def meta(self, name):
        m = self.hdr[name]
        return m["dtype"], tuple(m["shape"]), m["data_offsets"]

    def raw(self, name):
        dt, shape, (a, b) = self.meta(name)
        return dt, shape, memoryview(self.mm)[self.base + a : self.base + b]

    def f32(self, name):
        """Tensor as float32 numpy array (converts bf16/f16)."""
        dt, shape, buf = self.raw(name)
        if dt == "BF16":
            u = np.frombuffer(buf, dtype=np.uint16)
            return (u.astype(np.uint32) << 16).view(np.float32).reshape(shape)
        if dt == "F16":
            return np.frombuffer(buf, dtype=np.float16).astype(np.float32).reshape(shape)
        if dt == "F32":
            return np.frombuffer(buf, dtype=np.float32).reshape(shape)
        raise ValueError(f"{name}: cannot read {dt} as f32")


def scan_model_dir(model_dir):
    """name -> (STFile, dtype, shape) for every *.safetensors in the dir."""
    table = {}
    for fn in sorted(os.listdir(model_dir)):
        if not fn.endswith(".safetensors"):
            continue
        st = STFile(os.path.join(model_dir, fn))
        for name in st.hdr:
            if name == "__metadata__":
                continue
            dt, shape, _ = st.meta(name)
            table[name] = (st, dt, shape)
    return table


# ----------------------------------------------------------------------------
# quantizers
# ----------------------------------------------------------------------------


def lloyd_codebook(x, iters=20):
    """16-level Lloyd centroids for normalized samples x in [-1, 1]."""
    lo, hi = float(x.min()), float(x.max())
    cb = np.linspace(lo, hi, 16).astype(np.float64)
    for _ in range(iters):
        idx = np.searchsorted(cb, x)
        idx = np.clip(idx, 1, 15)
        left, right = cb[idx - 1], cb[idx]
        assign = np.where(x - left <= right - x, idx - 1, idx)
        sums = np.bincount(assign, weights=x, minlength=16)
        cnts = np.bincount(assign, minlength=16)
        new = np.where(cnts > 0, sums / np.maximum(cnts, 1), cb)
        if np.allclose(new, cb, rtol=0, atol=1e-7):
            cb = new
            break
        cb = new
    return np.sort(cb).astype(np.float32)


def lloyd_codebook_normalized(x, iters=20):
    """16-level Lloyd codebook for group-absmax-normalized samples in
    [-1, 1]. The endpoints are snapped to exactly +-1 afterwards so every
    group's absmax element stays exactly representable (NF4-style)."""
    cb = lloyd_codebook(x, iters)
    cb[0], cb[-1] = -1.0, 1.0
    return cb


def q4cp_size(rows, cols):
    stride = ((cols // 32 * 2) + 15) & ~15
    return 64 + rows * cols // 2 + rows * stride, stride


def q4cp_codebook(w, rng):
    """Per-tensor 16-entry codebook: Lloyd on a group-absmax-normalized row
    subsample. w: [R, C] float32, or uint16 raw bf16 (same values, no full
    f32 copy). Consumes rng exactly like quant_q4cp always did."""
    R, C = w.shape
    n_rows = min(R, max(1, (1 << 21) // C))
    ridx = rng.choice(R, size=n_rows, replace=False) if n_rows < R else np.arange(R)
    sub = w[ridx]
    if sub.dtype == np.uint16:
        sub = (sub.astype(np.uint32) << 16).view(np.float32)
    sub = sub.reshape(-1, 32)
    am = np.abs(sub).max(axis=1, keepdims=True)
    xn = (sub / np.where(am == 0, 1.0, am)).ravel()
    return lloyd_codebook_normalized(xn[:: max(1, xn.size // (1 << 21))])


def quant_q4cp(w, rng):
    """w: float32 [R, C], C%32==0 -> blob bytes (data-free RTN: scale =
    group absmax, nearest code)."""
    R, C = w.shape
    assert C % 32 == 0
    G = C // 32
    blob_size, stride = q4cp_size(R, C)
    cb = q4cp_codebook(w, rng)
    codes = np.empty((R, C // 2), np.uint8)
    scales = np.zeros((R, stride), np.uint8)
    rows_chunk = max(1, (1 << 24) // C)  # ~64MB f32 per chunk
    for r0 in range(0, R, rows_chunk):
        r1 = min(R, r0 + rows_chunk)
        g = w[r0:r1].reshape(-1, 32)
        a = np.abs(g).max(axis=1)
        a16 = a.astype(np.float16)
        s = a16.astype(np.float32)
        s = np.where(s == 0, 1.0, s)
        x = (g / s[:, None]).astype(np.float32)
        j = np.searchsorted(cb, x)
        j = np.clip(j, 1, 15)
        nib = np.where(x - cb[j - 1] <= cb[j] - x, j - 1, j).astype(np.uint8)
        nib = nib.reshape(r1 - r0, C)
        codes[r0:r1] = nib[:, 0::2] | (nib[:, 1::2] << 4)
        sv = a16.reshape(r1 - r0, G).view(np.uint8).reshape(r1 - r0, G * 2)
        scales[r0:r1, : G * 2] = sv
    blob = cb.astype("<f4").tobytes() + codes.tobytes() + scales.tobytes()
    assert len(blob) == blob_size
    return blob


def q8g64_size(rows, cols):
    return rows * (cols + cols // 64 * 4)


def quant_q8g64(w):
    """w: float32 [R, C], C%64==0 -> blob bytes (per row: codes then
    (fp16 scale, fp16 min) per 64-col group)."""
    R, C = w.shape
    assert C % 64 == 0
    G = C // 64
    stride = C + G * 4
    rows_chunk = max(1, (1 << 24) // C)
    out = np.empty((R, stride), np.uint8)
    for r0 in range(0, R, rows_chunk):
        r1 = min(R, r0 + rows_chunk)
        g = w[r0:r1].reshape(r1 - r0, G, 64)
        mn, mx = g.min(axis=2), g.max(axis=2)
        sc = (mx - mn) / 255.0
        sc16, mn16 = sc.astype(np.float16), mn.astype(np.float16)
        scf = np.where(sc16.astype(np.float32) == 0, 1.0, sc16.astype(np.float32))
        code = np.round((g - mn16.astype(np.float32)[:, :, None]) / scf[:, :, None])
        out[r0:r1, :C] = np.clip(code, 0, 255).astype(np.uint8).reshape(r1 - r0, C)
        sm = np.stack([sc16, mn16], axis=2)  # [r, G, 2] fp16
        out[r0:r1, C:] = sm.view(np.uint8).reshape(r1 - r0, G * 4)
    return out.tobytes()


def q8g32_size(rows, cols):
    return rows * cols + rows * cols // 32 * 2


def quant_q8g32(w):
    """w float32 [R, C], C%32==0 -> planar blob [R*C int8][R*C/32 fp16]
    (llama.cpp Q8_0 math: d = amax/127, q = roundf(x/d))."""
    R, C = w.shape
    assert C % 32 == 0
    codes = np.empty((R, C), np.int8)
    scales = np.empty((R, C // 32), np.float16)
    rows_chunk = max(1, (1 << 24) // C)
    for r0 in range(0, R, rows_chunk):
        r1 = min(R, r0 + rows_chunk)
        g = w[r0:r1].reshape(-1, 32)
        d = np.abs(g).max(axis=1) / 127.0
        idv = np.where(d > 0, 1.0 / np.where(d > 0, d, 1.0), 0.0).astype(np.float32)
        q = np.rint(g * idv[:, None])
        # np.rint rounds half to even; roundf rounds exact .5 ties away from zero
        frac = g * idv[:, None]
        tie = np.abs(frac - np.trunc(frac)) == 0.5
        q = np.where(tie, np.trunc(frac) + np.sign(frac), q)
        codes[r0:r1] = np.clip(q, -127, 127).astype(np.int8).reshape(r1 - r0, C)
        scales[r0:r1] = d.astype(np.float16).reshape(r1 - r0, C // 32)
    return codes.tobytes() + scales.tobytes()


def dequant_q8g32(blob, R, C):
    q = np.frombuffer(blob, np.int8, R * C).reshape(R, C // 32, 32).astype(np.float32)
    s = np.frombuffer(blob, np.float16, R * C // 32, R * C).astype(np.float32)
    return (q * s.reshape(R, C // 32, 1)).reshape(R, C)


# fp8 e4m3: magnitude codes 0x00..0x7E are ascending (0x7F = nan, unused).
def _e4m3_mags():
    mags = []
    for c in range(0x7F):
        e, m = (c >> 3) & 0xF, c & 7
        if e == 0:
            v = math.ldexp(m, -9)
        else:
            v = math.ldexp(8 + m, e - 10)
        mags.append(v)
    return np.array(mags, np.float32)


E4M3_MAGS = _e4m3_mags()


def fp8_encode(x, scale):
    """x float32 array -> e4m3 bytes with the given global scale."""
    q = np.clip(x / scale, -448.0, 448.0)
    mag = np.abs(q)
    j = np.searchsorted(E4M3_MAGS, mag)
    j = np.clip(j, 1, len(E4M3_MAGS) - 1)
    code = np.where(mag - E4M3_MAGS[j - 1] <= E4M3_MAGS[j] - mag, j - 1, j)
    return (code | (q < 0).astype(np.uint8) << 7).astype(np.uint8)


# ----------------------------------------------------------------------------
# model plan
# ----------------------------------------------------------------------------

LM_PREFIX = "model.language_model."
VIS_PREFIX = "model.visual."

BF16_KEEP = (
    "hc_norm.weight",
    "q_norm.weight",
    "k_norm.weight",
    "q_layernorm.weight",
    "k_layernorm.weight",
    "linear_attn.norm.weight",
    "conv1d.weight",
    "A_log",
    "dt_bias",
    "mlp.gate.weight",
    "ple.key_proj.weight",
    "ple.value_proj.weight",
    "ple.norm_key.weight",
    "ple.norm_query.weight",
    "ple.norm_conv.weight",
    "pre_fc_norm_embedding.weight",
    "pre_fc_norm_hidden.weight",
)

PLE_CFG = (
    "layers.1.ple.ple_embedding.layer_multipliers",
    "layers.1.ple.ple_embedding.ngram_heads_offsets",
    "layers.1.ple.ple_embedding.ngram_heads_vocab_sizes",
)
PLE_TABLE = "layers.1.ple.ngram_embedding.weight"
PLE_SHARD_PAT = "layers.1.ple.ple_embedding.ngram_embedding.shard_%d.weight"

MTP_Q8 = (
    "mtp.fc_embedding.weight",
    "mtp.fc_hidden.weight",
    "mtp.hyper_connection_mixer.input_mix_weight_down.weight",
    "mtp.hyper_connection_mixer.input_mix_weight_up.weight",
    "mtp.layers.0.attn_hyper_connection.input_mix_weight_down.weight",
    "mtp.layers.0.attn_hyper_connection.input_mix_weight_up.weight",
    "mtp.layers.0.attn_hyper_connection.block_inject_weight.weight",
    "mtp.layers.0.mlp_hyper_connection.input_mix_weight_down.weight",
    "mtp.layers.0.mlp_hyper_connection.input_mix_weight_up.weight",
    "mtp.layers.0.mlp_hyper_connection.block_inject_weight.weight",
    "mtp.layers.0.self_attn.q_proj.weight",
    "mtp.layers.0.self_attn.k_proj.weight",
    "mtp.layers.0.self_attn.v_proj.weight",
    "mtp.layers.0.self_attn.o_proj.weight",
    "mtp.layers.0.self_attn.indexer.index_qk_proj.weight",
    "mtp.layers.0.mlp.shared_expert.gate_proj.weight",
    "mtp.layers.0.mlp.shared_expert.up_proj.weight",
    "mtp.layers.0.mlp.shared_expert.down_proj.weight",
)

EXPECTED_CONFIG = {
    "num_hidden_layers": 48,
    "hidden_size": 2560,
    "num_experts": 512,
    "num_experts_per_tok": 10,
    "num_attention_heads": 24,
    "num_key_value_heads": 2,
    "head_dim": 256,
    "vocab_size": 248320,
    "moe_intermediate_size": 640,
    "shared_expert_intermediate_size": 640,
    "hc_count": 4,
    "hc_lowrank": 320,
    "full_attention_interval": 4,
    "linear_num_key_heads": 16,
    "linear_num_value_heads": 48,
    "linear_key_head_dim": 128,
    "linear_value_head_dim": 128,
    "linear_conv_kernel_dim": 4,
}


def check_config(cfg):
    t = cfg.get("text_config", cfg)
    bad = {k: (t.get(k), v) for k, v in EXPECTED_CONFIG.items() if t.get(k) != v}
    if bad:
        raise SystemExit(
            "unsupported architecture shape (engine hardcodes qwen4-exp "
            "48L/2560h/512E): " + json.dumps(bad)
        )


def map_lm_name(src):
    if src.startswith(LM_PREFIX):
        m = src[len(LM_PREFIX) :]
    elif src.startswith("lm_head.") or src.startswith("mtp."):
        m = src
    else:
        return None
    # HF stores the fused 3D expert params as plain Parameters (no .weight);
    # the engine names them with the suffix.
    if m.endswith("experts.gate_up_proj") or m.endswith("experts.down_proj"):
        m += ".weight"
    return m


def base_dtype(mapped):
    if mapped in PLE_CFG:
        return DT_I64
    if any(mapped.endswith(s) for s in BF16_KEEP):
        return DT_BF16
    return DT_Q4CP


# routed experts (trunk layers.N and the MTP layer): [512, rows, cols]
EXP_RE = re.compile(
    r"^(mtp\.)?layers\.(\d+)\.mlp\.experts\.(gate_up_proj|down_proj)\.weight$")

# overlay: few-row matrices that stay bf16 (the engine runs them in bf16 on
# the GGUF path too); everything else in the overlay is q8g32
OVL_BF16 = (
    "linear_attn.in_proj_a.weight",
    "linear_attn.in_proj_b.weight",
    "mlp.shared_expert_gate.weight",
    "block_inject_weight.weight",
    "indexer.index_qk_proj.weight",
)


def overlay_names(base_names, skip=()):
    """Overlay = every q4cp base tensor that is not a routed expert and not
    mtp.* (the MTP sidecar covers those): attention/GDN/HC/shared expert/
    lm_head, plus embed_tokens last (order as tools/hgn_hq.py --embed).
    Names containing a `skip` substring are left out (they stay 4-bit from
    the base)."""
    ns = [m for m in base_names
          if base_dtype(m) == DT_Q4CP and not m.startswith("mtp.")
          and not EXP_RE.match(m) and not any(s in m for s in skip)]
    emb = "embed_tokens.weight"
    return sorted(n for n in ns if n != emb) + ([emb] if emb in ns else [])


# ----------------------------------------------------------------------------
# imatrix (llama.cpp): GGUF (*.gguf / imatrix_unsloth.gguf_file) or legacy .dat
# ----------------------------------------------------------------------------


class Imatrix:
    """Per-expert mean squared input activation, [E, C] per expert tensor."""

    def __init__(self, path):
        self.path = path
        with open(path, "rb") as f:
            magic = f.read(4)
        if magic == b"GGUF":
            from gguf_mini import GGUF
            self.g = GGUF(path)
            self.leg = None
        else:
            self.g = None
            self.leg = self._legacy(path)

    @staticmethod
    def _legacy(path):
        buf = open(path, "rb").read()
        p = 0

        def i32():
            nonlocal p
            v = struct.unpack_from("<i", buf, p)[0]
            p += 4
            return v

        n = i32()
        if not 0 < n < 100000:
            raise SystemExit(f"{path}: neither GGUF nor a legacy imatrix.dat")
        out = {}
        for _ in range(n):
            ln = i32()
            name = buf[p:p + ln].decode("utf-8", "replace")
            p += ln
            ncall, nval = i32(), i32()
            out[name] = (ncall, np.frombuffer(buf, np.float32, nval, p))
            p += 4 * nval
        return out

    def has(self, name):
        if self.g is not None:
            return name + ".in_sum2" in self.g.tensors
        return name in self.leg

    def experts(self, layer, kind, E, C):
        """kind 'gate' (gate_up input) or 'down'. None if the imatrix has no
        entry (e.g. the MTP layer) -> caller falls back to x^2 weighting.
        Experts never routed during calibration get the layer mean."""
        cands = ("gate", "up") if kind == "gate" else ("down",)
        name = next((f"blk.{layer}.ffn_{k}_exps.weight" for k in cands
                     if self.has(f"blk.{layer}.ffn_{k}_exps.weight")), None)
        if name is None:
            return None
        if self.g is not None:
            s = self.g.array(name + ".in_sum2").astype(np.float32).reshape(-1)
            c = self.g.array(name + ".counts").astype(np.float32).reshape(-1)
            if s.size != E * C or c.size != E:
                raise SystemExit(f"imatrix {name}: {s.size}/{c.size} values, "
                                 f"expected {E}x{C}")
            v = s.reshape(E, C) / np.maximum(c, 1)[:, None]
            good = c > 0
        else:
            ncall, vals = self.leg[name]
            if vals.size != E * C:
                raise SystemExit(f"imatrix {name}: {vals.size} values, expected {E}x{C}")
            v = vals.reshape(E, C) / max(ncall, 1)
            good = np.isfinite(v).all(axis=1) & (v.sum(axis=1) > 0)
        if not good.any():
            return None
        v = v.copy()
        if not good.all():
            v[~good] = v[good].mean(axis=0)
        return np.ascontiguousarray(v, np.float32)


# ----------------------------------------------------------------------------
# weighted q4cp for routed experts (multiprocessing, one task = a few experts)
# ----------------------------------------------------------------------------

_WST = {}

CB_SAMPLE = 1 << 20  # values used to train an expert codebook
CB_ROUNDS = 8

# Default routed-expert codebook ("universal"): the shared shape of the
# production w4b.hgn expert codebooks (98 tensors, per-entry std <= 0.004),
# symmetrized and normalized to max|cb| = 1. With the weighted scale search
# it gives KLD 0.0558 vs 0.0587 for per-tensor trained codebooks and 0.0615
# for the classic Lloyd codebook (HGN-HQ.md), although the trained ones have
# lower imatrix-weighted error: that proxy does not rank codebooks reliably.
_UCB_HALF = (0.042080, 0.128362, 0.220943, 0.324402, 0.444132, 0.587296, 0.765293, 1.0)
UNIVERSAL_EXPERT_CB = np.array([-x for x in _UCB_HALF[::-1]] + list(_UCB_HALF), "<f4")


def _cb_weights(g, qw):
    """Per-value weights quant_groups minimizes (imatrix*sqrt(s2+x^2), or x^2)."""
    g2 = g * g
    if qw is None:
        return g2
    return qw * np.sqrt(2.0 * g2.mean(axis=1, keepdims=True) + g2)


def _cb_werr(g, qw, wt, cb):
    nib, s16 = q4cp_imat.quant_groups(g, qw, cb)
    d = q4cp_imat.dequant(nib, s16, cb)
    return float((wt * (d - g) ** 2).sum())


def train_expert_codebook(u16, R, C, iv, cb0, seed):
    """Data-weighted 16-entry codebook for a routed-expert tensor.

    u16: raw bf16 [E*R, C]; iv: [E, C] imatrix or None (-> x^2 weights);
    cb0: the classic codebook (kept if training does not beat it).
    Weighted Lloyd on absmax-normalized values (quantile init), then
    CB_ROUNDS rounds alternating quant_groups (scale search) with a weighted
    least-squares codebook refit; normalized so max|cb| = 1. Uses its own
    rng (seed), so the converter's main rng stream is untouched.
    Returns (cb, err_classic, err_trained) on the sample."""
    rs = np.random.default_rng(seed)
    n = min(u16.shape[0], max(1, CB_SAMPLE // C))
    rows = np.sort(rs.choice(u16.shape[0], size=n, replace=False))
    g = (u16[rows].astype(np.uint32) << 16).view(np.float32).reshape(-1, 32)
    qw = None
    if iv is not None:
        qw = np.ascontiguousarray(iv[rows // R], np.float32).reshape(-1, 32)
    wt = _cb_weights(g, qw)
    # weighted Lloyd in the absmax-normalized domain (error scales with am^2)
    am = np.abs(g).max(axis=1, keepdims=True)
    am = np.where(am == 0, 1.0, am)
    x = (g / am).ravel().astype(np.float64)
    w = (wt * am * am).ravel().astype(np.float64)
    cb = np.quantile(x, (np.arange(16) + 0.5) / 16)
    for _ in range(200):
        a = np.searchsorted((cb[:-1] + cb[1:]) * 0.5, x)
        s = np.bincount(a, weights=x * w, minlength=16)
        c = np.bincount(a, weights=w, minlength=16)
        new = np.where(c > 0, s / np.where(c > 0, c, 1.0), cb)
        done = np.abs(new - cb).max() < 1e-6
        cb = new
        if done:
            break
    del x, w
    cb = np.sort(cb)
    cb = (cb / np.abs(cb).max()).astype(np.float32)
    # alternate: scale search with this codebook, then LS codebook refit
    for _ in range(CB_ROUNDS):
        nib, s16 = q4cp_imat.quant_groups(g, qw, cb)
        sc = s16.astype(np.float32)[:, None]
        num = np.bincount(nib.ravel(), weights=(wt * sc * g).ravel(), minlength=16)
        den = np.bincount(nib.ravel(), weights=(wt * sc * sc).ravel(), minlength=16)
        cb = np.where(den > 0, num / np.where(den > 0, den, 1.0), cb)
        cb = np.sort(cb)
        cb = (cb / np.abs(cb).max()).astype(np.float32)
    e0 = _cb_werr(g, qw, wt, cb0)
    e1 = _cb_werr(g, qw, wt, cb)
    if not (np.isfinite(cb).all() and np.all(np.diff(cb) > 0) and e1 < e0):
        return cb0, e0, e0
    return cb.astype("<f4"), e0, e1


def _qjob(args):
    """Worker: experts e0..e1 of one [E, R, C] bf16 tensor -> (codes, scales)."""
    path, src, e0, e1, R, C, cb, iv = args
    st = _WST.get(path)
    if st is None:
        st = _WST[path] = STFile(path)
    u16 = np.frombuffer(st.raw(src)[2], np.uint16)
    codes, scales = [], []
    for k, e in enumerate(range(e0, e1)):
        w = (u16[e * R * C:(e + 1) * R * C].astype(np.uint32) << 16).view(np.float32)
        qw = None
        if iv is not None:
            qw = np.broadcast_to(iv[k].reshape(1, C), (R, C)).reshape(-1, 32)
        nib, s16 = q4cp_imat.quant_groups(w.reshape(-1, 32), qw, cb)
        c, s = q4cp_imat.pack(nib, s16, R, C)
        codes.append(c)
        scales.append(s)
    if DONTNEED is not None:
        a = st.base + st.hdr[src]["data_offsets"][0]
        os.posix_fadvise(st.fd, a + e0 * R * C * 2, (e1 - e0) * R * C * 2, DONTNEED)
    return b"".join(codes), b"".join(scales)


def expert_spot_check(st, src, out_path, off, E, R, C, iv, ref_cb=None):
    """Weighted rel. error of experts 0 and E-1 read back from the output
    vs the data-free absmax quantizer with ref_cb (the classic codebook,
    i.e. what --classic writes; default: the written codebook). Returns
    (absmax_err, written_err); written must be lower."""
    u16 = np.frombuffer(st.raw(src)[2], np.uint16)
    stride = ((C // 32 * 2) + 15) & ~15
    res = [0.0, 0.0]
    with open(out_path, "rb") as f:
        f.seek(off)
        cb = np.frombuffer(f.read(64), np.float32).copy()
        for e in (0, E - 1):
            x = (u16[e * R * C:(e + 1) * R * C].astype(np.uint32) << 16).view(
                np.float32).reshape(R, C)
            qw = (iv[e].reshape(1, C) if iv is not None else x * x)
            f.seek(off + 64 + e * R * C // 2)
            cd = np.frombuffer(f.read(R * C // 2), np.uint8).reshape(R, C // 2)
            f.seek(off + 64 + E * R * C // 2 + e * R * stride)
            sc = np.frombuffer(f.read(R * stride), np.uint8).reshape(R, stride)
            sc = sc[:, : C // 32 * 2].copy().view(np.float16).astype(np.float32)
            nib = np.empty((R, C), np.uint8)
            nib[:, 0::2] = cd & 15
            nib[:, 1::2] = cd >> 4
            dw = (cb[nib].reshape(R, C // 32, 32) * sc[:, :, None]).reshape(R, C)
            rc = cb if ref_cb is None else ref_cb
            nn, a16 = q4cp_imat.naive_groups(x.reshape(-1, 32), rc)
            dn = q4cp_imat.dequant(nn, a16, rc).reshape(R, C)
            den = float((x * x * qw).sum()) or 1.0
            res[0] += float(((dn - x) ** 2 * qw).sum()) / den
            res[1] += float(((dw - x) ** 2 * qw).sum()) / den
    return res[0] / 2, res[1] / 2


# ----------------------------------------------------------------------------
# hgn writer
# ----------------------------------------------------------------------------


class HgnWriter:
    """Plan first (names/dtypes/dims/sizes), then stream blobs in order."""

    def __init__(self, path, model_name):
        self.path = path
        self.model_name = model_name
        self.entries = []  # (name, dtype, dims, size)
        self.written = 0

    def plan(self, name, dtype, dims, size):
        assert len(name.encode()) < 96
        assert 1 <= len(dims) <= 4
        self.entries.append((name, dtype, tuple(dims), size))

    def begin(self):
        n = len(self.entries)
        self.data_off = (HDR + REC * n + 511) & ~511  # 512B, as production
        off = self.data_off
        offs = []
        for _, _, _, size in self.entries:
            offs.append(off)
            off += size + (-size) % 64
        self.offs = offs
        total = off
        hdr = bytearray(HDR)
        struct.pack_into("<4sIIIQQQ", hdr, 0, b"HGN1", 2, n, 0, HDR,
                         self.data_off, total)
        nb = self.model_name.encode()[:63]
        hdr[0x28 : 0x28 + len(nb)] = nb
        self.f = open(self.path, "wb")
        self.f.write(hdr)
        for (name, dt, dims, size), o in zip(self.entries, offs):
            rec = bytearray(REC)
            nb = name.encode()
            rec[0 : len(nb)] = nb
            struct.pack_into("<II", rec, 96, dt, len(dims))
            for k, d in enumerate(dims):
                struct.pack_into("<Q", rec, 104 + 8 * k, d)
            struct.pack_into("<QQQ", rec, 136, o, size, 0)
            self.f.write(rec)
        # pad header+records up to the 512B-aligned data start
        self.f.write(b"\0" * (self.data_off - self.f.tell()))
        self._cur = 0
        return self

    def write(self, blob):
        """Write the next planned blob; size must match the plan."""
        name, _, _, size = self.entries[self._cur]
        assert len(blob) == size, f"{name}: blob {len(blob)} != planned {size}"
        self.f.write(blob)
        pad = (-size) % 64
        if pad:
            self.f.write(b"\0" * pad)
        self._cur += 1

    def write_stream(self, parts):
        """Like write(), but the blob comes as an iterable of byte chunks."""
        name, _, _, size = self.entries[self._cur]
        n = 0
        for p in parts:
            self.f.write(p)
            n += len(p)
        assert n == size, f"{name}: streamed {n} != planned {size}"
        pad = (-size) % 64
        if pad:
            self.f.write(b"\0" * pad)
        self._cur += 1

    def drop_cache(self):
        """Flush and drop the written pages from the page cache (keeps a
        100+ GiB conversion from filling RAM; bytes unchanged)."""
        self.f.flush()
        if DONTNEED is not None:
            os.fdatasync(self.f.fileno())
            os.posix_fadvise(self.f.fileno(), 0, 0, DONTNEED)

    def finish(self):
        assert self._cur == len(self.entries)
        self.f.close()


# ----------------------------------------------------------------------------
# conversion
# ----------------------------------------------------------------------------


def human(n):
    return f"{n / 2**30:.2f} GiB"


def write_start_script(outdir, name, has_mtp, has_vision, engine_dir,
                       has_overlay=False):
    """Emit a self-contained start.sh next to the converted model."""
    def pick(*cands):
        for c in cands:
            p = os.path.join(engine_dir, c)
            if os.path.exists(p):
                return os.path.realpath(p)
        raise SystemExit(f"engine binary not found in {engine_dir} "
                         f"(tried {', '.join(cands)}); pass --engine-bin")

    engine_bin = pick("gdec", "gdec")
    api_bin = pick("gdec-api")
    run_capped = os.path.realpath(os.path.join(engine_dir, os.pardir,
                                               "tools", "run_capped.sh"))
    if not os.path.exists(run_capped):
        run_capped = ""
    script = f"""#!/usr/bin/env bash
# Auto-generated by flashnext2hgn.py — start engine + OpenAI API for {name}.
#   bash start.sh          (foreground; Ctrl+C stops both)
# Overrides: ENGINE_PORT=8730 API_PORT=8731 MAXCTX=262144 GAMMA=3 MEMCAP_GB=86 LOG_DIR=$HERE/logs
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${{BASH_SOURCE[0]}}")" && pwd)"
ENGINE_PORT="${{ENGINE_PORT:-8730}}"
API_PORT="${{API_PORT:-8731}}"
MAXCTX="${{MAXCTX:-262144}}"
GAMMA="${{GAMMA:-3}}"
MEMCAP_GB="${{MEMCAP_GB:-86}}"
ENGINE="{engine_bin}"
API="{api_bin}"
LOG_DIR="${{LOG_DIR:-$HERE/logs}}"
mkdir -p "$LOG_DIR"
LOG_ENGINE="$LOG_DIR/engine.log"
LOG_API="$LOG_DIR/api.log"

for port in "$ENGINE_PORT" "$API_PORT"; do
  [[ -z "$(ss -H -ltn "sport = :$port")" ]] || {{ echo "端口 $port 已被占用" >&2; exit 1; }}
done
if procs="$(pgrep -af '(^|/)(gdec[^/[:space:]]*|serve_api\\.py)([[:space:]]|$)')"; then
  echo "已有引擎/API 进程在跑：$procs" >&2; exit 1
fi

export GDEC_QSA_KV_BF16=1 GDEC_QSA_WMMA=1 GDEC_QSA_WMMA_BTV=1
export GDEC_MOE_LT=1 GDEC_MOE_LT_BF16=1 GDEC_GR_BF16=1
export GDEC_GDN_STREAM=1 GDEC_GDN_WAVE=1 GDEC_NOWARMUP=1
export GDEC_PREFILL_CHUNK=16384
export GDEC_INDEX_FUSED2=1 GDEC_PP_MOE_OUT=1 GDEC_INDEX_STREAM_SELECT=1
export GDEC_KVSNAP=1 GDEC_KVSNAP_MAX_GB=20
export GDEC_KVSNAP_DIR="${{GDEC_KVSNAP_DIR:-$HERE/kvsnap}}"
export GDEC_SPEC_GAMMA="$GAMMA"

engine_cmd=("$ENGINE" "$HERE/{name}.hgn")
"""
    if has_overlay:
        script += f'engine_cmd+=("$HERE/{name}.overlay.hgn")\n'
    if has_mtp:
        script += f'engine_cmd+=("$HERE/{name}-mtp.hgn")\n'
    script += 'engine_cmd+=(--serve --port "$ENGINE_PORT" --maxctx "$MAXCTX")\n'
    if has_vision:
        script += f'engine_cmd+=(--vision-tower "$HERE/{name}-vision.hgn")\n'
    script += f"""
echo "加载模型中(日志 $LOG_ENGINE);Ctrl+C 停止"
if [[ -x "{run_capped}" ]]; then
  setsid bash "{run_capped}" "$MEMCAP_GB" -- "${{engine_cmd[@]}}" >"$LOG_ENGINE" 2>&1 &
else
  setsid "${{engine_cmd[@]}}" >"$LOG_ENGINE" 2>&1 &
fi
engine_pid=$!
api_pid=''
cleanup() {{
  trap - EXIT INT TERM
  [[ -z "$api_pid" ]] || kill "$api_pid" 2>/dev/null || true
  kill -TERM -- "-$engine_pid" 2>/dev/null || true
  sleep 2
  kill -KILL -- "-$engine_pid" 2>/dev/null || true
  [[ -z "$api_pid" ]] || kill -KILL "$api_pid" 2>/dev/null || true
  echo "已停止"
}}
trap cleanup EXIT INT TERM

begin=$SECONDS
until grep -q 'serve: listening' "$LOG_ENGINE" 2>/dev/null; do
  kill -0 "$engine_pid" 2>/dev/null || {{ tail -n 15 "$LOG_ENGINE" >&2; echo "引擎提前退出,见 $LOG_ENGINE" >&2; exit 1; }}
  (( SECONDS - begin < 900 )) || {{ echo "引擎加载超时,见 $LOG_ENGINE" >&2; exit 1; }}
  sleep 2
done
"$API" --tokenizer "$HERE/tokenizer" --engine "127.0.0.1:$ENGINE_PORT" \
  --host 0.0.0.0 --port "$API_PORT" --context "$MAXCTX" >"$LOG_API" 2>&1 &
api_pid=$!
begin=$SECONDS
until [[ -n "$(ss -H -ltn "sport = :$API_PORT")" ]]; do
  kill -0 "$api_pid" 2>/dev/null || {{ tail -n 15 "$LOG_API" >&2; echo "API 提前退出,见 $LOG_API" >&2; exit 1; }}
  (( SECONDS - begin < 15 )) || {{ echo "API 启动超时,见 $LOG_API" >&2; exit 1; }}
  sleep 0.2
done
echo "就绪:http://127.0.0.1:$API_PORT/v1(引擎日志 $LOG_ENGINE)"
wait -n "$engine_pid" "$api_pid"
"""
    path = os.path.join(outdir, "start.sh")
    with open(path, "w") as f:
        f.write(script)
    os.chmod(path, 0o755)
    return path


def drop_src(st, src):
    """Drop a source tensor's pages from the page cache (bytes unchanged)."""
    if DONTNEED is not None:
        _, _, (a, b) = st.meta(src)
        os.posix_fadvise(st.fd, st.base + a, b - a, DONTNEED)


def write_overlay(path, model_name, base_src, skip=(), check=25):
    """8-bit dense overlay: q8g32 (dtype 8) for every overlay_names() tensor,
    bf16 for the OVL_BF16 few-row ones. Byte-identical (after the header
    name) to tools/hgn_hq.py --embed on the same safetensors."""
    names = overlay_names(base_src, skip)
    tmp = path + ".part"
    w = HgnWriter(tmp, model_name)
    kinds = []
    for n in names:
        _st, sdt, shape, _src = base_src[n]
        if sdt != "BF16":
            raise SystemExit(f"overlay {n}: source dtype {sdt}, expected BF16")
        R, C = int(np.prod(shape[:-1])), shape[-1]
        if any(n.endswith(s) for s in OVL_BF16):
            kinds.append("bf16")
            w.plan(n, DT_BF16, shape, R * C * 2)
        else:
            if C % 32:
                raise SystemExit(f"overlay {n}: cols {C} not a multiple of 32")
            kinds.append("q8")
            w.plan(n, DT_Q8G32, shape, q8g32_size(R, C))
    w.begin()
    worst = (0.0, "")
    t0 = time.time()
    for i, (n, k) in enumerate(zip(names, kinds)):
        st, _sdt, shape, src = base_src[n]
        R, C = int(np.prod(shape[:-1])), shape[-1]
        if k == "bf16":
            w.write(bytes(st.raw(src)[2]))
        else:
            x = st.f32(src).reshape(R, C)
            blob = quant_q8g32(x)
            if check and i % check == 0:
                ref = x.astype(np.float64)
                e = np.linalg.norm(dequant_q8g32(blob, R, C) - ref) / (np.linalg.norm(ref) or 1.0)
                worst = max(worst, (float(e), n))
            w.write(blob)
            del x, blob
        drop_src(st, src)
        if (i + 1) % 100 == 0 or i + 1 == len(names):
            print(f"  overlay [{i + 1}/{len(names)}] {time.time() - t0:.0f}s", flush=True)
    w.finish()
    if worst[0] > 1e-2:
        raise SystemExit(f"FAIL: overlay q8g32 rel-L2 {worst[0]:.2e} on {worst[1]}")
    os.replace(tmp, path)
    print(f"wrote {path} ({human(os.path.getsize(path))}; {kinds.count('q8')} q8g32 + "
          f"{kinds.count('bf16')} bf16; worst sampled q8 rel-L2 {worst[0]:.1e})", flush=True)


def convert(model_dir, outdir, name, skip_vision, skip_ple, skip_mtp_sidecar,
            dry_run=False, engine_dir=None, classic=False, imatrix=None, jobs=1,
            overlay=True, only_overlay=False, overlay_skip=(), per_job=4,
            expert_cb="universal"):
    """classic=True reproduces the original data-free converter bit for bit
    (absmax q4cp experts, no overlay). Otherwise (HQ, default): routed
    experts get weighted q4cp (imatrix if given, else x^2 weighting) with
    the universal expert codebook (expert_cb="trained": per-tensor trained;
    all other codebooks as classic), and
    <name>.overlay.hgn holds the dense weights in 8-bit."""
    t_start = time.time()
    hq = not classic
    if classic and (imatrix or only_overlay):
        raise SystemExit("--classic cannot be combined with --imatrix/--only-overlay")
    overlay = overlay and hq
    cfg = json.load(open(os.path.join(model_dir, "config.json")))
    check_config(cfg)
    print(f"scanning {model_dir} ...", flush=True)
    table = scan_model_dir(model_dir)
    print(f"  {len(table)} source tensors", flush=True)

    # ---- partition source tensors ----
    ple_shards = {}
    base_src, vis_src, mtp_src = {}, {}, {}
    for src_name, (st, dt, shape) in table.items():
        if src_name.startswith(VIS_PREFIX):
            vis_src["visual." + src_name[len(VIS_PREFIX) :]] = (
                st, dt, shape, src_name)
            continue
        m = map_lm_name(src_name)
        if m is None:
            raise SystemExit(f"unmapped source tensor: {src_name}")
        if m.startswith("layers.1.ple.ple_embedding.ngram_embedding.shard_"):
            s = int(m.rsplit("shard_", 1)[1].split(".")[0])
            ple_shards[s] = (st, dt, shape, src_name)
            continue
        if m.startswith("mtp."):
            mtp_src[m] = (st, dt, shape, src_name)
            base_src[m] = (st, dt, shape, src_name)  # q4cp fallback in base
        else:
            base_src[m] = (st, dt, shape, src_name)

    if ple_shards and sorted(ple_shards) != list(range(128)):
        raise SystemExit(f"PLE shard set incomplete: {len(ple_shards)}/128")
    if not ple_shards and not skip_ple:
        print("WARNING: no PLE ngram shards found; base will lack the PLE "
              "table (engine falls back to ple=off)")

    rng = np.random.default_rng(0x5EED)
    ovl_path = os.path.join(outdir, f"{name}.overlay.hgn")

    imat = None
    if hq:
        n_exp = sum(1 for m in base_src if EXP_RE.match(m))
        if imatrix:
            imat = Imatrix(imatrix)
            miss = sorted({int(EXP_RE.match(m).group(2)) for m in base_src
                           if EXP_RE.match(m) and not EXP_RE.match(m).group(1)
                           and not imat.has(f"blk.{EXP_RE.match(m).group(2)}.ffn_"
                                            f"{'down' if 'down_proj' in m else 'gate'}_exps.weight")})
            if miss:
                raise SystemExit(f"imatrix {imatrix} has no expert entries for layers {miss}")
            print(f"HQ mode: {n_exp} expert tensors, imatrix-weighted q4cp "
                  f"({'GGUF' if imat.g is not None else 'legacy .dat'} imatrix; MTP "
                  f"layer uses x^2 weighting unless the imatrix has blk.48)", flush=True)
        else:
            print(f"HQ mode: {n_exp} expert tensors, x^2-weighted q4cp (no imatrix)",
                  flush=True)
        if overlay:
            nov = overlay_names(base_src, overlay_skip)
            print(f"  overlay: {len(nov)} dense tensors -> 8-bit {ovl_path}", flush=True)
    else:
        print("classic mode: data-free absmax q4cp, no overlay", flush=True)

    if only_overlay:
        if dry_run:
            return
        write_overlay(ovl_path, f"{name}.overlay", base_src, overlay_skip)
        print(f"done in {(time.time() - t_start) / 60:.1f} min; launch with "
              f"<base>.hgn {ovl_path} [<mtp>.hgn]")
        return

    # ---- base file ----
    base_path = os.path.join(outdir, f"{name}.hgn")
    w = HgnWriter(base_path, name)
    order = sorted(base_src.keys())  # stable; loader is order-independent
    for m in order:
        _, _, shape, _ = base_src[m]
        dt = base_dtype(m)
        if dt == DT_Q4CP:
            cols = shape[-1]
            rows = int(np.prod(shape)) // cols
            size, _ = q4cp_size(rows, cols)
        elif dt == DT_I64:
            size = int(np.prod(shape)) * 8
        else:
            size = int(np.prod(shape)) * 2
        w.plan(m, dt, shape, size)
    if ple_shards:
        sh0 = ple_shards[0][2]
        n_el = 128 * int(np.prod(sh0))
        w.plan(PLE_TABLE, DT_FP8, (128,) + tuple(sh0), n_el + 4)
    if dry_run:
        from collections import Counter
        print("DRY RUN — no files written")
        for tag, entries in (("base", w.entries),):
            hist = Counter(e[1] for e in entries)
            sz = Counter()
            for e in entries:
                sz[e[1]] += e[3]
            print(f"  {tag}: {len(entries)} tensors, dtypes "
                  f"{dict(hist)}, bytes { {k: human(v) for k, v in sz.items()} }")
        n_q8 = sum(1 for m in MTP_Q8 if m in mtp_src)
        print(f"  mtp sidecar: {n_q8} q8g64 tensors planned")
        print(f"  vision: {len(vis_src)} tensors (bf16)")
        return
    pool = None
    if hq and jobs > 1:
        import multiprocessing as mp
        pool = mp.Pool(jobs)
    w.begin()
    total_bytes = sum(e[3] for e in w.entries)
    done_bytes = 0
    bad = []
    n_layers = EXPECTED_CONFIG["num_hidden_layers"]
    for i, m in enumerate(order):
        st, sdt, shape, src = base_src[m]
        dt = base_dtype(m)
        t0 = time.time()
        size = w.entries[i][3]
        tag = f"dt{dt}"
        mt = EXP_RE.match(m) if hq else None
        if dt == DT_BF16 or dt == DT_I64:
            want = "BF16" if dt == DT_BF16 else "I64"
            if sdt != want:
                raise SystemExit(f"{m}: expected {want} source, got {sdt}")
            w.write(bytes(st.raw(src)[2]))
        elif mt:
            # weighted q4cp, streamed. The classic codebook is still drawn
            # (keeps the main rng stream, so dense codebooks match --classic),
            # then replaced by a codebook trained on the same weighting.
            if sdt != "BF16" or len(shape) != 3:
                raise SystemExit(f"{m}: expected a BF16 [E, R, C] expert tensor")
            E, R, C = shape
            layer = n_layers if mt.group(1) else int(mt.group(2))
            kind = "gate" if mt.group(3) == "gate_up_proj" else "down"
            iv = imat.experts(layer, kind, E, C) if imat is not None else None
            u16 = np.frombuffer(st.raw(src)[2], np.uint16).reshape(E * R, C)
            cb_classic = q4cp_codebook(u16, rng).astype("<f4")
            if expert_cb == "trained":
                tc = time.time()
                cb, ce0, ce1 = train_expert_codebook(u16, R, C, iv, cb_classic,
                                                     seed=zlib.crc32(m.encode()))
                print(f"    codebook {m}: sample w-err {ce0:.4e} -> {ce1:.4e} "
                      f"(x{ce0 / max(ce1, 1e-30):.3f}, {time.time() - tc:.1f}s)", flush=True)
            else:
                cb = UNIVERSAL_EXPERT_CB
            del u16
            if not (np.diff(cb) >= 0).all():
                raise SystemExit(f"{m}: codebook not sorted")
            tl = [(st.path, src, e, min(E, e + per_job), R, C, cb,
                   None if iv is None else iv[e:e + per_job])
                  for e in range(0, E, per_job)]
            it = pool.imap(_qjob, tl) if pool is not None else map(_qjob, tl)

            def parts(it=it, cb=cb):
                yield cb.tobytes()
                sc = []
                for c, s in it:
                    yield c
                    sc.append(s)
                yield from sc

            w.write_stream(parts())
            w.drop_cache()
            tag = "q4i+imat" if iv is not None else "q4i x^2"
            if layer in (0, n_layers // 2, n_layers - 1, n_layers):
                # pass/fail: the scale search must beat absmax with the same
                # codebook (trained: the classic codebook it replaced). The
                # universal codebook is tuned for real heavy-tailed experts, so
                # vs-classic is informational only (weighted error misranks
                # codebooks; see UNIVERSAL_EXPERT_CB).
                ref = cb_classic if expert_cb == "trained" else None
                eo, en = expert_spot_check(st, src, base_path, w.offs[i], E, R, C, iv,
                                           ref_cb=ref)
                extra = ""
                if ref is None:
                    ec, _ = expert_spot_check(st, src, base_path, w.offs[i], E, R, C, iv,
                                              ref_cb=cb_classic)
                    extra = f"; classic file {ec:.3e}"
                print(f"    check {m}: weighted err absmax {eo:.3e} -> {en:.3e} "
                      f"(x{eo / max(en, 1e-30):.2f}{extra})", flush=True)
                if not en < eo:
                    bad.append(m)
        else:
            w.write(quant_q4cp(st.f32(src).reshape(-1, shape[-1]), rng))
        if size > (256 << 20):
            w.drop_cache()
            drop_src(st, src)
        done_bytes += size
        el = time.time() - t0
        if mt or size > (64 << 20) or i % 100 == 0 or not hq:
            eta = (time.time() - t_start) / done_bytes * (total_bytes - done_bytes) / 60
            print(f"  base [{i + 1}/{len(order)}] {m} {shape} {tag} "
                  f"{human(size)} in {el:.1f}s "
                  f"({100 * done_bytes / total_bytes:.0f}%, eta {eta:.0f} min)", flush=True)
    if pool is not None:
        pool.close()
        pool.join()
    if ple_shards:
        write_ple(w, ple_shards)
        w.drop_cache()
    w.finish()
    if bad:
        raise SystemExit(f"FAIL: weighted expert error did not improve on {bad}")
    print(f"wrote {base_path} ({human(os.path.getsize(base_path))})", flush=True)

    # ---- 8-bit dense overlay ----
    if overlay:
        write_overlay(ovl_path, f"{name}.overlay", base_src, overlay_skip)

    # ---- MTP sidecar ----
    if mtp_src and not skip_mtp_sidecar:
        side_path = os.path.join(outdir, f"{name}-mtp.hgn")
        ws = HgnWriter(side_path, name + "-mtp")
        for m in sorted(MTP_Q8):
            if m not in mtp_src:
                raise SystemExit(f"missing MTP tensor: {m}")
            _, _, shape, _ = mtp_src[m]
            cols = shape[-1]
            rows = int(np.prod(shape)) // cols
            ws.plan(m, DT_Q8G64, shape, q8g64_size(rows, cols))
        ws.begin()
        for m in sorted(MTP_Q8):
            st, _, shape, src = mtp_src[m]
            t0 = time.time()
            blob = quant_q8g64(st.f32(src).reshape(-1, shape[-1]))
            ws.write(blob)
            print(f"  mtp8 {m} {shape} {human(len(blob))} "
                  f"in {time.time() - t0:.1f}s", flush=True)
            del blob
        ws.finish()
        print(f"wrote {side_path} ({human(os.path.getsize(side_path))})",
              flush=True)
    elif mtp_src:
        print("MTP sidecar skipped (--skip-mtp-sidecar); base keeps the "
              "q4cp mtp.* fallback")

    # ---- vision ----
    if vis_src and not skip_vision:
        vis_path = os.path.join(outdir, f"{name}-vision.hgn")
        wv = HgnWriter(vis_path, name + "-vision")
        for m in sorted(vis_src.keys()):
            _, sdt, shape, _ = vis_src[m]
            if m == "visual.patch_embed.proj.weight":
                shape = (shape[0], int(np.prod(shape[1:])))
            wv.plan(m, DT_BF16, shape, int(np.prod(shape)) * 2)
        wv.begin()
        for m in sorted(vis_src.keys()):
            st, sdt, shape, src = vis_src[m]
            _, _, buf = st.raw(src)
            if sdt != "BF16":
                raise SystemExit(f"vision {m}: expected BF16, got {sdt}")
            wv.write(bytes(buf))
        wv.finish()
        print(f"wrote {vis_path} ({human(os.path.getsize(vis_path))})",
              flush=True)
    elif vis_src:
        print("vision tower skipped (--skip-vision)")

    # ---- tokenizer ----
    tok_dir = os.path.join(outdir, "tokenizer")
    os.makedirs(tok_dir, exist_ok=True)
    for f in ("tokenizer.json", "vocab.json", "merges.txt",
              "tokenizer_config.json", "chat_template.jinja",
              "generation_config.json", "preprocessor_config.json"):
        src = os.path.join(model_dir, f)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(tok_dir, f))
    print(f"tokenizer files -> {tok_dir}", flush=True)

    el = time.time() - t_start
    if engine_dir is None:
        engine_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  os.pardir, "build")
    sp = write_start_script(outdir, name,
                            bool(mtp_src) and not skip_mtp_sidecar,
                            bool(vis_src) and not skip_vision, engine_dir,
                            has_overlay=overlay)
    print(f"\ndone in {el / 60:.1f} min. Start the service with:")
    print(f"  bash {sp}")


def write_ple(w, ple_shards):
    """Stack 128 bf16 shards -> one fp8 e4m3 tensor + trailing global scale."""
    print("  PLE table: pass 1/2 (global absmax over 128 shards) ...",
          flush=True)
    absmax = 0.0
    t0 = time.time()
    for s in range(128):
        st, _, _, src = ple_shards[s]
        x = st.f32(src)
        absmax = max(absmax, float(np.abs(x).max()))
        del x
        drop_src(st, src)
        if (s + 1) % 16 == 0:
            print(f"    absmax {s + 1}/128 ({time.time() - t0:.0f}s)",
                  flush=True)
    scale = np.float32(absmax / 448.0)
    print(f"  PLE table: absmax={absmax:.4f} scale={scale:.6e}; "
          f"pass 2/2 (quantize+write) ...", flush=True)
    # The blob is one tensor; write through a temp stream by buffering per
    # shard into one bytearray is 47 GiB — instead the writer's blob API
    # takes the whole blob, so stream to a side file and copy in chunks.
    # Simpler: build the blob incrementally in a spare file, then stream-copy.
    tmp = w.path + ".ple-tmp"
    with open(tmp, "wb") as f:
        for s in range(128):
            st, _, _, src = ple_shards[s]
            x = st.f32(src)
            f.write(fp8_encode(x, scale).tobytes())
            del x
            drop_src(st, src)
            if (s + 1) % 16 == 0:
                print(f"    quantize {s + 1}/128 ({time.time() - t0:.0f}s)",
                      flush=True)
        f.write(np.float32(scale).tobytes())
    with open(tmp, "rb") as f:
        shutil.copyfileobj(f, w.f, length=1 << 28)
    os.unlink(tmp)
    # account for this entry in the writer
    w._cur += 1
    pad = (-(w.entries[w._cur - 1][3])) % 64
    if pad:
        w.f.write(b"\0" * pad)
    print(f"  PLE table done ({time.time() - t0:.0f}s)", flush=True)


# ----------------------------------------------------------------------------
# self-test: quantizer round-trip error on synthetic + real-ish distributions
# ----------------------------------------------------------------------------


def dequant_q4cp(blob, rows, cols):
    cb = np.frombuffer(blob[:64], dtype="<f4")
    codes = np.frombuffer(blob[64 : 64 + rows * cols // 2], np.uint8).reshape(
        rows, cols // 2)
    stride = ((cols // 32 * 2) + 15) & ~15
    sc = np.frombuffer(blob[64 + rows * cols // 2 :], np.uint8).reshape(rows, stride)
    nib = np.empty((rows, cols), np.uint8)
    nib[:, 0::2] = codes & 0xF
    nib[:, 1::2] = codes >> 4
    s = sc[:, : cols // 32 * 2].copy().view(np.float16).astype(np.float32)
    s = np.repeat(s, 32, axis=1)
    return cb[nib] * s


def dequant_q8g64(blob, rows, cols):
    G = cols // 64
    stride = cols + G * 4
    raw = np.frombuffer(blob, np.uint8).reshape(rows, stride)
    code = raw[:, :cols].astype(np.float32)
    sm = raw[:, cols:].copy().view(np.float16).astype(np.float32).reshape(rows, G, 2)
    out = code.reshape(rows, G, 64) * sm[:, :, 0:1] + sm[:, :, 1:2]
    return out.reshape(rows, cols)


def quant_selftest():
    rng = np.random.default_rng(1)
    for dist, gen in (
        ("normal", lambda s: rng.normal(0, 0.02, s)),
        ("heavy-tail", lambda s: rng.standard_t(3, s) * 0.01),
    ):
        w = gen((512, 2560)).astype(np.float32)
        amax = np.abs(w).max()

        def report(tag, dq):
            err = np.abs(dq - w) / amax  # vs tensor absmax, not per-element
            print(f"{tag} {dist:10s} max={err.max():.4f} "
                  f"mean={err.mean():.5f} (of absmax)")

        report("q4cp", dequant_q4cp(quant_q4cp(w, rng), 512, 2560))
        report("q8g64", dequant_q8g64(quant_q8g64(w), 512, 2560))
        scale = np.float32(amax / 448.0)
        c = fp8_encode(w, scale)
        mag = E4M3_MAGS[c & 0x7F] * np.where(c & 0x80, -1, 1) * scale
        report("fp8  ", mag)
    print("quant selftest done")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model_dir", nargs="?")
    ap.add_argument("--out", help="output directory")
    ap.add_argument("--name", help="output model name (default: dir name)")
    ap.add_argument("--skip-vision", action="store_true")
    ap.add_argument("--skip-ple", action="store_true")
    ap.add_argument("--skip-mtp-sidecar", action="store_true")
    ap.add_argument("--quant-selftest", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="validate mapping and print the plan; write nothing")
    ap.add_argument("--engine-bin",
                    help="dir containing the gdec/gdec-api binaries "
                         "(default: ../build relative to this script)")
    ap.add_argument("--start-script-only", action="store_true",
                    help="only (re)write OUTDIR/start.sh for an existing "
                         "conversion; no model work")
    ap.add_argument("--imatrix", metavar="FILE",
                    help="llama.cpp imatrix (GGUF, e.g. imatrix_unsloth.gguf_file, "
                         "or legacy imatrix.dat) to weight the routed-expert "
                         "quantization; without it x^2 weighting is used")
    ap.add_argument("--classic", action="store_true",
                    help="original data-free converter, bit-identical output "
                         "(absmax q4cp experts, no 8-bit overlay)")
    ap.add_argument("--expert-codebook", choices=("universal", "trained"), default="universal",
                    help="routed-expert q4cp codebook: universal (default, best KLD) or "
                         "trained per tensor on the weighted error")
    ap.add_argument("--skip-overlay", action="store_true",
                    help="do not write <name>.overlay.hgn (8-bit dense weights)")
    ap.add_argument("--only-overlay", action="store_true",
                    help="only write <name>.overlay.hgn (~1 min; pairs with an "
                         "existing base)")
    ap.add_argument("--overlay-skip", default="", metavar="SUB1,SUB2",
                    help="leave overlay tensors whose name contains one of these "
                         "out (they stay 4-bit from the base), e.g. lm_head")
    ap.add_argument("--jobs", type=int,
                    default=max(1, min(32, (os.cpu_count() or 4) - 4)),
                    help="worker processes for the expert quantization "
                         "(default: min(32, cores-4); ~0.7 GiB RAM each)")
    a = ap.parse_args()
    if a.quant_selftest:
        quant_selftest()
        return
    if not a.model_dir or not a.out:
        ap.error("MODEL_DIR and --out are required")
    name = a.name or os.path.basename(os.path.normpath(a.model_dir)).lower()
    if a.start_script_only:
        engine_dir = a.engine_bin or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), os.pardir, "build")
        sp = write_start_script(
            a.out, name,
            os.path.exists(os.path.join(a.out, f"{name}-mtp.hgn")),
            os.path.exists(os.path.join(a.out, f"{name}-vision.hgn")),
            engine_dir,
            has_overlay=os.path.exists(os.path.join(a.out, f"{name}.overlay.hgn")))
        print(f"wrote {sp} — run: bash {sp}")
        return
    if a.imatrix and not os.path.isfile(a.imatrix):
        ap.error(f"--imatrix {a.imatrix}: no such file")
    if not a.dry_run:
        os.makedirs(a.out, exist_ok=True)
    convert(a.model_dir, a.out, name, a.skip_vision, a.skip_ple,
            a.skip_mtp_sidecar, a.dry_run, a.engine_bin,
            classic=a.classic, imatrix=a.imatrix, jobs=a.jobs,
            overlay=not a.skip_overlay, only_overlay=a.only_overlay,
            overlay_skip=tuple(s for s in a.overlay_skip.split(",") if s),
            expert_cb=a.expert_codebook)


if __name__ == "__main__":
    main()
