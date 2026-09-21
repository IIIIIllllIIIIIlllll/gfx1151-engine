#!/usr/bin/env python3
"""flashnext2hgn.py — convert a Qwen3.8-Flash-Next (qwen4_exp) HF safetensors
model into the .hgn checkpoint format consumed by gdec (gfx1151).

Self-contained: stdlib + numpy only (no torch, no safetensors).

Usage:
  flashnext2hgn.py MODEL_DIR --out OUTDIR [--name NAME]
                 [--skip-vision] [--skip-ple] [--skip-mtp-sidecar]
  flashnext2hgn.py --quant-selftest

Outputs (into OUTDIR):
  <name>.hgn          base checkpoint (q4cp linears, bf16 norms, fp8 PLE table)
  <name>-mtp.hgn      MTP sidecar (q8g64 linears; optional at engine launch)
  <name>-vision.hgn   vision tower (all bf16; optional --vision-tower arg)
  tokenizer/          tokenizer files copied from MODEL_DIR

Only this exact architecture shape is supported (the engine hardcodes it):
48 layers (GDN + every-4th QSA), hidden 2560, 512 experts, 24 heads,
vocab 248320, MTP 1 layer, vision depth 27. config.json is validated.

Quantization is data-free RTN: q4cp uses a per-tensor 16-level Lloyd codebook
in group-absmax-normalized space (groups of 32 columns, fp16 scales);
q8g64 is per-64-column affine uint8; the PLE table is fp8 e4m3 with one
global scale. No calibration data is needed.

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
import shutil
import struct
import sys
import time

import numpy as np

DT_BF16, DT_I64, DT_Q4CP, DT_Q8G64, DT_FP8 = 0, 4, 5, 7, 10
HDR, REC = 104, 160

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


def quant_q4cp(w, rng):
    """w: float32 [R, C], C%32==0 -> (blob bytes, max_rel_err_sample)."""
    R, C = w.shape
    assert C % 32 == 0
    G = C // 32
    blob_size, stride = q4cp_size(R, C)
    # codebook from a normalized row subsample
    n_rows = min(R, max(1, (1 << 21) // C))
    ridx = rng.choice(R, size=n_rows, replace=False) if n_rows < R else np.arange(R)
    sub = w[ridx].reshape(-1, 32)
    am = np.abs(sub).max(axis=1, keepdims=True)
    xn = (sub / np.where(am == 0, 1.0, am)).ravel()
    cb = lloyd_codebook_normalized(xn[:: max(1, xn.size // (1 << 21))])
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

    def finish(self):
        assert self._cur == len(self.entries)
        self.f.close()


# ----------------------------------------------------------------------------
# conversion
# ----------------------------------------------------------------------------


def human(n):
    return f"{n / 2**30:.2f} GiB"


def write_start_script(outdir, name, has_mtp, has_vision, engine_dir):
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


def convert(model_dir, outdir, name, skip_vision, skip_ple, skip_mtp_sidecar,
            dry_run=False, engine_dir=None):
    t_start = time.time()
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
    w.begin()
    total_bytes = sum(e[3] for e in w.entries)
    done_bytes = 0
    for i, m in enumerate(order):
        st, sdt, shape, src = base_src[m]
        dt = base_dtype(m)
        t0 = time.time()
        if dt == DT_BF16 or dt == DT_I64:
            want = "BF16" if dt == DT_BF16 else "I64"
            if sdt != want:
                raise SystemExit(f"{m}: expected {want} source, got {sdt}")
            blob = bytes(st.raw(src)[2])
        else:
            blob = quant_q4cp(st.f32(src).reshape(-1, shape[-1]), rng)
        w.write(blob)
        done_bytes += len(blob)
        el = time.time() - t0
        print(f"  base [{i + 1}/{len(order)}] {m} {shape} dt{dt} "
              f"{human(len(blob))} in {el:.1f}s "
              f"({100 * done_bytes / total_bytes:.0f}%)", flush=True)
        del blob
    if ple_shards:
        write_ple(w, ple_shards)
    w.finish()
    print(f"wrote {base_path} ({human(os.path.getsize(base_path))})", flush=True)

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
                            bool(vis_src) and not skip_vision, engine_dir)
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
            engine_dir)
        print(f"wrote {sp} — run: bash {sp}")
        return
    if not a.dry_run:
        os.makedirs(a.out, exist_ok=True)
    convert(a.model_dir, a.out, name, a.skip_vision, a.skip_ple,
            a.skip_mtp_sidecar, a.dry_run, a.engine_bin)


if __name__ == "__main__":
    main()
