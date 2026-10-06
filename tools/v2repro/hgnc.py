"""HGN container + safetensors IO for the v2 reproduction pipeline.

Layout follows src/hgn.h (Header 104B / Record 160B, 64B-aligned payloads):
  header: magic "HGN1" | u32 ver | u32 count | u32 reserved |
          u64 records_off | u64 data_off | u64 file_size | name[64]
  record: name[96] | u32 dtype | u32 ndims | u64 dims[4] |
          u64 data_offset | u64 data_size | u64 extra
  extra = checksum(u32 xor-fold) | variant(u32) << 32
"""
import json
import os
import struct

import numpy as np

HDR, ENT = 0x68, 0xA0
MAGIC = 0x314E4748  # "HGN1"


class Entry:
    __slots__ = ("name", "store", "shape", "offset", "size", "checksum", "variant")

    def pack(self):
        b = bytearray(ENT)
        n = self.name.encode()
        assert 1 <= len(n) <= 95, self.name
        b[: len(n)] = n
        struct.pack_into("<II", b, 0x60, self.store, len(self.shape))
        struct.pack_into("<4q", b, 0x68, *(list(self.shape) + [0] * (4 - len(self.shape))))
        struct.pack_into("<QQII", b, 0x88, self.offset, self.size, self.checksum, self.variant)
        return bytes(b)


def read(path):
    """-> (ver, ident, data_off, [Entry])"""
    with open(path, "rb") as f:
        h = f.read(HDR)
        if len(h) != HDR:
            raise ValueError(f"truncated HGN header: {path}")
        magic, ver, n, _res, toff, doff, fsize = struct.unpack("<IIIIQQQ", h[:0x28])
        actual = os.path.getsize(path)
        if magic != MAGIC or fsize != actual:
            raise ValueError(f"invalid HGN magic or file size: {path}")
        ident = h[0x28:0x68].rstrip(b"\0").decode()
        f.seek(toff)
        ents = []
        for _ in range(n):
            b = f.read(ENT)
            if len(b) != ENT:
                raise ValueError(f"truncated HGN tensor table: {path}")
            e = Entry()
            e.name = b[:96].split(b"\0")[0].decode()
            e.store, rank = struct.unpack_from("<II", b, 0x60)
            if rank > 4:
                raise ValueError(f"invalid rank for {e.name}: {rank}")
            e.shape = tuple(struct.unpack_from("<4q", b, 0x68)[:rank])
            e.offset, e.size, e.checksum, e.variant = struct.unpack_from("<QQII", b, 0x88)
            if e.offset + e.size > actual:
                raise ValueError(f"payload outside file: {e.name}")
            ents.append(e)
    return ver, ident, doff, ents


def write_header(f, ver, n, doff, fsize, ident):
    idb = ident.encode()
    assert len(idb) <= 64
    f.seek(0)
    f.write(struct.pack("<IIIIQQQ", MAGIC, ver, n, 0, HDR, doff, fsize))
    f.write(idb + b"\0" * (64 - len(idb)))


def xor_fold(buf):
    a = np.frombuffer(buf, dtype=np.uint8)
    pad = (-len(a)) % 4
    if pad:
        a = np.concatenate([a, np.zeros(pad, np.uint8)])
    return int(np.bitwise_xor.reduce(a.view("<u4"))) if len(a) else 0


def payload(path, e):
    return np.memmap(path, dtype=np.uint8, mode="r", offset=e.offset, shape=(e.size,))


def align64(x):
    return (x + 63) // 64 * 64


# ------------------------------------------------------------- BF16 source
class SafeTensors:
    """Indexed BF16 safetensors directory (uint16 bits view; f32 on demand)."""

    def __init__(self, d):
        self.d = os.path.realpath(os.fspath(d))
        with open(os.path.join(self.d, "model.safetensors.index.json"), encoding="utf-8") as f:
            wm = json.load(f)["weight_map"]
        self.loc = {}
        for name, shard in wm.items():
            p = os.path.join(self.d, shard)
            if name in self.loc:
                raise ValueError(f"duplicate tensor: {name}")
            with open(p, "rb") as fh:
                hn = struct.unpack("<Q", fh.read(8))[0]
                hdr = json.loads(fh.read(hn))
            spec = hdr[name]
            shape = tuple(spec["shape"])
            start, end = spec["data_offsets"]
            dt = spec["dtype"]
            unit = {"BF16": 2, "F16": 2, "I64": 8, "F32": 4, "I32": 4}[dt]
            assert end - start == int(np.prod(shape, dtype=np.int64)) * unit, name
            self.loc[name] = (p, 8 + hn + start, shape, dt)

    def raw(self, name, lo=None, hi=None):
        """uint16 bits; optional first-axis slice [lo:hi]."""
        p, off, shape, dt = self.loc[name]
        assert dt == "BF16", (name, dt)
        first = shape[0] if shape else 1
        per = int(np.prod(shape[1:])) if len(shape) > 1 else 1
        lo = 0 if lo is None else lo
        hi = first if hi is None else hi
        return np.fromfile(p, dtype="<u2", count=(hi - lo) * per,
                           offset=off + lo * per * 2).reshape(([hi - lo] + list(shape[1:])) if shape else ())

    def f32(self, name, lo=None, hi=None):
        return (self.raw(name, lo, hi).astype(np.uint32) << 16).view(np.float32)


LM_PREFIX = "model.language_model."


def map_name(bf16_name):
    """HF safetensors name -> hgn name (mirrors tools/flashnext2hgn.py)."""
    if bf16_name.startswith(LM_PREFIX):
        m = bf16_name[len(LM_PREFIX):]
    elif bf16_name.startswith(("lm_head.", "mtp.")):
        m = bf16_name
    else:
        return None
    if m.endswith(("experts.gate_up_proj", "experts.down_proj")):
        m += ".weight"
    return m
