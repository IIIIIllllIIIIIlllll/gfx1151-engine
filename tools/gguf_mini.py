"""gguf_mini.py — minimal read-only GGUF parser (KV + tensor views over mmap).

Enough for llama.cpp imatrix files (*.gguf, tensors "<name>.in_sum2" /
"<name>.counts", F32) and plain F32/F16/BF16 tensors. No dependencies
beyond numpy.
"""

import mmap
import struct

import numpy as np

_SCALAR = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?",
           10: "<Q", 11: "<q", 12: "<d"}
# ggml type -> (numpy dtype, block elems, block bytes) for the types we view
_TYPES = {0: (np.float32, 1, 4), 1: (np.float16, 1, 2), 30: (np.uint16, 1, 2)}


class GGUF:
    def __init__(self, path):
        self.f = open(path, "rb")
        self.mm = mmap.mmap(self.f.fileno(), 0, access=mmap.ACCESS_READ)
        self.p = 0
        if self._r("<4s") != b"GGUF":
            raise ValueError(f"{path}: not GGUF")
        self.version = self._r("<I")
        nt, nkv = self._r("<Q"), self._r("<Q")
        self.kv = {}
        for _ in range(nkv):
            k = self._str()
            self.kv[k] = self._val(self._r("<I"))
        infos = []
        for _ in range(nt):
            name = self._str()
            nd = self._r("<I")
            dims = [self._r("<Q") for _ in range(nd)]
            infos.append((name, dims, self._r("<I"), self._r("<Q")))
        al = int(self.kv.get("general.alignment", 32))
        base = (self.p + al - 1) // al * al
        self.tensors = {n: (d, t, base + o) for n, d, t, o in infos}

    def _r(self, fmt):
        v = struct.unpack_from(fmt, self.mm, self.p)
        self.p += struct.calcsize(fmt)
        return v[0] if len(v) == 1 else v

    def _str(self):
        n = self._r("<Q")
        s = bytes(self.mm[self.p:self.p + n]).decode("utf-8", "replace")
        self.p += n
        return s

    def _val(self, t):
        if t in _SCALAR:
            return self._r(_SCALAR[t])
        if t == 8:
            return self._str()
        if t == 9:
            et, n = self._r("<I"), self._r("<Q")
            return [self._val(et) for _ in range(n)]
        raise ValueError(f"gguf kv type {t}")

    def array(self, name):
        """numpy view, shape in torch order (reversed ggml ne)."""
        dims, t, off = self.tensors[name]
        dt, be, bb = _TYPES[t]
        n = int(np.prod(dims))
        a = np.frombuffer(self.mm, dt, n, off)
        if t == 30:
            a = (a.astype(np.uint32) << 16).view(np.float32)
        return a.reshape(tuple(reversed(dims)))
