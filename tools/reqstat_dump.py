#!/usr/bin/env python3
# reqstat_dump.py — 读取 gdec-api 的请求统计文件（格式见 src/api/reqstat.h 头注释）。
#
# 用法:
#   tools/reqstat_dump.py summary [--from YYYY-MM-DD] [--to YYYY-MM-DD] [--dir data]
#   tools/reqstat_dump.py tail [N] [--dir data]
#   tools/reqstat_dump.py files [--dir data]
#
# 时间过滤按记录 ts_ms 二分定位后前后线性收口（不假设严格单调）。
import argparse
import datetime as dt
import os
import struct
import sys

MAGIC = 0x47525153  # "GRQS"
HSZ, RSZ = 256, 64
REC = struct.Struct("<QQ10IHHI")
assert REC.size == RSZ

FINISH = {1: "stop", 2: "length", 3: "cancel", 4: "error"}
DRAFTER = {0: "serial", 1: "mtp", 3: "ngram", 4: "chain"}


def crc16(data: bytes) -> int:
    c = 0xFFFF
    for b in data:
        c ^= b << 8
        for _ in range(8):
            c = ((c << 1) ^ 0x1021) & 0xFFFF if c & 0x8000 else (c << 1) & 0xFFFF
    return c


class StatFile:
    def __init__(self, path):
        self.path = path
        with open(path, "rb") as f:
            h = f.read(HSZ)
            if len(h) < HSZ or struct.unpack_from("<I", h, 0)[0] != MAGIC:
                raise ValueError(f"{path}: 不是 reqstat 文件")
            self.seq = struct.unpack_from("<I", h, 12)[0]
            self.hdr_count = struct.unpack_from("<Q", h, 16)[0]
            self.data = f.read()
        # 真相是文件大小：头部 count 只是提示
        self.count = len(self.data) // RSZ
        self.bad_crc = 0

    def record(self, i):
        raw = self.data[i * RSZ : (i + 1) * RSZ]
        if crc16(raw[:56]) != struct.unpack_from("<H", raw, 56)[0]:
            self.bad_crc += 1
            return None
        f = REC.unpack(raw)
        return {
            "ts": f[0], "seq": f[1], "prompt": f[2], "cached": f[3],
            "gen": f[4], "proposed": f[5], "commit": f[6], "rounds": f[7],
            "prefill_us": f[8], "decode_us": f[9], "ttft_us": f[10],
            "flags": f[11],
        }

    def lower_bound(self, ts_ms):
        lo, hi = 0, self.count  # 第一条 ts >= ts_ms（近似，调用方线性收口）
        while lo < hi:
            mid = (lo + hi) // 2
            ts = struct.unpack_from("<Q", self.data, mid * RSZ)[0]
            if ts < ts_ms:
                lo = mid + 1
            else:
                hi = mid
        return lo


def date_ms(s, end_of_day=False):
    d = dt.date.fromisoformat(s)
    t = dt.time(23, 59, 59, 999999) if end_of_day else dt.time(0, 0)
    return int(dt.datetime.combine(d, t, dt.timezone.utc).timestamp() * 1000)


def open_chain(dirpath):
    files = []
    for name in os.listdir(dirpath):
        p = os.path.join(dirpath, name)
        if name == "reqstat.bin" or (
            name.startswith("reqstat-") and name.endswith(".bin")
            and name[8:14].isdigit()
        ):
            try:
                files.append(StatFile(p))
            except ValueError as e:
                print(f"跳过: {e}", file=sys.stderr)
    files.sort(key=lambda f: f.seq)
    return files


def iter_range(files, t0, t1):
    for sf in files:
        if sf.count == 0:
            continue
        first = struct.unpack_from("<Q", sf.data, 0)[0]
        last = struct.unpack_from("<Q", sf.data, (sf.count - 1) * RSZ)[0]
        if last < t0 or first > t1:
            continue
        # 二分近似定位，前后各退 64 条线性收口（ts 允许局部乱序）
        i = max(0, sf.lower_bound(t0) - 64)
        while i < sf.count:
            r = sf.record(i)
            i += 1
            if r is None:
                continue
            if r["ts"] < t0:
                continue
            if r["ts"] > t1:
                break
            yield r


def summarize(recs, label):
    recs = list(recs)
    n = len(recs)
    if n == 0:
        print(f"{label}: 无记录")
        return
    tp = sum(r["prompt"] for r in recs)
    tc = sum(r["cached"] for r in recs)
    tg = sum(r["gen"] for r in recs)
    dec_s = sum(r["decode_us"] for r in recs) / 1e6
    pre_s = sum(r["prefill_us"] for r in recs) / 1e6
    ttft = sorted(r["ttft_us"] for r in recs if r["ttft_us"] > 0)
    print(f"{label}: {n} 请求")
    print(
        f"  输入 {tp:,} token（缓存命中 {tc:,}，{100.0 * tc / tp:.1f}%）"
        if tp else "  输入 0 token"
    )
    print(f"  输出 {tg:,} token")
    if dec_s > 0:
        print(f"  decode {dec_s:.1f}s（{tg / dec_s:.1f} tok/s 均值）"
              f"  prefill {pre_s:.1f}s")
    if ttft:
        p50 = ttft[len(ttft) // 2] / 1000.0
        p95 = ttft[int(len(ttft) * 0.95)] / 1000.0
        print(f"  TTFT p50={p50:.0f}ms p95={p95:.0f}ms（{len(ttft)} 条流式）")
    by_fin = {}
    for r in recs:
        by_fin[FINISH.get(r["flags"] & 0xF, "?")] = by_fin.get(
            FINISH.get(r["flags"] & 0xF, "?"), 0) + 1
    print(f"  finish: {by_fin}")
    # 各草稿器的接受率
    for dval, dname in sorted(DRAFTER.items()):
        sub = [r for r in recs if (r["flags"] >> 4 & 0xF) == dval and r["proposed"] > 0]
        if not sub:
            continue
        prop = sum(r["proposed"] for r in sub)
        acc = sum(r["commit"] - r["rounds"] for r in sub)
        print(f"  {dname}: {len(sub)} 请求，接受率 {100.0 * acc / prop:.1f}%"
              f"（{acc:,}/{prop:,}）")


def main():
    ap = argparse.ArgumentParser(description="gdec reqstat 统计读取")
    ap.add_argument("cmd", choices=["summary", "tail", "files"])
    ap.add_argument("n", nargs="?", type=int, default=10)
    ap.add_argument("--dir", default="data")
    ap.add_argument("--from", dest="frm")
    ap.add_argument("--to", dest="to")
    a = ap.parse_args()

    files = open_chain(a.dir)
    if a.cmd == "files":
        for sf in files:
            print(f"seq={sf.seq:<6} records={sf.count:<9} hdr_count={sf.hdr_count:<9} {sf.path}")
        return

    if a.cmd == "tail":
        allr = []
        for sf in files:
            for i in range(sf.count):
                r = sf.record(i)
                if r:
                    allr.append(r)
        for r in allr[-a.n:]:
            t = dt.datetime.fromtimestamp(r["ts"] / 1000, dt.timezone.utc)
            fin = FINISH.get(r["flags"] & 0xF, "?")
            dr = DRAFTER.get(r["flags"] >> 4 & 0xF, "?")
            acc = (
                f" acc={100.0 * (r['commit'] - r['rounds']) / r['proposed']:.0f}%"
                if r["proposed"] else ""
            )
            print(
                f"{t:%Y-%m-%d %H:%M:%S} req={r['seq']} {fin:6s} {dr:6s} "
                f"prompt={r['prompt']}(命中{r['cached']}) gen={r['gen']} "
                f"ttft={r['ttft_us'] / 1000:.0f}ms prefill={r['prefill_us'] / 1000:.0f}ms "
                f"decode={r['decode_us'] / 1000:.0f}ms{acc}"
            )
        return

    # summary
    t0 = date_ms(a.frm) if a.frm else 0
    t1 = date_ms(a.to, True) if a.to else (1 << 62)
    label = "summary"
    if a.frm or a.to:
        label = f"{a.frm or '...'} ~ {a.to or '...'}"
    summarize(iter_range(files, t0, t1), label)
    bad = sum(sf.bad_crc for sf in files)
    if bad:
        print(f"警告：{bad} 条记录 CRC 不匹配（已跳过）", file=sys.stderr)


if __name__ == "__main__":
    main()
