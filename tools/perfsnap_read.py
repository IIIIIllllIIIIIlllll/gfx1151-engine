#!/usr/bin/env python
# perfsnap_read.py — 轮询引擎行协议的 SNAP 动词（验证托盘 tooltip 的数据源，
# 模拟 launcher 读端：3s 轮询 + 6s TTL 判过期）。
# 用法:
#   python tools/perfsnap_read.py [engine_port]            # 查一次
#   python tools/perfsnap_read.py [engine_port] --poll 1   # 每 N 秒查一次
import socket
import sys
import time

TTL_MS = 6000
STATES = {0: "idle", 1: "prefill", 2: "decode"}


def snap(port):
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=2) as s:
            s.settimeout(2)
            s.sendall(b"SNAP\n")
            data = b""
            while not data.endswith(b"\n"):
                chunk = s.recv(256)
                if not chunk:
                    break
                data += chunk
    except OSError as e:
        return None, "connect/recv failed: %s" % e
    f = data.split()
    if len(f) != 10 or f[0] != b"P":
        return None, "bad reply: %r" % data
    return {
        "ts_ms": int(f[1]), "state": int(f[2]), "req_id": int(f[3]),
        "pp_toks": float(f[4]), "prompt_done": int(f[5]),
        "prompt_total": int(f[6]), "tg_inst": float(f[7]),
        "tg_avg": float(f[8]), "accept_pct": float(f[9]),
    }, None


def show(d):
    age = time.time() * 1000 - d["ts_ms"]
    stale = " STALE(tooltip 回归纯地址)" if age > TTL_MS else ""
    st = STATES.get(d["state"], "?")
    if d["state"] == 1:
        extra = "pp %.0f tok/s · %d/%d" % (d["pp_toks"], d["prompt_done"],
                                           d["prompt_total"])
    elif d["state"] == 2:
        extra = "tg %.1f tok/s (avg %.1f)" % (d["tg_inst"], d["tg_avg"])
        if d["accept_pct"] >= 0:
            extra += " · accept %.1f%%" % d["accept_pct"]
    else:
        extra = ""
    print("[%s] state=%s req=%d age=%.1fs%s %s"
          % (time.strftime("%H:%M:%S"), st, d["req_id"], age / 1000, stale,
             extra), flush=True)


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 8730
    poll = float(sys.argv[sys.argv.index("--poll") + 1]) if "--poll" in sys.argv else 0
    while True:
        d, err = snap(port)
        if d:
            show(d)
        else:
            print("[%s] %s" % (time.strftime("%H:%M:%S"), err), flush=True)
        if not poll:
            return 0 if d else 1
        time.sleep(poll)


if __name__ == "__main__":
    sys.exit(main())
