#!/usr/bin/env python3
"""gdec --serve 行协议探针（Windows 移植验证用）。
PING -> PONG；INFO -> 一行字段；GEN 1 8 0 3 1 2 3 -> T 行流 + D 终止行。
用法: python tools/winprobe/serve_probe.py [port]"""
import socket
import sys

port = int(sys.argv[1]) if len(sys.argv) > 1 else 8730
s = socket.create_connection(("127.0.0.1", port), timeout=30)
f = s.makefile("r", encoding="utf-8", newline="\n")


def send(line):
    s.sendall((line + "\n").encode("utf-8"))


send("PING")
pong = f.readline().strip()
assert pong == "PONG", f"PING 应答异常: {pong!r}"
print("PASS PING -> PONG")

send("INFO")
info = f.readline().strip()
print("INFO:", info[:200])
assert info, "INFO 应答为空"

send("GEN 1 8 0 3 1 2 3")
tokens = []
done = None
while True:
    line = f.readline()
    if not line:
        break
    parts = line.split()
    if not parts:
        continue
    if parts[0] == "T":
        tokens.append(int(parts[2]))
    elif parts[0] == "D":
        done = line.strip()
        break
print("GEN tokens:", tokens)
print("GEN done:", done)
assert done is not None and "error" not in done, f"GEN 失败: {done}"
assert len(tokens) > 0, "GEN 没有产出 token"
print("== SERVE PROBE PASS ==")
