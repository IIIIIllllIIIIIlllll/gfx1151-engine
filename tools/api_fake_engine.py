#!/usr/bin/env python3
"""Deterministic line-protocol engine used by api_regression_test.py.

Run this on port 18730, start gdec-api with --engine 127.0.0.1:18730 and
--port 18731, then run api_regression_test.py against that API port.
"""
import argparse
import socket
import struct
import threading
import time


def send_line(conn, line):
    conn.sendall((line + "\n").encode("ascii"))


SLOTS = 1


def handle(conn):
    pending = b""

    def read_exact(size):
        nonlocal pending
        while len(pending) < size:
            chunk = conn.recv(min(65536, size - len(pending)))
            if not chunk:
                return None
            pending += chunk
        result, pending = pending[:size], pending[size:]
        return result

    with conn:
        while True:
            while b"\n" not in pending:
                chunk = conn.recv(65536)
                if not chunk:
                    return
                pending += chunk
            raw, pending = pending.split(b"\n", 1)
            line = raw.decode("ascii", "strict")
            if line == "PING":
                send_line(conn, "PONG")
                continue
            if line == "INFO":
                send_line(conn, f"I 1 0 262144 8 1 1 0 0 0 {SLOTS} 262144 0 1")
                continue
            if line == "MEM":
                send_line(conn, "M 1 100 120 200 10 20 310 900 1000 100 400 50")
                continue
            if line == "CSTAT":
                send_line(conn, "C fake")
                continue
            if line.startswith("X "):
                continue
            if not line.startswith("GEN "):
                continue

            fields = line.split()
            req = int(fields[1])
            n_eos = int(fields[3])
            n_ids_at = 4 + n_eos
            n_ids = int(fields[n_ids_at])
            ids = [int(v) for v in fields[n_ids_at + 1:n_ids_at + 1 + n_ids]]

            grids = []
            if "MROPE" in fields:
                at = fields.index("MROPE")
                count = int(fields[at + 1])
                values = [int(value) for value in fields[at + 2:at + 2 + count * 3]]
                if len(values) != count * 3:
                    return
                grids = [tuple(values[index:index + 3])
                         for index in range(0, len(values), 3)]

            frame_ok = True
            if "VIMG" in fields:
                at = fields.index("VIMG")
                count = int(fields[at + 1])
                frame_ok = count == len(grids) and count <= 8
                for index in range(count):
                    header = read_exact(16)
                    if header is None:
                        return
                    magic, patches, nbytes = struct.unpack("<IIQ", header)
                    if nbytes > 512 * 1024 * 1024:
                        return
                    payload = read_exact(nbytes)
                    if payload is None:
                        return
                    expected = grids[index][0] * grids[index][1] * grids[index][2]
                    frame_ok = (frame_ok and magic == 0x56494D31 and
                                patches == expected and nbytes == patches * 1536 * 4)
            elif grids:
                frame_ok = False
            if not frame_ok:
                send_line(conn, f"D {req} error 0 0 0.0 0.0 0 0 0 0 0")
                continue

            if "424242" in fields:
                return
            if "31337" in fields:
                time.sleep(2)
            if "5150" in fields:  # aborted mid-decode (shared KV pool full)
                send_line(conn, f"T {req} 12675 -0.125")
                send_line(conn, f"D {req} error 0 0 0.0 0.0 0 0 0 0 0")
                continue
            if "9001" in fields:
                tokens = [
                    248069, 271, 248058, 198, 27, 1628, 27362, 67017, 29, 198,
                    27, 15704, 28, 8656, 29, 198, 98116, 198, 510, 15704, 29,
                    198, 27, 15704, 28, 13382, 29, 198, 18, 198, 510, 15704,
                    29, 198, 510, 1628, 29, 198, 248059,
                ]
                tokens = [(token, -0.1) for token in tokens]
            elif "9002" in fields:
                tokens = [27, 15704, 28, 41843, 29, 198, 37186, 73594, 29086,
                          198, 510, 15704, 29, 198, 510, 1628, 29, 198, 248059]
                tokens = [(token, -0.1) for token in tokens]
            elif "9003" in fields:
                tokens = [447, 67017, 29, 198, 27, 15704, 28, 8656, 29, 198,
                          98817, 198, 510, 15704, 29, 198, 510, 1628, 29, 198,
                          248059]
                tokens = [(token, -0.1) for token in tokens]
            elif n_ids <= 2:
                tokens = [(4130, -0.5), (284, -0.25)]  # " Par" + "is"
            elif 248069 in ids:
                tokens = [(12675, -0.125)]  # enable_thinking=false -> "Hi"
            else:
                tokens = [(248069, -0.75), (12675, -0.125)]  # "</think>Hi"
            for token, logprob in tokens:
                send_line(conn, f"T {req} {token} {logprob}")
            send_line(conn, f"D {req} done {n_ids} {len(tokens)} 1.0 2.0 0 0 0 0 0")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=18730)
    parser.add_argument("--slots", type=int, default=1, help="INFO kv_slots")
    args = parser.parse_args()
    global SLOTS
    SLOTS = args.slots
    with socket.socket() as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("127.0.0.1", args.port))
        server.listen(16)
        while True:
            conn, _ = server.accept()
            threading.Thread(target=handle, args=(conn,), daemon=True).start()


if __name__ == "__main__":
    main()
