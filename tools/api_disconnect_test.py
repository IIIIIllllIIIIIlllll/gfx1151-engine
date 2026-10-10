#!/usr/bin/env python3
"""CPU-only disconnect regression using a fake engine and synthetic tokenizer.

Run: python tools/api_disconnect_test.py --api build/qwenox-win.exe
     python tools/api_disconnect_test.py --api build/qwenox-api --slots 2
"""
import argparse
import json
import os
from pathlib import Path
import socket
import struct
import subprocess
import sys
import tempfile
import time

from api_regression_test import check, get, request


def wait_for(predicate, name, timeout=3.5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError(f"{name}: timed out after {timeout}s")


def free_port():
    with socket.socket() as conn:
        conn.bind(("127.0.0.1", 0))
        return conn.getsockname()[1]


def make_tokenizer(directory):
    byte_values = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    alphabet = {value: value for value in byte_values}
    extra = 256
    for value in range(256):
        if value not in alphabet:
            alphabet[value] = extra
            extra += 1
    vocab = {chr(alphabet[value]): 1024 + value for value in range(256)}
    vocab.update({"Hi": 12675, chr(alphabet[32]) + "Par": 4130, "is": 284})
    special = [("<|endoftext|>", 248043), ("<|im_start|>", 248045),
               ("<|im_end|>", 248046), ("<think>", 248068), ("</think>", 248069)]
    tokenizer = {"model": {"vocab": vocab, "merges": []}, "added_tokens": [
        {"content": content, "id": token_id, "special": True}
        for content, token_id in special
    ]}
    directory.mkdir()
    (directory / "tokenizer.json").write_text(json.dumps(tokenizer), encoding="utf-8")


def run_checks(base, api_port, engine_port, slots):
    def state():
        with socket.create_connection(("127.0.0.1", engine_port), timeout=2) as conn:
            conn.sendall(b"TESTSTATE\n")
            with conn.makefile("rb") as reader:
                return json.loads(reader.readline())

    def health():
        return json.loads(get(base, "/health")[2])

    def idle():
        status = health()
        return status["in_flight"] == 0 and status["queued"] == 0

    def body_for(path, stream=False, seed=616161):
        body = {"stream": stream, "temperature": 1, "seed": seed,
                "max_tokens": 65536, "enable_thinking": False}
        if path == "/v1/completions":
            body["prompt"] = "hi"
        elif path == "/v1/chat/completions":
            body["messages"] = [{"role": "user", "content": "hi"}]
        else:
            body["input"] = "hi"
        return body

    def open_request(path, body):
        payload = json.dumps(body).encode("utf-8")
        conn = socket.create_connection(("127.0.0.1", api_port), timeout=2)
        header = (f"POST {path} HTTP/1.1\r\nHost: localhost\r\n"
                  f"Content-Type: application/json\r\nContent-Length: {len(payload)}\r\n"
                  "Connection: close\r\n\r\n").encode("ascii")
        conn.sendall(header + payload)
        return conn

    def next_request():
        status, content_type, raw = request(base, "/v1/chat/completions", {
            "messages": [{"role": "user", "content": "hi"}],
            "enable_thinking": False, "temperature": 0, "max_tokens": 4,
        })
        check(status == 200 and "application/json" in content_type and
              json.loads(raw)["choices"][0]["message"]["content"] == "Hi",
              "next-request-after-draining", raw[:300])

    paths = ("/v1/chat/completions", "/v1/completions", "/v1/responses")
    for path in paths:
        status, content_type, raw = request(base, path, body_for(path, seed=616163))
        check(status == 200 and "application/json" in content_type and raw,
              f"nonstream-normal-completion {path}")
        check(state()[-1]["cancel"] == 0, f"live-client-not-cancelled {path}")

    for seed in (616161, 616162):
        for stream in (False, True):
            for path in paths:
                before = len(state())
                with open_request(path, body_for(path, stream, seed)) as conn:
                    wait_for(lambda: len(state()) == before + 1, "generation-started")
                    if seed == 616161:
                        wait_for(lambda: state()[-1]["tokens"] > 0, "continuous-tokens")
                    if stream:
                        check(b"text/event-stream" in conn.recv(4096), "sse-started")
                    else:
                        conn.settimeout(0.1)
                        try:
                            data = conn.recv(1)
                        except socket.timeout:
                            data = None
                        check(data is None, "nonstream-no-premature-response")
                wait_for(idle, f"disconnect seed={seed} stream={stream} {path}")
                entry = state()[-1]
                check(entry["cancel"] == 1 and entry["done"],
                      f"cancel-once-and-drain seed={seed} stream={stream} {path}", entry)
                next_request()

    before = len(state())
    with open_request(paths[0], body_for(paths[0])) as conn:
        wait_for(lambda: len(state()) == before + 1, "reset-generation-started")
        conn.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER,
                       struct.pack("hh" if os.name == "nt" else "ii", 1, 0))
    wait_for(idle, "reset-client-disconnect")
    check(state()[-1]["cancel"] == 1 and state()[-1]["done"], "reset-cancel-and-drain")
    next_request()

    owners = []
    before = len(state())
    try:
        for unused in range(slots):
            owners.append(open_request(paths[0], body_for(paths[0])))
        wait_for(lambda: len(state()) == before + slots, "all-slots-occupied")
        with open_request(paths[0], body_for(paths[0], seed=616162)):
            wait_for(lambda: health()["queued"] == 1, "disconnected-request-queued")
        wait_for(lambda: health()["queued"] == 0, "disconnected-request-left-queue")
        check(len(state()) == before + slots, "queued-disconnect-never-sends-gen")
    finally:
        for conn in owners:
            conn.close()
    wait_for(idle, "queue-owners-cancelled")
    next_request()

    status, _, raw = request(base, paths[1], dict(body_for(paths[1]), stop="Hi"))
    check(status == 200 and json.loads(raw)["choices"][0]["text"] == "",
          "stop-sequence-cancels", raw[:300])
    check(state()[-1]["cancel"] == 1 and state()[-1]["done"],
          "no-heartbeat-after-token-cancellation")
    next_request()
    print("RESULT PASS")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", required=True, help="path to the compiled qwenox-api binary")
    parser.add_argument("--slots", type=int, default=1)
    args = parser.parse_args()
    api_binary = str(Path(args.api).resolve())
    fake_engine = str(Path(__file__).with_name("api_fake_engine.py").resolve())
    engine_port, api_port = free_port(), free_port()
    while engine_port == api_port:
        api_port = free_port()
    base = f"http://127.0.0.1:{api_port}"
    children = []
    with tempfile.TemporaryDirectory(prefix="qwenox-api-disconnect-") as temporary:
        root = Path(temporary)
        make_tokenizer(root / "tokenizer")
        env = dict(os.environ, QWENOX_API_TOKCACHE_FILE="", QWENOX_API_ADMIN_KEY="",
                   QWENOX_API_TOKCACHE="0", QWENOX_REQSTAT="0", ROPE_FACTOR="1")
        creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        with (root / "test.log").open("w+") as log:
            try:
                children.append(subprocess.Popen(
                    [sys.executable, fake_engine, "--port", str(engine_port),
                     "--slots", str(args.slots)], cwd=root, stdout=log, stderr=log,
                    creationflags=creationflags))

                def engine_ready():
                    try:
                        with socket.create_connection(("127.0.0.1", engine_port), timeout=0.1):
                            return True
                    except OSError:
                        return False

                wait_for(engine_ready, "fake-engine-ready")
                children.append(subprocess.Popen(
                    [api_binary, "--tokenizer", str(root / "tokenizer"),
                     "--engine", f"127.0.0.1:{engine_port}", "--host", "127.0.0.1",
                     "--port", str(api_port), "--overrides", ""],
                    cwd=root, env=env, stdout=log, stderr=log, creationflags=creationflags))

                def api_ready():
                    try:
                        return get(base, "/health")[0] == 200
                    except OSError:
                        return False

                wait_for(api_ready, "api-ready", timeout=10)
                run_checks(base, api_port, engine_port, args.slots)
            except Exception:
                log.flush()
                log.seek(0)
                print(log.read(), file=sys.stderr)
                raise
            finally:
                for child in reversed(children):
                    child.terminate()
                    child.wait(timeout=5)


if __name__ == "__main__":
    main()
