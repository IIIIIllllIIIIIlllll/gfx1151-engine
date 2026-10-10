#!/usr/bin/env python3
"""Dry-run ENGINE_HOST regression with fake files and mocked system probes.

Run: python tools/engine_host_config_test.py --bash /path/to/bash
Windows: also pass --launcher /path/to/compiled/start_win.exe
No engine, model, or API is started; all launchers run with --check.
"""
import argparse
import os
from pathlib import Path
import shutil
import socket
import subprocess
import tempfile


def free_port():
    with socket.socket() as conn:
        conn.bind(("127.0.0.1", 0))
        return conn.getsockname()[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bash", default="bash")
    parser.add_argument("--launcher", help="compiled Windows launcher for native --check tests")
    args = parser.parse_args()
    source = Path(__file__).resolve().parent.parent
    original_conf = (source / "service.conf").read_text(encoding="utf-8")
    original_conf = "\n".join(line for line in original_conf.splitlines()
                              if not line.startswith("ENGINE_HOST=")) + "\n"
    engine_port, api_port = free_port(), free_port()
    while engine_port == api_port:
        api_port = free_port()
    with tempfile.TemporaryDirectory(prefix="qwenox-engine-host-") as temporary:
        root = Path(temporary)
        (root / "tools").mkdir()
        (root / "build").mkdir()
        (root / "build" / "rocblas" / "library").mkdir(parents=True)
        (root / "build" / "hipblaslt" / "library").mkdir(parents=True)
        (root / "bin").mkdir()
        (root / "models" / "tokenizer").mkdir(parents=True)
        for name in ("start_hgn.sh", "start_gguf.sh", "start_win.sh", "tools/serve_common.sh"):
            (root / name).write_text((source / name).read_text(encoding="utf-8"),
                                     encoding="utf-8", newline="\n")
        for name in ("qwenox-engine", "qwenox-api", "qwenox-engine-win.exe", "qwenox-win.exe"):
            (root / "build" / name).write_text("#!/usr/bin/env bash\nexit 0\n", newline="\n")
            (root / "build" / name).chmod(0o755)
        for name in ("qwen38-flash-next-w4b.hgn", "qwen38-flash-next-w4b.overlay.hgn",
                     "qwen38-flash-next-mtp.hgn", "qwen38-flash-next-vision.hgn",
                     "mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf", "mmproj-BF16.gguf"):
            (root / "models" / name).touch()
        for shard in range(1, 5):
            (root / "models" / f"Qwen3.8-Flash-Next-UD-Q4_K_XL-{shard:05d}-of-00004.gguf").touch()
        (root / "models" / "tokenizer" / "tokenizer.json").write_text("{}")
        for name, body in {"ss": "exit 0", "pgrep": "exit 1", "awk": "echo 2000",
                           "netstat": "exit 0", "flock": "exit 0", "systemctl": "exit 0",
                           "setsid": "exit 0"}.items():
            (root / "bin" / name).write_text("#!/usr/bin/env bash\n" + body + "\n", newline="\n")
            (root / "bin" / name).chmod(0o755)
        env = dict(os.environ)
        for line in original_conf.splitlines():
            if "=" in line and not line.startswith("#"):
                env.pop(line.split("=", 1)[0].strip(), None)
        env.pop("ENGINE_HOST", None)
        env["PATH"] = str(root / "bin") + os.pathsep + env["PATH"]
        env["ENGINE_PORT"] = str(engine_port)
        env["API_PORT"] = str(api_port)
        env["MSYS_NO_PATHCONV"] = "1"
        shell = ('root="$1"; if command -v cygpath >/dev/null; then '
                 'root="$(cygpath -u "$root")"; fi; '
                 'export PATH="$root/bin:$PATH"; exec bash "$root/$2" --check')
        commands = [(name, [args.bash, "-c", shell, "engine-host-test", root.as_posix(), name])
                    for name in ("start_hgn.sh", "start_gguf.sh", "start_win.sh")]
        if args.launcher:
            launcher = root / "start_win.exe"
            shutil.copy2(Path(args.launcher).resolve(), launcher)
            commands.append(("start_win.exe", [str(launcher), "--check"]))
        cases = [
            ("default", "127.0.0.1", None, "127.0.0.1", True),
            ("old-conf", None, None, "127.0.0.1", True),
            ("configured-loopback", "127.0.0.2", None, "127.0.0.2", True),
            ("explicit-wildcard", "0.0.0.0", None, "0.0.0.0", True),
            ("environment-overrides-conf", "127.0.0.2", "127.0.0.1", "127.0.0.1", True),
            ("reject-invalid-octet", "999.0.0.1", None, None, False),
            ("reject-leading-zero", "127.00.0.1", None, None, False),
            ("reject-hostname", "localhost", None, None, False),
        ]
        for case, configured, override, expected, valid in cases:
            conf = original_conf
            if configured is not None:
                conf += 'ENGINE_HOST="${ENGINE_HOST:-' + configured + '}"\n'
            (root / "service.conf").write_text(conf, encoding="utf-8", newline="\n")
            case_env = dict(env)
            if override is not None:
                case_env["ENGINE_HOST"] = override
            for name, command in commands:
                result = subprocess.run(command, cwd=root, env=case_env,
                                        capture_output=True, text=True, encoding="utf-8",
                                        errors="replace", timeout=15,
                                        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
                output = result.stdout + result.stderr
                if valid:
                    assert result.returncode == 0, f"{name} {case}: {output}"
                    assert f"engine {expected}:{engine_port}" in output, f"{name} {case}: {output}"
                    if name in ("start_hgn.sh", "start_gguf.sh"):
                        assert f"--host {expected}" in output, f"{name} {case}: {output}"
                else:
                    assert result.returncode != 0 and "ENGINE_HOST" in output, f"{name} {case}: {output}"
                print(f"PASS {name} {case}")
    print("RESULT PASS")


if __name__ == "__main__":
    main()
