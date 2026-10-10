# Building and Verification

*中文版:[BUILD.md](BUILD.md)*

## Environment

Build on a Linux / AMD ROCm machine. GPU target `gfx1151` (AMD Strix Halo),
compiler is ROCm's bundled `hipcc` (HIP 7.x / AMD clang).

The engine requires HIP, rocBLAS, hipBLASLt, rocPRIM development files and the
Linux C++ standard library. The API additionally requires g++, libpng, libjpeg,
and libwebp development packages. nlohmann/json is vendored at
`third_party/nlohmann/json.hpp`, so no system nlohmann-json package is required.
Ubuntu install command:

```bash
sudo apt install build-essential libpng-dev libjpeg-dev libwebp-dev
```

ROCm must be a version that supports `gfx1151`. Do not apply the GPU
architecture flags directly to other graphics cards.

## build.sh (unified entry point)

```bash
bash build.sh                 # all: build engine + benchmark + API in parallel (default)
bash build.sh --bundle        # distribution build with all runtime dependencies
bash build.sh engine [name]   # engine only → build/<name> (default qwenox)
bash build.sh bench           # standalone performance benchmark only
bash build.sh api             # API server + CLI tools only
bash build.sh test            # build ktest and run kernel unit tests
```

Artifacts:

| File | Contents |
|---|---|
| `build/qwenox-engine` | GPU engine (`src/gpu/qwenox.cpp`) |
| `build/qwenox-bench` | Standalone performance benchmark (`src/gpu/bench_main.cpp`) |
| `build/qwenox-api` | OpenAI-compatible API server (`src/api/*.cpp`) |
| `build/tok_cli` `tpl_cli` `eng_cli` | tokenizer / template / engine protocol CLIs |
| `build/http_selftest` `toolparse_test` `vision_test` `engine_host_test` | API / engine listen-address component self-tests |
| `build/ktest` | engine kernel unit tests |
| `build/lib` | ROCm/image runtimes and gfx1151 kernel databases from `--bundle` |

Behavior notes:

- Compilation runs under a memory limit (8 GiB) and timeout protection: 600
  seconds for engine/ktest, 120 seconds for each API target. On slower
  machines a timeout does not imply a source error — after confirming the
  compiler is still making progress, adjust the timeout seconds in the
  `compile` calls in `build.sh` to match your machine's capability.
- Each target is first compiled to a temporary file and atomically replaced
  on success; on compilation failure the last successful binary is kept and
  a non-zero status is returned.
- Compilation is refused while the engine or API is running (or another
  build/launch task holds the lock); stop the service before building.
- Environment variables: `HIPCC` (defaults to hipcc on PATH, then
  `/opt/rocm/bin/hipcc`), `GPU_ARCH` (default `gfx1151`), `CXX` (default
  `g++`), and `ROCM_PATH` (only needed when the ROCm root cannot be inferred
  from `HIPCC`). Example:

```bash
HIPCC=/opt/rocm/bin/hipcc GPU_ARCH=gfx1151 bash build.sh
```

- The default build is for local use and loads the already installed system
  runtimes; it does not create `build/lib/`. For distribution, pass
  `--bundle`; the script follows the ELF dependencies and copies the ROCm
  user-space runtime plus the libpng/libjpeg/libwebp dependency closure
  into `build/lib/`. It also copies only the rocBLAS/hipBLASLt kernel database
  for `GPU_ARCH`. The binaries contain an `$ORIGIN/lib` RPATH and `start_hgn.sh` /
  `start_gguf.sh` explicitly select the bundled files, so deployments should copy the whole
  `build/` directory and do not need these runtimes installed separately.

## Compile options (for reference)

Engine:

```bash
hipcc -O3 -Werror --offload-arch=gfx1151 \
  -Wl,-rpath,'$ORIGIN/lib' -Wl,--disable-new-dtags \
  -o build/qwenox-engine src/gpu/qwenox.cpp -lrocblas -lhipblaslt
```

The RPATH flags above are added only to `--bundle` distribution builds.

Do not add `-ffast-math` yourself; it changes numerical behavior.

API: C++17, `-O2 -Wall -Wextra -Wpedantic -Werror`, linked with
`-lpng -ljpeg -lwebp -lpthread`. `tools/build_api.sh` is just a compatibility
wrapper pointing to `build.sh api`.

## Kernel unit tests (no model loaded)

`bash build.sh test` compiles and runs `tools/ktest.cu`; the expected output
ends with `ALL PASS`. These are kernel unit tests and do not replace
full-model numerical and performance regression (for the full regression see
the "Reproduction" section of NGRAM_EN.md).

## Manual inference launch

Model weights, overlay, tokenizer, and the optional vision tower are not in
the source repository; convert them from the HF model with
`tools/flashnext2hgn.py` (see CONVERT_EN.md).
Before actual inference check `free -g`: running only one engine, available
memory should be at least 100 GiB.
For daily serving use the root `start_hgn.sh` (hgn weights) or `start_gguf.sh`
(GGUF weights, see GGUF.md) directly (see QUICKSTART_EN.md); the following is the
manual approach for hgn.

Production options (some optimizations are enabled via environment variables):

```bash
export QWENOX_QSA_KV_BF16=1 QWENOX_QSA_WMMA=1 QWENOX_QSA_WMMA_BTV=1
export QWENOX_MOE_LT=1 QWENOX_MOE_LT_BF16=1 QWENOX_GR_BF16=1
export QWENOX_GDN_STREAM=1 QWENOX_GDN_WAVE=1 QWENOX_NOWARMUP=1
export QWENOX_PREFILL_CHUNK=16384
export QWENOX_INDEX_FUSED2=1 QWENOX_PP_MOE_OUT=1 QWENOX_INDEX_STREAM_SELECT=1
MODEL_BASE=./models/qwen38-flash-next-w4b
```

Short token-ID inference example (`--tokens` accepts token IDs; for text go
through the tokenizer/API):

```bash
bash tools/run_capped.sh 86 -- build/qwenox-engine \
  "$MODEL_BASE.hgn" "$MODEL_BASE.overlay.hgn" \
  --tokens 1,2,3 --gen 8 --maxctx 4096
```

Launch a 256K service:

```bash
bash tools/run_capped.sh 86 -- build/qwenox-engine \
  "$MODEL_BASE.hgn" "$MODEL_BASE.overlay.hgn" \
  --serve --port 8730 --maxctx 262144 --gamma 3 \
  --vision-tower ./models/qwen38-flash-next-vision.hgn
```

Wait until the engine prints `serve: listening`, then start the API in
another terminal:

```bash
build/qwenox-api --tokenizer ./models/tokenizer \
  --engine 127.0.0.1:8730 --host 127.0.0.1 --port 8731 --context 262144
```

A text-only service can omit the engine's `--vision-tower`. `QWENOX_NOWARMUP=1`
makes the first request bear the warmup cost, so first-request latency cannot
be taken directly as steady-state prefill performance.

The API disconnect regression needs no GPU or model weights: it creates a
synthetic tokenizer in a temporary directory and starts a local fake engine
and API. It covers streaming/non-streaming disconnects on all three generation
endpoints, prefill, queuing, protocol draining, and the next request. Build the
API first; on Windows, use `--api build/qwenox-win.exe` instead:

```bash
python tools/api_disconnect_test.py --api build/qwenox-api
python tools/api_disconnect_test.py --api build/qwenox-api --slots 2
```

Engine listen-address and launcher configuration regressions (no model needed):

```bash
build/engine_host_test
python tools/engine_host_config_test.py --bash bash
```

On Windows, `--launcher` used to also test the old native launcher; it is now
archived in `attic/launcher/` (no longer built) — compile it manually and pass
the path if needed.

## Windows (TheRock)

The Windows version has its own entry points, parallel to build.sh /
start_hgn.sh, covering the engine and the API frontend
(multimodal already supports PNG/JPEG, only WebP is not wired up; see
PORTING-WINDOWS_EN.md):

```bash
bash build_win.sh           # all artifacts: engine, benchmark, API, root stub
bash build_win.sh bench     # standalone performance benchmark only
bash build_win.sh api       # OpenAI API frontend → build/qwenox-win.exe (GUI tray app)
bash build_win.sh launcher  # minimal root stub → ./start_win.exe
bash build_win.sh test      # build ktest-win and run kernel unit tests
bash start_win.sh           # under Git Bash: start engine (line protocol 8730) + API (8731, --console) as two processes
```

**Double-click `start_win.exe` for daily launches**: a minimal root stub whose
only job is launching `build\qwenox-win.exe` with the repo root as working
directory. **On Windows the API component itself is the tray app** (GUI
subsystem, no console window on double-click), taking over the niche of the
old `start_win.exe` launcher: right-click the tray icon to open the web
console / start·stop the engine / copy the API URL / view the engine or API
log / open the log folder / quit (the engine lives in a KILL_ON_JOB_CLOSE
job, so it dies with the API — no orphans); double-click opens the web
console. Engine state is polled every 2 s, with balloons on ready or
unexpected exit; while running, the tooltip's second line shows live pp/tg
rates (SNAP polling). The API's own output goes to `logs\api-win-*.log`,
engine output to `logs\engine-api-*.log`. **The engine does not start
automatically**: edit settings on the console's "Engine" page (`#/engine`)
and start it there, or use the tray right-click menu. The old launcher with
the native setup panel is archived in `attic/launcher/` and no longer built
or maintained.
**Configuration lives in the root `service.conf`** (the same file as the Linux
launchers; Windows currently supports hgn only and reads the "hgn" section, the
GGUF section has no effect): edit it directly, or on the console's `#/engine`
page (backed up to `service.conf.bak` before writing, applied on the next
engine start); precedence is environment variables > service.conf >
built-in defaults. Clients connect to `http://127.0.0.1:8731/v1` (standard OpenAI
interface, including streaming); 8730 is the engine's internal line protocol,
automatically bridged by the API — no need to connect to it directly.

**Standalone API details**: running `build\qwenox-win.exe` with no arguments
(double-clicking works too) enters tray mode; `--console` restores console
mode (when output is redirected by a script, the console is not claimed). It
reads the root `service.conf` itself for the tokenizer, ports and context
(explicit command-line flags still win; precedence is argv > environment >
service.conf > built-in defaults; `--service-conf FILE` or `QWENOX_SERVICE_CONF`
selects another config file). Without an engine it still serves the console
web UI, `/health` and friends; only inference requests return 502. A missing
tokenizer falls back to the repo-bundled `data/tokenizer` (see its README;
an explicit `--tokenizer` failure does not fall back); if that is also
missing it is only a warning (inference returns 503) — console, config and
engine management keep working. On a fatal startup error (port in use, bad
arguments), tray mode shows a MessageBox and writes the log, while console
mode pauses the window so you can read the message. Engine start/stop is done
by the API process itself spawning `build\qwenox-engine-win.exe` (same QWENOX_*
environment and arguments; stopping terminates the process). On
Linux the start/stop endpoints return 501 and the page only edits config.

Build entry points run in Git Bash (double-clicking `build_win.bat` also
works — it locates Git Bash automatically; it only accepts Git for Windows'
bash and deliberately avoids WSL's `System32\bash.exe`, because the build
scripts depend on Git Bash path semantics). **Git is only a build-time
dependency; it is not needed at runtime.**

Prerequisites (**build-time only**): the
[TheRock](https://github.com/ROCm/TheRock) Windows multi-arch package
(default `C:\therock-dist-windows-multiarch-10.0.0\...`, overridable with
`THEROCK=`) + Git Bash. The first build stages all runtime dependencies into
`build/`: 6 TheRock DLLs, 3 MSVC runtimes, and the actual gfx1151 kernel db
of rocBLAS/hipBLASLt (~30M total, not the all-architecture 1.2G). **The
artifacts are self-contained: copy `build/` to any Windows machine of the
same architecture and it runs — no need to install ROCm/TheRock, and no
HIP_PATH/ROCM_PATH is set** (see PORTING-WINDOWS_EN.md for de-rooted
real-world testing). The GPU needs enough VRAM partitioned in BIOS (heretic
68 GiB weights + 256K context measured at a full 95 GiB, so partition 96
GiB). `start_win.exe` / `start_win.sh` share the root `service.conf` with
the Linux launchers (hgn section only) (model paths, ports, context window, etc. are all edited
there; environment variables can override temporarily).

## Optional tools and common build problems

CPU reference implementation and weight checker (not part of build.sh;
compile manually as needed):

```bash
g++ -O3 -Werror -std=c++17 -o build/ref src/ref.cpp
g++ -O3 -Werror -std=c++17 -o build/hgn_dump src/hgn_dump.cpp
```

Can't find `hipcc`: use `/opt/rocm/bin/hipcc` and check the ROCm
installation.
Can't find `rocprim/...`, `hipblaslt/...` or `-lhipblaslt`: check whether the
development headers and libraries of the same ROCm installation are
installed; avoid mixing versions.
Can't find `nlohmann/json.hpp`:
make sure the repository's `third_party/nlohmann/json.hpp` is present.
Can't find `png.h`, `jpeglib.h`, or `webp/decode.h`:
install the image development packages listed above.
Getting `no kernel image` / architecture errors: verify the graphics card
against `--offload-arch`; do not just remove the flag to mask the problem.
`-Werror` failures: keep the diagnostics and fix the corresponding
compatibility issues; disabling it outright is not recommended.
