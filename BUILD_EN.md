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
bash build.sh                 # all: build engine + API in parallel (default)
bash build.sh engine [name]   # engine only → build/<name> (default gdec)
bash build.sh api             # API server + CLI tools only
bash build.sh test            # build ktest and run kernel unit tests
```

Artifacts:

| File | Contents |
|---|---|
| `build/gdec` | GPU engine (`src/gpu/gdec.cpp`) |
| `build/gdec-api` | OpenAI-compatible API server (`src/api/*.cpp`) |
| `build/tok_cli` `tpl_cli` `eng_cli` | tokenizer / template / engine protocol CLIs |
| `build/http_selftest` `toolparse_test` `vision_test` | API component self-tests |
| `build/ktest` | engine kernel unit tests |

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
  `g++`). Example:

```bash
HIPCC=/opt/rocm/bin/hipcc GPU_ARCH=gfx1151 bash build.sh
```

## Compile options (for reference)

Engine:

```bash
hipcc -O3 -Werror --offload-arch=gfx1151 \
  -o build/gdec src/gpu/gdec.cpp -lrocblas -lhipblaslt
```

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
For daily serving use the root `start.sh` directly (see QUICKSTART_EN.md); the
following is the manual approach.

Production options (some optimizations are enabled via environment variables):

```bash
export GDEC_QSA_KV_BF16=1 GDEC_QSA_WMMA=1 GDEC_QSA_WMMA_BTV=1
export GDEC_MOE_LT=1 GDEC_MOE_LT_BF16=1 GDEC_GR_BF16=1
export GDEC_GDN_STREAM=1 GDEC_GDN_WAVE=1 GDEC_NOWARMUP=1
export GDEC_PREFILL_CHUNK=16384
export GDEC_INDEX_FUSED2=1 GDEC_PP_MOE_OUT=1 GDEC_INDEX_STREAM_SELECT=1
MODEL_BASE=./models/qwen38-flash-next-w4b
```

Short token-ID inference example (`--tokens` accepts token IDs; for text go
through the tokenizer/API):

```bash
bash tools/run_capped.sh 86 -- build/gdec \
  "$MODEL_BASE.hgn" "$MODEL_BASE.overlay.hgn" \
  --tokens 1,2,3 --gen 8 --maxctx 4096
```

Launch a 256K service:

```bash
bash tools/run_capped.sh 86 -- build/gdec \
  "$MODEL_BASE.hgn" "$MODEL_BASE.overlay.hgn" \
  --serve --port 8730 --maxctx 262144 --gamma 3 \
  --vision-tower ./models/qwen38-flash-next-vision.hgn
```

Wait until the engine prints `serve: listening`, then start the API in
another terminal:

```bash
build/gdec-api --tokenizer ./models/tokenizer \
  --engine 127.0.0.1:8730 --host 127.0.0.1 --port 8731 --context 262144
```

A text-only service can omit the engine's `--vision-tower`. `GDEC_NOWARMUP=1`
makes the first request bear the warmup cost, so first-request latency cannot
be taken directly as steady-state prefill performance.

## Windows (TheRock)

The Windows version has its own entry points, parallel to build.sh /
start.sh, covering the engine and the API frontend
(multimodal already supports PNG/JPEG, only WebP is not wired up; see
PORTING-WINDOWS_EN.md):

```bash
bash build_win.sh           # engine → build/gdec-win.exe
bash build_win.sh api       # OpenAI API frontend → build/gdec-api-win.exe
bash build_win.sh launcher  # script-free launcher → ./start_win.exe
bash build_win.sh test      # build ktest-win and run kernel unit tests
bash start_win.sh           # under Git Bash: start engine (line protocol 8730) + API (8731) as two processes
```

**Use `start_win.exe` for daily launches**: a native Win32 launcher,
double-click and go, no Git Bash / PowerShell / any script host needed. It
brings up the engine + API as two processes; child process output is shown
live on the console and written to `logs\`; Ctrl+C or closing the window
takes both down together.
**Configuration lives in the root `service.conf`** (the same file as Linux
`start.sh`): edit it to change the model filename, adjust the context window,
or change ports; precedence is environment variables > service.conf >
built-in defaults (`set MAX_CONTEXT=131072 && start_win.exe` overrides
temporarily; `start_win.exe --check` only checks the configuration without
starting). Clients connect to `http://127.0.0.1:8731/v1` (standard OpenAI
interface, including streaming); 8730 is the engine's internal line protocol,
automatically bridged by the API — no need to connect to it directly.

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
Linux `start.sh` (model paths, ports, context window, etc. are all edited
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
