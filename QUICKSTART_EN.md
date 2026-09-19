# Build and Launch
*中文版:[QUICKSTART.md](QUICKSTART.md)*

On a Linux / ROCm machine, enter the project directory and run:

```bash
bash build.sh
bash start.sh
```

`build.sh` compiles the GPU engine and the native API in one pass, producing `build/gdec` and `build/gdec-api`.
`start.sh` loads the engine first, then starts the API once it is ready. Defaults: 256K context, MTP gamma 3.
Press **Ctrl+C** to stop both the API and the engine started this time. Startup does not exit the terminal; keep the
SSH session open, or run it in tmux.

After startup, the API is available at `http://<host>:8731/v1` by default.
Logs for each launch are saved under `logs/datetime-PID/engine.log` and `api.log`.

## Model Files

Edit `service.conf` in the root directory; usually you only need to change `MODEL_DIR` (default `./models`):

```bash
MODEL_DIR="./models"
```

The directory should contain:

```text
models/
  qwen38-flash-next-w4b.hgn
  qwen38-flash-next-w4b.overlay.hgn
  qwen38-flash-next-mtp.hgn
  qwen38-flash-next-vision.hgn
  tokenizer/tokenizer.json
```

If your model names differ, edit `MODEL_FILE`, `OVERLAY_FILE`, `MTP_FILE`,
`VISION_FILE`, and `TOKENIZER_DIR` directly. For text-only use, set `VISION_FILE=""`; if you don't need an
overlay, set `OVERLAY_FILE=""`; `MTP_FILE` is a standalone 8-bit MTP speculative
draft weights file — setting `MTP_FILE=""` falls back to the 4-bit draft head built into the overlay.
Converting your own model with `tools/flashnext2hgn.py` generates this set of files automatically;
see CONVERT_EN.md.

All relative paths are based on the project directory containing the scripts, regardless of where the terminal was opened.
You can also override the configuration temporarily without editing the file:

```bash
MODEL_DIR=./models VISION_FILE="" bash start.sh
```

## Speculative Decoding

The default drafter is chain (ngram first, MTP as fallback); it is enabled by default on the engine side with no
arguments needed; both greedy and sampling requests go through chain. The HTTP API does not expose drafter selection — sending
a `drafter` field in the request body is silently ignored. Environment variable `GDEC_DRAFTER=ngram` selects pure ngram,
`=mtp` pure MTP, `=serial` the serial baseline.

`MTP_GAMMA=1 bash start.sh` tries a draft length of 1 per round; the range is 1–8, default 3;
after changing it you must restart the engine, no rebuild needed. See MTP_EN.md for the meaning of the parameters and acceptance rate.

## Other Common Commands

```bash
bash start.sh --check   # only check files, ports, memory, etc.; do not start the service
bash build.sh engine   # build the engine only
bash build.sh api      # build the API only
bash build.sh test     # build and run kernel unit tests without loading the model
```

Port, listen address, context, MTP, and memory cap are centralized in `service.conf`. By default the API
listens on `0.0.0.0:8731`, accessible from the LAN; if you only need local access, change it to
`API_HOST="127.0.0.1"`.

The engine uses `tools/run_capped.sh`, with a default memory cap of 86 GiB and at least
100 GiB of free memory required before startup; it refuses to start while another engine or a startup task from the same project is running.
If there is no progress in logs, engine I/O, or GPU GTT allocation for 60 seconds during loading, startup stops; the maximum
startup time is 360 seconds. Exit only cleans up the processes started this time.

## Build Environment

Verified environment: Ubuntu, `gfx1151` GPU, ROCm HIP 7.x / AMD clang.
GPU builds always use `-O3 -Werror`, link rocBLAS and hipBLASLt, and also require
the rocPRIM headers. The API uses C++17 / `-O2 -Werror`, depending on nlohmann-json,
libpng, libjpeg, libwebp, and pthread. Ubuntu install command:

```bash
sudo apt install build-essential nlohmann-json3-dev libpng-dev libjpeg-dev libwebp-dev
```

The scripts first look for `hipcc` in PATH, otherwise they use `/opt/rocm/bin/hipcc`.
Other ROCm locations or GPUs can be specified explicitly:

```bash
HIPCC=/opt/rocm/bin/hipcc GPU_ARCH=gfx1151 bash build.sh
```

Compilation runs under an 8 GiB memory limit and timeout protection (600 seconds for the engine,
120 seconds per API target). On slower machines, a timeout does not mean a source error; confirm the compiler is still making progress,
then adjust the `build.sh` timeouts to match your machine's capability. See BUILD_EN.md for details.
