# Build and Launch
*中文版:[QUICKSTART.md](QUICKSTART.md)*

On a Linux / ROCm machine, enter the project directory, build, then start with the launcher that matches
the weight format you have:

```bash
bash build.sh
bash start_hgn.sh    # hgn weights (converted with tools/flashnext2hgn.py)
bash start_gguf.sh   # or GGUF weights (Unsloth UD-Q4_K_XL, the same files llama.cpp uses)
```

`build.sh` compiles the GPU engine and the native API in one pass, producing `build/gdec` and `build/gdec-api`.
Apart from which weights they read, the two launchers are identical: they load the engine first, then start the
API once it is ready. Defaults: 256K context, MTP gamma 3. Press **Ctrl+C** to stop both the API and the engine
started this time. Startup does not exit the terminal; keep the SSH session open, or run it in tmux. The old
`start.sh` only prints the choice above and exits.

After startup, the API is available at `http://<host>:8731/v1` by default.
Logs for each launch are saved under `logs/datetime-PID/engine.log` and `api.log`.

## Model Files

Both launchers read weights from `models/` in the project root (`MODEL_DIR` in `service.conf`, default
`./models`). `tokenizer/` is shared by both formats (only `tokenizer.json` is needed, taken from the original
model's HF repo). Only the format you use needs to be present:

```text
models/
  tokenizer/tokenizer.json                              # needed by both
  # --- bash start_hgn.sh ---
  qwen38-flash-next-w4b.hgn                             # main model
  qwen38-flash-next-w4b.overlay.hgn                     # overlay
  qwen38-flash-next-mtp.hgn                             # 8-bit MTP draft
  qwen38-flash-next-vision.hgn                          # vision tower
  # --- bash start_gguf.sh ---
  Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf     # main model, all 4 shards
  Qwen3.8-Flash-Next-UD-Q4_K_XL-00002-of-00004.gguf
  Qwen3.8-Flash-Next-UD-Q4_K_XL-00003-of-00004.gguf
  Qwen3.8-Flash-Next-UD-Q4_K_XL-00004-of-00004.gguf
  mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf               # MTP draft sidecar
  mmproj-BF16.gguf                                      # vision tower
```

If files are missing the launcher does not start; it lists every missing file with its config key and exits
with code 1, e.g.:

```text
错误：start_gguf.sh（GGUF 权重）缺少以下文件：
  分片 3/4：./models/Qwen3.8-Flash-Next-UD-Q4_K_XL-00003-of-00004.gguf
```

`start_hgn.sh` only accepts `.hgn` and `start_gguf.sh` only accepts `.gguf`; a file of the wrong format points
you to the other launcher. If your filenames differ, edit the matching section of `service.conf`:

- hgn: `MODEL_FILE`, `OVERLAY_FILE`, `MTP_FILE`, `VISION_FILE`. Set `OVERLAY_FILE=""` if you don't need an
  overlay; `MTP_FILE` is a standalone 8-bit MTP speculative draft weights file — setting `MTP_FILE=""` falls
  back to the 4-bit draft head built into the overlay. Converting your own model with `tools/flashnext2hgn.py`
  generates this set of files automatically; see CONVERT_EN.md.
- GGUF: `GGUF_FILE` (the first shard; the other shards must be in the same directory), `GGUF_MTP_FILE`,
  `GGUF_VISION_FILE`. `GGUF_MTP_FILE=""` disables MTP speculation (the ngram drafter still works). See GGUF.md
  for accuracy and performance.

For text-only serving either format can drop the vision tower (`VISION_FILE=""` / `GGUF_VISION_FILE=""`).
`TOKENIZER_DIR` is shared.

All relative paths are based on the project directory containing the scripts, regardless of where the terminal
was opened. You can also override the configuration temporarily without editing the file (environment variables
of the same name win):

```bash
MODEL_DIR=/data/models VISION_FILE="" bash start_hgn.sh
GGUF_VISION_FILE="" bash start_gguf.sh
```

## Speculative Decoding

The default drafter is chain (ngram first, MTP as fallback); it is enabled by default on the engine side with no
arguments needed; both greedy and sampling requests go through chain. The HTTP API does not expose drafter selection — sending
a `drafter` field in the request body is silently ignored. Environment variable `GDEC_DRAFTER=ngram` selects pure ngram,
`=mtp` pure MTP, `=serial` the serial baseline.

`MTP_GAMMA=1 bash start_hgn.sh` (or `start_gguf.sh`) tries a draft length of 1 per round; the range is 1–8, default 3;
after changing it you must restart the engine, no rebuild needed. See MTP_EN.md for the meaning of the parameters and acceptance rate.

## Other Common Commands

```bash
bash start_hgn.sh --check    # only check files, ports, memory, etc.; do not start (same for start_gguf.sh)
bash build.sh engine         # build the engine only
bash build.sh api            # build the API only
bash build.sh test           # build and run kernel unit tests without loading the model
```

Port, listen address, context, MTP, and memory cap are centralized in `service.conf`. By default the API
listens on `0.0.0.0:8731`, accessible from the LAN; if you only need local access, change it to
`API_HOST="127.0.0.1"`.

Query live memory accounting with `curl http://127.0.0.1:8731/memory`. The
response separates HIP device allocations, expert weights backed by
`mmap + hipHostRegister`, pinned host memory, and process RSS.
`gpu_accessible_committed_bytes` is the engine-accounted total; unregistered,
reclaimable file-mmap page cache is intentionally excluded.

The engine uses `tools/run_capped.sh`, with a default memory cap of 86 GiB and at least
100 GiB of free memory required before startup; it refuses to start while another engine or a startup task from the same project is running.
If there is no progress in logs, engine I/O, or GPU GTT allocation for 60 seconds during loading, startup stops; the maximum
startup time is 360 seconds. Exit only cleans up the processes started this time.

## Build Environment

Verified environment: Ubuntu, `gfx1151` GPU, ROCm HIP 7.x / AMD clang.
GPU builds always use `-O3 -Werror`, link rocBLAS and hipBLASLt, and also require
the rocPRIM headers. The API uses C++17 / `-O2 -Werror`; nlohmann/json is vendored,
and it depends on libpng, libjpeg, libwebp, and pthread. Ubuntu install command:

```bash
sudo apt install build-essential libpng-dev libjpeg-dev libwebp-dev
```

The scripts first look for `hipcc` in PATH, otherwise they use `/opt/rocm/bin/hipcc`.
Other ROCm locations or GPUs can be specified explicitly:

```bash
HIPCC=/opt/rocm/bin/hipcc GPU_ARCH=gfx1151 bash build.sh
```

Compilation runs under an 8 GiB memory limit and timeout protection (600 seconds for the engine,
120 seconds per API target). On slower machines, a timeout does not mean a source error; confirm the compiler is still making progress,
then adjust the `build.sh` timeouts to match your machine's capability. See BUILD_EN.md for details.
