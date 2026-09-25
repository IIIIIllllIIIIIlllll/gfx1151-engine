# gfx1151-engine

*中文版:[README.md](README.md)*

A local inference engine that runs a 177B MoE model on a single AMD Strix
Halo APU (gfx1151). Target model: Qwen3.8-Flash-Next (qwen4_exp architecture)
and fine-tunes with the same architecture.

The 68 GiB of quantized weights are stored in host memory in 4-bit
quantization and read directly by the GPU kernels — no large VRAM needed;
a machine with 122 GiB of RAM can serve a 256K context.

## Measured Performance

Development machine: **GMK EVO-X2 (AMD Ryzen AI Max+ 395, Strix Halo /
gfx1151)**, 122 GiB RAM:

| Metric | Value |
| --- | --- |
| Prefill (128K context) | ~1100–1200 tok/s |
| Decode | ~30–55 tok/s (depends on speculative hit rate) |
| Average power draw | ~120 W |
| Peak (instantaneous) power draw | ~130 W (bursts for a few seconds, then settles back to ~120 W) |

## Features

- **Speculative decoding chain**: ngram drafts first, MTP as fallback, with
  round-by-round fallback; greedy does bit-exact comparison, sampling
  resamples and compares against the target distribution, so the output
  distribution is identical to serial decoding. Enabled by default, no
  parameters needed. Highly repetitive content (code comments, template
  text) shows significant measured speedups.
- **Standalone 8-bit MTP draft weights** sidecar, with a higher acceptance
  rate than the built-in 4-bit draft head.
- **Vision**: supports image input (OpenAI `image_url`), with KV reuse
  across turns.
- **OpenAI-compatible API**: streaming, tool calling,
  `/v1/chat/completions`.
- **256K context**, with two-tier prompt caching: in-RAM checkpoints at
  message boundaries (edit-and-resend replies instantly) plus KV snapshots
  recovered across restarts.
- **Model conversion tool**: HF safetensors → `.hgn`; quantization needs no
  calibration data, and can be distributed for your own fine-tuned models
  (see CONVERT_EN.md).

## Requirements

- Linux + ROCm (HIP 7.x), GPU architecture `gfx1151`; or Windows + AMD GPU
  driver (the GPU needs VRAM carved out in BIOS), see the "Windows" section
- Available memory ≥ 100 GiB (68 GiB weights pinned (page-locked) + KV)
- Build dependencies: rocBLAS, hipBLASLt, rocPRIM; the API frontend also
  needs libpng, libjpeg, libwebp; nlohmann/json is vendored in the repository

## Quick Start

```bash
bash build.sh        # Build engine + API, output goes to build/
bash start_hgn.sh    # hgn weights: load the model and start the service (reads service.conf)
bash start_gguf.sh   # or GGUF weights (Unsloth UD-Q4_K_XL, the same files llama.cpp uses)
```

Both launchers read weights from `./models` and list any missing files before
exiting; configuration is centralized in `service.conf` (one section each for
hgn and GGUF; GGUF details in [GGUF.md](GGUF.md)). See [QUICKSTART_EN.md](QUICKSTART_EN.md)
for details.

Converting your own fine-tuned model (HF safetensors, same architecture):

```bash
python3 tools/flashnext2hgn.py /path/to/hf-model --out ./models
```

See [CONVERT_EN.md](CONVERT_EN.md) for details.

## Windows

The Windows version has feature parity with the Linux version (engine +
OpenAI API + multimodal). Porting notes and measurements are in
[PORTING-WINDOWS_EN.md](PORTING-WINDOWS_EN.md). Builds run in Git Bash
(or double-click `build_win.bat`; Git is only needed at build time):

```bash
bash build_win.sh           # Engine
bash build_win.sh api       # OpenAI API frontend
bash build_win.sh launcher  # Script-free launcher start_win.exe
```

For daily use, double-click `start_win.exe` (native Win32, no
Git/PowerShell needed): it brings up the engine + API dual processes,
shows output in real time and writes it to `logs\`; Ctrl+C or closing the
window stops it. Configuration is **shared with Linux via
`service.conf`** (edit it to change the model file name or context
window); environment variables can temporarily override it. Clients
connect to `http://<host>:8731/v1`.

Distribution: copy `build/` + `start_win.exe` + `models/` to any gfx1151
Windows machine and it just works — **no ROCm/TheRock installation
needed**; only the AMD GPU driver, plus enough VRAM carved out for the GPU
in BIOS (a 256K context needs 96 GiB).

Differences from the Linux version:

- Image decoding supports PNG/JPEG via stb_image (WebP not wired up)
- Prefill chunk defaults to 8192
- Cold loading reads the full weights from disk (minute-scale, progress
  shown in console/logs)
- The launchers do not enable `GDEC_GEMM_WMMA` or `GDEC_GDN_FUSED` (the
  self-written WMMA GEMM and fused GDN kernel already promoted on Linux
  launchers, worth ~8-10% PP combined but unverified under TheRock — so
  Windows prefill uses hipBLASLt plus the legacy GDN path)

Build details are in [BUILD_EN.md](BUILD_EN.md).

## Documentation

- [QUICKSTART_EN.md](QUICKSTART_EN.md) — build, launch, configuration
- [BUILD_EN.md](BUILD_EN.md) — build environment details and
  troubleshooting
- [CONVERT_EN.md](CONVERT_EN.md) — model conversion tool
- [MTP_EN.md](MTP_EN.md) — speculative decoding parameters and comparison
  methods
- [NGRAM_EN.md](NGRAM_EN.md) — ngram verification design, benefits, and
  known divergences
- [HGN-FORMAT_EN.md](HGN-FORMAT_EN.md) — the `.hgn` weight container
  format
- [data/README_EN.md](data/README_EN.md) — numerical regression benchmark
  (data/qsa-oracle) description
- [PORTING-WINDOWS_EN.md](PORTING-WINDOWS_EN.md) — Windows porting notes
  and measurements

## Tests

```bash
bash build.sh test   # Kernel unit tests, no model loading, expect ALL PASS
```

## Acknowledgements

This project's implementation borrows from [halogen-flash-server](https://github.com/peonist-ai/halogen-flash-server) by peonist-ai. The `.hgn` weight container format is halogen's checkpoint container format — see [HGN-FORMAT_EN.md](HGN-FORMAT_EN.md). Many thanks to the halogen authors.

The ngram speculative drafting approach also borrows ideas from the open-source [llama.cpp](https://github.com/ggml-org/llama.cpp) (MIT license); see the "Attribution" section of [NGRAM_EN.md](NGRAM_EN.md).

## License

[AGPL-3.0](LICENSE)
