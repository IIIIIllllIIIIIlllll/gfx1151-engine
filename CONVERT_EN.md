# Converting Your Own Model (flashnext2hgn)

*中文版:[CONVERT.md](CONVERT.md)*

`tools/flashnext2hgn.py` converts a HuggingFace safetensors-format
Qwen3.8-Flash-Next (qwen4_exp architecture) model or fine-tune into this
engine's `.hgn` weights. Single-file script, depends only on Python 3 +
numpy (no torch required).

```bash
python3 tools/flashnext2hgn.py /path/to/hf-model --out /path/to/outdir
```

Outputs (inside outdir):

- `<name>.hgn` — main weights (q4cp 4-bit linear layers + bf16
  norm/router/conv + fp8 PLE table + q4cp MTP fallback), about 116 GiB
- `<name>-mtp.hgn` — MTP 8-bit sidecar (q8g64); pass it to the engine as
  the second positional argument at startup to upgrade draft head precision
- `<name>-vision.hgn` — vision tower (all bf16), used with `--vision-tower`
- `tokenizer/` — tokenizer files copied from the source directory
- `start.sh` — one-shot launch script (engine + OpenAI API, with production
  environment variables and a memory-cap watchdog), generated automatically
  during conversion

Launch:

```bash
bash /path/to/outdir/start.sh
```

Once ready, the API is at `http://127.0.0.1:8731/v1`. Environment variables
you can override: `ENGINE_PORT` (default 8730), `API_PORT` (8731),
`MAXCTX` (262144), `GAMMA` (MTP draft length, 3), `MEMCAP_GB` (memory cap,
86). The script bakes in the engine directory pointed to by `--engine-bin`
at conversion time (default: `build` in this repo); if you move the engine,
regenerate it with `--start-script-only --engine-bin <new-build-dir>` —
no need to re-convert the model.

## Notes

- **Only the qwen4_exp shape is supported** (48 layers / hidden 2560 /
  512 experts / 24 heads / vocab 248320 / MTP 1 layer / vision 27 layers):
  the engine's dimensions are hard-coded. The tool validates `config.json`
  and rejects mismatched shapes outright. Any other Flash-Next fine-tune
  (weights changed, structure unchanged) can be converted.
- **Quantization needs no calibration data**: q4cp = per-tensor 16-level
  Lloyd codebook + 32-column group absmax (fp16 scale); MTP sidecar =
  q8g64 (64-column group affine); PLE table = fp8 e4m3 global scale. All of
  it is data-free RTN, quality on par with GGUF's Q4_K.
- Full conversion takes about 1-2 hours (177B parameters, pure CPU), peak
  memory ~12 GiB, and output requires about 120 GiB of disk.
- `--skip-vision` / `--skip-ple` / `--skip-mtp-sidecar` trim the
  corresponding artifacts; `--dry-run` only validates the mapping and
  prints the output plan; `--quant-selftest` runs the quantizer's numeric
  self-check.
- The base file contains q4cp-precision MTP weights as a fallback: without
  the sidecar, speculative decoding still runs; with the sidecar, draft
  quality is better (higher acceptance rate).
