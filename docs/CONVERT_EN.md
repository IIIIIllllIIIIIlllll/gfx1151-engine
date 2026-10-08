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
- `<name>.overlay.hgn` — quality overlay (optional; passed to the engine as
  a positional argument after the main weights and before the MTP sidecar,
  overriding same-named tensors): a re-quantization of the 723 dense linear
  layers (attention projections / shared experts / hyper-connections /
  lm_head — excluding embed_tokens, MoE experts and mtp.*). The 12 full-
  attention o_proj are upgraded to q8g64 8-bit (halving their error); the
  other 711 stay q4cp but with per-group scales optimized by an MSE grid
  search (better than the base file's absmax RTN). About 2.3 GiB
- `<name>.overlay-speed.hgn` — only with `--overlay-speed`: the same 723
  tensors, all q4cp (no q8g64 o_proj; faster to load/decode)
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
- **Quantization needs no calibration data**: everything is data-free RTN;
  see the "Quantization scheme" section below for details (the four
  quantization formats, the file partition rules, and error levels).
- Quantization is parallelized per tensor (`--jobs`, defaults to all cores;
  huge tensors are automatically throttled to bound memory). Measured on a
  12-core / 32 GiB machine: main weights + MTP + vision ≈ 3.7 hours,
  overlay ≈ 1 hour; peak memory ~20 GiB, output needs about 121 GiB of
  disk.
- `--skip-vision` / `--skip-ple` / `--skip-mtp-sidecar` / `--skip-overlay`
  trim the corresponding artifacts; `--overlay-speed` additionally writes
  the all-q4cp speed overlay; `--only-overlay` (re)builds just the overlay
  (skips re-running the 116 GiB base when you already have it);
  `--jobs N` sets the number of quantization worker processes (default:
  all cores; per-tensor rng seeds make the output bit-identical for any
  N); `--dry-run` only validates the mapping and prints the output plan;
  `--quant-selftest` runs the quantizer's numeric self-check.
- The base file contains q4cp-precision MTP weights as a fallback: without
  the sidecar, speculative decoding still runs; with the sidecar, draft
  quality is better (higher acceptance rate).

## Quantization scheme

All quantization is **calibration-free RTN** (round-to-nearest): no model
forward passes are needed, conversion runs on pure CPU, and results are
reproducible. Overall precision is on par with GGUF's Q4_K.

### How tensors are partitioned into files

- **Main weights `<name>.hgn`**: the full 48-layer backbone. Sensitive
  small operators — norms, router gates, convolutions, PLE projections —
  are kept as passthrough bf16 (the `BF16_KEEP` list); PLE configuration is
  i64; the big PLE ngram table is fp8; every other linear layer (including
  the 3D fused experts, flattened as `[E*rows, cols]`) is q4cp; a
  q4cp-precision mtp.* fallback is also included.
- **Overlay `<name>.overlay.hgn`**: a re-quantization of the 723 dense
  linear layers (attention projections, shared experts, hyper-connections,
  lm_head — **excluding** embed_tokens, MoE experts and mtp.*), which the
  engine applies on top of the main weights by name. The 12 full-attention
  o_proj use q8g64, the other 711 use q4cp_opt.
- **MTP sidecar `<name>-mtp.hgn`**: the 18 mtp.* tensors, all q8g64.
- **Vision tower `<name>-vision.hgn`**: all passthrough bf16.

### The four quantization formats

| Format | Algorithm | Used for |
|---|---|---|
| q4cp | per-tensor 16-level Lloyd codebook (fitted in 32-column group-absmax-normalized space, endpoints snapped to ±1) + one fp16 scale per group (= group absmax); `w = cb[nib] * scale` | main-weight linears |
| q4cp_opt | same codebook, but each group's scale is searched over a 21-step grid of 0.75–1.25 × absmax for the best weight-space MSE (clipping a few extreme values in exchange for a lower overall error) | overlay dense layers |
| q8g64 | per-64-column affine: `w = code * scale + min` (fp16 scale/min, uint8 codes) | MTP sidecar, overlay o_proj |
| fp8 | e4m3 + one trailing fp32 global scale (= absmax/448) | PLE ngram table |

### Error levels and trade-offs

- RTN is lossy by nature: q4cp's relative RMSE is about 8–10%, but LLM
  weights tolerate this kind of noise well and the impact on downstream
  quality is usually acceptable.
- q4cp_opt's scale search trades max error for mean error (~5% lower RMSE
  measured on real tensors; and since the grid includes the 1.0 step, its
  weight-space MSE is never worse than absmax RTN).
- Upgrading o_proj from q4cp to q8g64 roughly halves its error
  (0.099 → 0.048) — the overlay's main gain.
- Refitting the codebook after the scale search was tried and abandoned:
  plain Lloyd minimizes normalized-space error and ignores the s² weighting
  of the weight-space objective, so the alternation diverged. The kept
  scheme — fixed codebook + scale search — has a strict lower-bound
  guarantee.
- **The official halogen overlay's q4cp part uses calibration-driven
  activation-aware quantization** (GPTQ-style): it optimizes output error,
  not weight error (some tensors are 40% *worse* in weight-space RMSE), so
  it cannot be reproduced bit-for-bit without calibration data. If you need
  higher quality, you would have to bring your own calibration set and do
  activation-aware quantization — a much larger engineering effort.

