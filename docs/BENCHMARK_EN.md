# Performance Benchmark

`qwenox-bench` is the standalone command-line benchmark for this engine. It
does not start the HTTP service and it does not accept model paths on the
command line. This keeps a result tied to the model configuration in
`service.conf` instead of to a temporary shell command.

## Build

Linux (the default build also produces the engine and API artifacts):

```bash
bash build.sh
```

The executable is `build/qwenox-bench`.

Windows (the default build also produces the engine, API, and launcher artifacts):

```bash
bash build_win.sh
```

The executable is `build/qwenox-bench.exe`.

The `bench` target remains available when only the benchmark needs to be
rebuilt.

Opening the executable without arguments exits immediately. Run the benchmark
explicitly with `run`:

```bash
build/qwenox-bench run       # Linux
build/qwenox-bench.exe run   # Windows
```

Other arguments are rejected; edit `service.conf` or use the documented
environment overrides instead.

## What it measures

1. **Model loading** — selects one format only. By default it uses the
   configured HGN model when `MODEL_FILE` exists, otherwise the configured
   GGUF shard. Set `QWENOX_BENCH_FORMAT=hgn` or `QWENOX_BENCH_FORMAT=gguf` to make
   the choice explicit. HGN overlays and MTP weights, or the GGUF MTP sidecar,
   are loaded exactly from `service.conf`. The test stops on file, mapping,
   HIP, arena, graph, or allocation failures.
2. **Prefill** — benchmarks every `data/qsa-oracle/*.tokens` file. Each file
   is measured from a reset state after a non-measured engine warm-up. The
   report includes one row per prompt and a weighted average over all prompts.
3. **TG / MTP decode** — loads the tokenizer configured by `TOKENIZER_DIR` and
   runs three fixed English workloads: Python code generation, creative
   writing, and common-sense question answering. Each row reports decode
   tokens/second and the weighted MTP draft acceptance rate. The benchmark
   uses greedy MTP with the configured `MTP_GAMMA`; `MTP_GAMMA=0` uses the
   stable benchmark default of gamma 4.

All benchmark-generated messages are in English. The existing engine may
still print its own diagnostic lines while kernels are being initialized.

## Optional environment overrides

These do not change which model files are selected:

| Variable | Default | Meaning |
| --- | ---: | --- |
| `QWENOX_BENCH_FORMAT` | auto | `hgn` or `gguf` |
| `BENCH_PREFILL_REPEATS` | `2` | Samples per prefill prompt |
| `BENCH_DECODE_REPEATS` | `2` | Samples per decode workload |
| `BENCH_DECODE_TOKENS` | `128` | Generated tokens per decode sample |
| `PREFILL_CHUNK` | `0` | Prefill chunk size; `0` uses the platform default |

The benchmark reads paths relative to the project root containing
`service.conf`, so it can be launched from the project root or from the
`build` directory.
