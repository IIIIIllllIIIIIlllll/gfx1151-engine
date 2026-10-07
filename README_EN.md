# gfx1151-engine

![Strix Halo — Qwen3.8-Flash-Next](media/strix_banner_21x9_v2.png)

*中文版:[README.md](README.md)*

A local inference engine that runs a 177B MoE model on a single AMD Strix
Halo APU (gfx1151). Target model: Qwen3.8-Flash-Next (qwen4_exp architecture)
and fine-tunes with the same architecture.

The routed experts are stored in host memory in 4-bit quantization and read
directly by the GPU kernels — no large VRAM needed. The memory-resident
weights are ~67–77 GiB depending on the weight format (the PLE n-gram table
stays on disk and is read on demand); a machine with 122 GiB of RAM can serve
a 256K context.

## Measured Performance

Development machine: **GMK EVO-X2 (AMD Ryzen AI Max+ 395, Strix Halo /
gfx1151)**, 122 GiB RAM:

| Metric | Value |
| --- | --- |
| Prefill (8K, chunk 16384) | ~1730–1750 tok/s (measured 10-07) |
| Prefill (128K context) | ~1650 tok/s (measured 10-07) |
| Prefill (256K context) | ~1590 native, ~1580 with YaRN factor 2 (~0.5% overhead, measured 10-07) |
| Decode (speculative, greedy, γ=4) | ~51 tok/s at 8K, ~50 at 64K (real text, 10-07) |
| Decode (no speculation) | ~34 tok/s at 8K (measured 10-07, hgn standard weights; HQ/GGUF read more per token, ~16% slower) |
| Average power draw | ~120 W |
| Peak (instantaneous) power draw | ~130 W (bursts for a few seconds, then settles back to ~120 W) |

Speculative decode varies strongly with text repetitiveness: the row above was
measured on real-text prompts; highly repetitive content (code, template text)
hits the chain drafter well and exceeds 60 tok/s in the same configuration;
sampling (adaptive γ) is slightly lower than greedy. Numbers vary with the
weight format (GGUF / hgn); ROCm 7.14 and 10.1 measured identical. All dates
are in 2026; test environment ROCm 10.1, production config
(PREFILL_CHUNK=16384).

## Weight Formats and Quality

The engine supports two weight formats. Service, API and speculative
decoding are identical; switch by using the other launcher:

| | hgn standard | hgn high quality (HQ) | GGUF UD-Q4_K_XL |
| --- | --- | --- | --- |
| Files | current default (`qwen38-flash-next-v2.hgn` + ngram/MTP) | converted from the original weights, see [HGN-HQ.md](HGN-HQ.md) (Chinese) | released by Unsloth, the same files llama.cpp uses |
| Routed experts | 4-bit (q4cp) | 4-bit (q4cp, imatrix-weighted) | mostly Q4_K / Q5_1 |
| Dense (attention, GDN, shared expert, embed, lm_head) | 4-bit | 8-bit (q8g32 overlay) | 8-bit (Q8_0) |
| KLD vs BF16 (lower is better) | 0.163 | **0.0558** | 0.0511 |
| top1 agreement with BF16 | 86.8% | 92.4% | 92.6% |
| Resident weights bpw / size | 4.55 / 66.6 GiB | 4.70 / 68.8 GiB | 5.25 / 76.9 GiB |
| Prefill (8K prompt, chunk 16384) | ~1730–1750 tok/s | about the same | about the same |
| Decode (no speculation) | ~34 tok/s | ~29 tok/s | ~29 tok/s |
| Launch | `start_hgn.sh` | `start_hgn.sh` (set `MODEL_FILE` / `OVERLAY_FILE`) | `start_gguf.sh` |
| Windows | yes | yes (not yet measured) | no |

- KLD: BF16 reference, wikitext-2, 64 chunks × 512, measured the same way as
  unsloth / llama.cpp (see [KLD.md](KLD.md), Chinese); llama.cpp on the same
  GGUF gives 0.049.
- Almost the whole quality gap comes from the dense bit width: 8-bit dense
  takes KLD from 0.163 to 0.063, imatrix-weighted experts bring it to 0.0558.
  The cost is more bytes read per decode token, ~16% slower decode (same as
  GGUF); prefill is unaffected.
- The two performance rows were measured 2026-10-07 on hgn standard (v2
  weights); the HQ / GGUF columns follow from the rule above (prefill is
  insensitive to bit width, decode is ~16% slower).
- Resident weights exclude the PLE n-gram table (hgn fp8 47.7 GiB, GGUF
  IQ4_NL 26.8 GiB), which stays on disk and is read on demand.
  `python3 tools/bpw.py` reports bpw per category for each file (reads headers
  only, a few seconds).
- The default configuration is still hgn standard. The HQ files pass
  `tools/hq_verify.sh`; deployment is described in section 6 of
  [HGN-HQ.md](HGN-HQ.md).

## Features

- **Speculative decoding chain**: ngram drafts first, MTP as fallback, with
  round-by-round fallback; greedy does bit-exact comparison, sampling
  resamples and compares against the target distribution, so the output
  distribution is identical to serial decoding. Enabled by default, no
  parameters needed; the draft length γ is chosen per mode (fixed 4 for
  greedy, adaptive by acceptance rate for sampling) and can be pinned with
  `MTP_GAMMA=1-8`. Highly repetitive content (code comments, template
  text) shows significant measured speedups.
- **Standalone 8-bit MTP draft weights** sidecar, with a higher acceptance
  rate than the built-in 4-bit draft head.
- **Vision**: supports image input (OpenAI `image_url`), with KV reuse
  across turns.
- **OpenAI-compatible API**: streaming, tool calling,
  `/v1/chat/completions`.
- **Native 256K / YaRN 512K context**, with native RoPE as the default and
  optional YaRN; a paged KV pool and two-tier prompt caching:
  in-RAM checkpoints at message boundaries (edit-and-resend replies
  instantly) plus KV snapshots recovered across restarts.
- **Concurrent requests**: multiple requests share one paged KV pool (4
  slots by default, 256K total — similar to llama.cpp's shared context);
  the GPU round-robins between requests and each request's output is
  bit-identical to running alone; long-prompt prefill is chunked to yield
  the GPU, capping other sessions' worst stall at ~0.6 s. Configuration
  and semantics in [CONCURRENCY.md](CONCURRENCY.md) (Chinese).
- **Two weight formats**: the native `.hgn` (Linux / Windows) and llama.cpp
  GGUF (Unsloth UD-Q4_K_XL, Linux); comparison above.
- **Model conversion tool**: HF safetensors → `.hgn`. The default output is
  the high-quality variant (8-bit dense overlay + weighted 4-bit experts); a
  llama.cpp-format imatrix is used if you have one, and conversion works
  without one too. Usable for your own fine-tunes of the same architecture
  (see [HGN-HQ.md](HGN-HQ.md), [CONVERT_EN.md](CONVERT_EN.md)).

## Requirements

- Linux + ROCm (HIP 7.x), GPU architecture `gfx1151`; or Windows + AMD GPU
  driver (the GPU needs VRAM carved out in BIOS), see the "Windows" section
- Available memory ≥ 100 GiB (pinned (page-locked) weights: ~68 GiB for hgn,
  ~80 GiB for GGUF, plus KV)
- Build dependencies: rocBLAS, hipBLASLt, rocPRIM; the API frontend also
  needs libpng, libjpeg, libwebp; nlohmann/json is vendored in the repository

## Quick Start

```bash
bash build.sh        # Build all required artifacts, output goes to build/
bash start_hgn.sh    # hgn weights: load the model and start the service (reads service.conf)
bash start_gguf.sh   # or GGUF weights (Unsloth UD-Q4_K_XL, the same files llama.cpp uses)
```

Both launchers read weights from `./models` and list any missing files before
exiting; configuration is centralized in `service.conf` (one section each for
hgn and GGUF; GGUF details in [GGUF.md](GGUF.md)). See [QUICKSTART_EN.md](QUICKSTART_EN.md)
for details.

For repeatable standalone performance measurements, build and run the
command-line benchmark described in [BENCHMARK_EN.md](BENCHMARK_EN.md):
`bash build.sh` on Linux or `bash build_win.sh` on Windows. Each default build
produces all required platform artifacts, including the engine, API, and
benchmark; Windows also builds the native launcher. It loads only the model
selected from `service.conf`, benchmarks all
`data/qsa-oracle/*.tokens` prompts for prefill, and reports TG speed plus MTP
acceptance for Python code, creative writing, and common-sense QA workloads.

Converting your own fine-tuned model (HF safetensors, same architecture):

```bash
# without an imatrix
python3 tools/flashnext2hgn.py /path/to/hf-model --out ./models
# with an imatrix (llama.cpp format: GGUF or legacy imatrix.dat)
python3 tools/flashnext2hgn.py /path/to/hf-model --out ./models --imatrix /path/to/imatrix.gguf
```

It writes the base `.hgn`, the 8-bit dense overlay, the 8-bit MTP draft, the
vision tower, the tokenizer and a ready-to-run `start.sh`; ~1.5 hours on 32
cores, ~125 GiB of disk, needs only numpy. `--classic` is the old data-free
converter (byte-identical output to before). High-quality conversion: see
[HGN-HQ.md](HGN-HQ.md); format and the old converter: see
[CONVERT_EN.md](CONVERT_EN.md).

## YaRN and Shared KV Pool Configuration

**Keep three independent concepts separate: the per-request context limit,
the service-wide KV pool capacity, and RoPE scaling.** Here K means 1024 tokens:
256K = 262144 and 512K = 524288. Context includes input and generated output,
including templates, conversation history, and image tokens, not just the
latest user message.

### Native versus YaRN

| | Native RoPE | YaRN factor 2 |
| --- | --- | --- |
| Purpose | The model's native 256K range; default mode | Inference configuration extending a sequence to 512K |
| Position encoding | `ROPE_FACTOR=1` preserves native frequencies and default amplitude | Adjusts RoPE frequencies and applies amplitude scaling to main-attention Q/K |
| Per-request limit | Set `MAX_CONTEXT=262144` | Set `MAX_CONTEXT=524288` |
| Short prompts | Default baseline | Static YaRN applies to short prompts too and can change the output distribution |
| KV storage | Determined by KV format and pool size | No KV compression or automatic pool expansion |

The factor is a position-encoding extension ratio, not a concurrency count,
KV compression ratio, or automatic capacity allocation. Changing `MAX_CONTEXT`
alone does not enable YaRN; changing `ROPE_FACTOR` alone does not change the
context limit. For more native short/medium-context sessions, enlarge the pool
without enabling YaRN.

The per-request limit must stay within the position-encoding range: with YaRN
enabled, `MAX_CONTEXT` may not exceed `ROPE_FACTOR × ROPE_ORIGINAL_CTX` —
the launchers and the engine refuse to start otherwise. With `ROPE_FACTOR=1`,
exceeding the native 256K only warns (positions beyond it are unvalidated).
For more concurrent KV capacity, enlarge `KV_POOL_TOKENS` instead of
stretching `MAX_CONTEXT`.

RoPE settings are **instance-wide**. Factor 2 applies to every request, not
only after token 262144; one instance cannot serve one request with native
RoPE and another with YaRN. Each sequence still has independent positions:
two 200K requests are not concatenated into one 400K sequence. Keep native
mode for primarily short-context traffic. To offer both modes, use separate
instances after checking memory/ports, or stop and restart with another config.

### service.conf parameters

Edit the existing lines rather than appending duplicates. With the
`${VARIABLE:-default}` syntax, same-name environment variables take precedence.
Restart both engine and API after changing the file.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `MAX_CONTEXT` | `262144` | Per-request input + output limit, not the sum of concurrent requests; must not exceed `ROPE_FACTOR × ROPE_ORIGINAL_CTX` (startup is refused otherwise; factor 1 above the native length only warns) |
| `ROPE_FACTOR` | `1` | Native at 1; use 2 for 512K and change MAX_CONTEXT separately |
| `ROPE_ORIGINAL_CTX` | `262144` | Native model length; keep this value with YaRN, do not set it to 524288 |
| `ROPE_BETA_FAST` / `ROPE_BETA_SLOW` | `32` / `1` | Frequency mixing boundaries; normally unchanged; engine requires beta_fast >= beta_slow > 0 |
| `ROPE_ATTN_SCALE` | `0` | Automatic amplitude: 1 for native, `1 + 0.1 * ln(factor)` for YaRN (about 1.0693 at factor 2); a positive value overrides it |
| `KV_PAGED` | `1` | Paged shared pool; required for PARALLEL > 1 with a pagination-compatible kernel combination |
| `KV_POOL_TOKENS` | `0` | Physical pool shared by all slots, in tokens; 0 follows MAX_CONTEXT, smaller values are clamped to MAX_CONTEXT, rounded up to 256-token pages |
| `PARALLEL` | `4` | Concurrent slots, range 1–8; neither multiplies the pool nor divides the per-request limit |
| `GDEC_KV_RESERVE_DECODE` | `4096` | Engine environment variable limiting decode tokens reserved at admission, not generated output; valid range 0–2147483647 |

Script launchers export `ROPE_*` as the corresponding `GDEC_ROPE_*` variables for
both engine and API. Direct engine invocation also supports `--rope-factor`,
`--rope-original-ctx`, `--rope-beta-fast`, `--rope-beta-slow`, and `--rope-attn-scale`;
the API uses `GDEC_ROPE_*` environment variables. Keep both sides consistent:
changing only the API's health metadata does not alter engine RoPE.

### Two different 512K configurations

`MAX_CONTEXT` expresses only the per-request limit; concurrent capacity is
always expressed with `KV_POOL_TOKENS`. Configurations where `MAX_CONTEXT`
exceeds `ROPE_FACTOR × ROPE_ORIGINAL_CTX` are refused at startup.

| Mode | MAX_CONTEXT | KV_POOL_TOKENS | ROPE_FACTOR | Example PARALLEL |
| --- | --- | --- | --- | --- |
| Default native 256K pool | `262144` | `0` | `1` | `4` |
| Native 256K per request, 512K total pool | `262144` | `524288` | `1` | `2` |
| YaRN 512K per request, 512K pool, single slot | `524288` | `0` or `524288` | `2` | `1` |
| YaRN 512K per request, 512K pool, multiple slots | `524288` | `0` or `524288` | `2` | `2` |

**Mode A: native RoPE, only a larger pool.** Two sequences dynamically share
512K; each remains limited to 256K. This does not grant one request a 512K
context. Replace the corresponding service.conf lines:

```bash
MAX_CONTEXT="${MAX_CONTEXT:-262144}"
KV_POOL_TOKENS="${KV_POOL_TOKENS:-524288}"
ROPE_FACTOR="${ROPE_FACTOR:-1}"
KV_PAGED="${KV_PAGED:-1}"
PARALLEL="${PARALLEL:-2}"
```

**Mode B: YaRN extends a sequence to 512K.** The total pool is still 512K.
This example enables two shared slots; set `PARALLEL` to 1 for single-slot
long-context operation. Replace the corresponding lines:

```bash
MAX_CONTEXT="${MAX_CONTEXT:-524288}"
KV_POOL_TOKENS="${KV_POOL_TOKENS:-0}"
ROPE_FACTOR="${ROPE_FACTOR:-2}"
ROPE_ORIGINAL_CTX="${ROPE_ORIGINAL_CTX:-262144}"
ROPE_BETA_FAST="${ROPE_BETA_FAST:-32}"
ROPE_BETA_SLOW="${ROPE_BETA_SLOW:-1}"
ROPE_ATTN_SCALE="${ROPE_ATTN_SCALE:-0}"
KV_PAGED="${KV_PAGED:-1}"
PARALLEL="${PARALLEL:-2}"
```

For a temporary override without editing the file, check first and remove
`--check` to start the service:

```bash
MAX_CONTEXT=524288 KV_POOL_TOKENS=0 KV_PAGED=1 PARALLEL=2 \
ROPE_FACTOR=2 ROPE_ORIGINAL_CTX=262144 ROPE_BETA_FAST=32 \
ROPE_BETA_SLOW=1 ROPE_ATTN_SCALE=0 bash start_hgn.sh --check
```

Linux GGUF uses the same variables with `start_gguf.sh`. Hardware acceptance
used Linux hgn + BF16 paged KV + WMMA + BTV; not every weight/kernel/platform
combination has been verified. Windows `start_win.sh` forwards YaRN settings,
but Windows 512K hardware acceptance remains pending. **The current native
start_win.exe does not translate service.conf ROPE_* into GDEC_ROPE_*.**
Do not assume editing the file enables YaRN when launching by double-click.
Advanced manual invocation needs matching GDEC_ROPE_* for both engine and
API, plus separate validation of Windows arena/device-memory limits.

### Two typical scenarios (reference)

**Scenario 1: no YaRN, two concurrent slots each up to the native 256K.**
This is mode A above: `MAX_CONTEXT=262144` + `KV_POOL_TOKENS=524288` +
`PARALLEL=2` + `ROPE_FACTOR=1`. The intuitive "set 512K so two concurrent
requests get 256K each" is expressed by keeping the per-request limit at the
native 256K and giving 512K to the shared pool. The two sequences share 512K
dynamically; a single request beyond 256K is still rejected.

**Scenario 2: YaRN on, two concurrent slots sharing a 512K pool.** This is
mode B above: `MAX_CONTEXT=524288` + `ROPE_FACTOR=2` + `KV_POOL_TOKENS=0` +
`PARALLEL=2`. When two requests together exceed 512K, the later one is
rejected at admission (the API returns an error, engine logs "rejected at
admission") while the earlier one is unaffected. Set `PARALLEL=1` if 512K
requests should never contend.

### How concurrency occupies the pool

`PARALLEL` specifies the maximum number of active slots. The pool is
preallocated at startup and never automatically multiplied by concurrency;
slots do not receive fixed `pool/PARALLEL` partitions. One sequence can use up
to its own context limit and others use the remaining pages. Requests queue
when all slots are busy. With an idle slot but insufficient KV budget,
multi-slot admission rejects the later request. GPU work interleaves at safe
scheduling boundaries; twice as many slots does not imply twice the throughput
and can increase request latency.

For `MAX_CONTEXT=524288, KV_POOL_TOKENS=0, PARALLEL=2, ROPE_FACTOR=2`:

| Requests | Behavior |
| --- | --- |
| One sequence close to 512K input + output; other slot idle | Can use the full pool; two slots do not halve its context limit |
| Two prompts around 200K each | Both can be admitted if their decode reservations also fit |
| Two prompts around 300K each | Exceed the shared 512K budget; the later request is rejected at the next safe scheduling boundary, not after a full prefill |
| Initially fit, then decode grows beyond its reservation | Can still exhaust the pool and trigger eviction/abort; admission does not guarantee unlimited generation |

Admission uses 256-token pages and conservative active-sequence accounting:

```text
target_pages = ceil(min(MAX_CONTEXT,
                        prompt_tokens + min(max_tokens, GDEC_KV_RESERVE_DECODE)) / 256)
other_active_slot_budget = max(target_pages, actual_mapped_pages)
available_budget = pool_pages - sum(other_active_slot_budgets)
request_need = target_pages; for live continuation, max(target_pages, mapped_pages)
```

- Non-prefix reuse resets the old slot. Released old pages are not counted
  again as the new request's growth baseline.
- Cache hits reduce prefill work, not the storage occupied by existing KV.
  Active sequences are charged independently even when they physically share
  a prefix; admission does not depend on those savings to allow oversubscription.
- Idle slots and RAM-checkpoint pins occupy physical pages; COW may need
  another page. Admission treats reclaimable pages as evictable rather than
  permanently subtracting every cache, and allocation retains pressure handling.
- On exhaustion, evict RAM checkpoints, then idle-slot KV; if still necessary,
  abort a later-admitted active request. A request needing pages can itself fail
  when it is the later one. Aborted generation returns an API error.
- `GDEC_KV_RESERVE_DECODE=4096` does not truncate output at 4096. Request budget
  and `MAX_CONTEXT` still limit generation. Smaller reservations can admit more
  requests but increase mid-generation starvation risk.
- `PARALLEL=1` bypasses multi-slot admission, not context/pool limits or cache
  eviction.

To hold two complete 512K sequences concurrently, set `KV_POOL_TOKENS` to at
least `1048576` and reassess memory, checkpoints and COW. This is neither the
default pool nor part of the reported hardware acceptance. A larger pool
mainly increases KV storage; more slots add per-sequence state such as GDN.
Equal pool sizes do not imply identical total memory with different
`MAX_CONTEXT`/`PARALLEL` settings.

### Startup checks, caching and validation scope

- Use BF16 KV + WMMA + BTV for 512K; the Linux launchers already set this
  combination. Above native 256K, the engine disables `GDEC_QSA_UNION`;
  do not force unverified kernel combinations.
- Inspect `--check` for the per-sequence limit, shared pool and slot count.
  After startup, verify engine logs for `RoPE: YaRN factor=2`, `[kvpage]` page
  count and KV format; `/health` should show `context=524288` and
  `rope_scaling.factor=2`. Native mode reports `rope_scaling=null`. The API takes
  its context from engine INFO, not just its command-line argument.
- The Linux hgn 512K test used about 108 GiB of gpu-accessible committed
  memory. KV array size is not the whole-machine budget; do not add UMA RSS
  and device measurements as independent allocations. Check weights, KV/BTV,
  workspaces and OS headroom before enlarging the pool.
- RoPE parameters are included in the SSD KV fingerprint. Restart to switch
  modes/parameters; incompatible snapshots will not be reused. Deleting them
  is unnecessary for correctness, but old files can occupy disk quota until
  LRU eviction.
- The 2026-09-30 Linux/gfx1151 results cover 500K prefill/needles, exact 512K
  boundaries, factor-2 two-slot concurrency, reset/COW/SSD restore, decode
  beyond reservations, and cancellation. At 256K, native prefill measured
  1589.4 versus 1582.1 tok/s with YaRN (re-measured 2026-10-07, about 0.5%
  difference), not a promise of zero overhead for every workload. A short
  probe had factor-2 versus
  native mean KLD 0.0234 and same-top 93.5%; outputs need not be identical.
  See [YARN-512K.md](YARN-512K.md) for implementation/limits and
  [YARN-512K-RESULTS.md](YARN-512K-RESULTS.md) for the hardware report.

## Windows

The Windows version has feature parity with the Linux version (engine +
OpenAI API + multimodal). Porting notes and measurements are in
[PORTING-WINDOWS_EN.md](PORTING-WINDOWS_EN.md). Builds run in Git Bash
(or double-click `build_win.bat`; Git is only needed at build time):

```bash
bash build_win.sh           # All required artifacts: engine, benchmark, API, launcher
bash build_win.sh api       # OpenAI API frontend
bash build_win.sh launcher  # Script-free launcher start_win.exe
```

For daily use, double-click `start_win.exe` (native Win32, no
Git/PowerShell needed): it brings up the engine + API dual processes,
without a console window: it only puts a tray icon in the notification area
(right-click: open dashboard / copy API URL / view logs / quit; double-click:
open dashboard), and output goes to `logs\`. For troubleshooting,
`start_win.exe --console` restores the console mode (Ctrl+C or closing the
window stops it). Configuration is **shared with Linux via
`service.conf`** (edit it to change the model file name or context
window); environment variables can temporarily override it. Clients
connect to `http://<host>:8731/v1`.

Distribution: copy `build/` + `start_win.exe` + `models/` to any gfx1151
Windows machine and it just works — **no ROCm/TheRock installation
needed**; only the AMD GPU driver, plus enough VRAM carved out for the GPU
in BIOS (a 256K context needs 96 GiB).

Differences from the Linux version:

- Only hgn weights are supported: usable VRAM on Windows is capped at
  about 96 GiB, and GGUF weights are larger (hgn saves ~11 GiB over GGUF)
  and do not fit — `start_gguf.sh` does not apply; hgn weights are
  produced by the conversion tool, see [CONVERT_EN.md](CONVERT_EN.md).
  The high-quality hgn works by swapping files: weight arena +2.2 GiB,
  estimated ~93.2 GiB at 256K / chunk 8192 (limit 95); not yet measured
  on Windows
- Image decoding supports PNG/JPEG via stb_image (WebP not wired up)
- Prefill chunk defaults to 8192
- Cold loading reads the full weights from disk (minute-scale, progress
  shown in console/logs)
- The launchers do not enable `GDEC_GEMM_WMMA` or `GDEC_GDN_FUSED` (the
  self-written WMMA GEMM and fused GDN kernel already promoted on Linux
  launchers, worth ~8-10% PP combined but unverified under TheRock — so
  Windows prefill uses hipBLASLt plus the legacy GDN path)

Known issues (root causes unknown; there are quite a few quirks, fixes
pending, priority very low):

- VRAM allocations above 41 GiB or 63 GiB fail
- The model hangs during decode (suspected console output backpressure: once
  the console is paused by a click/selection, child processes block on logging.
  Fixed: the launcher is now a tray app whose logs bypass the console, and
  kvsnap no longer prints while holding its lock; pending verification)

Build details are in [BUILD_EN.md](BUILD_EN.md).

## Documentation

- [QUICKSTART_EN.md](QUICKSTART_EN.md) — build, launch, configuration
- [BUILD_EN.md](BUILD_EN.md) — build environment details and
  troubleshooting
- [GGUF.md](GGUF.md) (Chinese) — GGUF weight loading, performance vs hgn
- [HGN-HQ.md](HGN-HQ.md) (Chinese) — high-quality hgn: one-step conversion
  (optional imatrix), results, deployment
- [CONVERT_EN.md](CONVERT_EN.md) — model conversion tool
- [KLD.md](KLD.md) (Chinese) — quality testing (KLD, same method as
  unsloth / llama.cpp)
- [MTP_EN.md](MTP_EN.md) — speculative decoding parameters and comparison
  methods
- [NGRAM_EN.md](NGRAM_EN.md) — ngram verification design, benefits, and
  known divergences
- [CONCURRENCY.md](CONCURRENCY.md) — concurrent requests (PARALLEL)
  configuration and semantics (Chinese)
- [YARN-512K.md](YARN-512K.md) — YaRN 512K implementation, pool admission and validation scope
- [YARN-512K-RESULTS.md](YARN-512K-RESULTS.md) (Chinese) — Linux/gfx1151 long-context and concurrency results
- [HGN-FORMAT_EN.md](HGN-FORMAT_EN.md) — the `.hgn` weight container
  format
- [GGUF.md](GGUF.md) — running directly from llama.cpp GGUF weights
  (Chinese)
- [data/README_EN.md](data/README_EN.md) — numerical regression benchmark
  (data/qsa-oracle) description
- [PORTING-WINDOWS_EN.md](PORTING-WINDOWS_EN.md) — Windows porting notes
  and measurements

## Tests

```bash
bash build.sh test     # Kernel unit tests, no model loading, expect ALL PASS
python3 tools/bpw.py   # bpw of the weights under models/ by category (headers only)
```

Quality (KLD) testing needs a BF16 reference; see [KLD.md](KLD.md).

## Acknowledgements

This project's implementation borrows from [halogen-flash-server](https://github.com/peonist-ai/halogen-flash-server) by peonist-ai. The `.hgn` weight container format is halogen's checkpoint container format — see [HGN-FORMAT_EN.md](HGN-FORMAT_EN.md). Many thanks to the halogen authors.

GGUF support heavily references [gufo](https://github.com/gufo-org/gufo) (MIT license): the routed-expert F16 WMMA GEMM kernel is ported from its RoutedF16GEMMKernel (`src/gpu/parts/26_kernels_moe_gguf.inc`); the LUT-decoded variants for hgn q4cp / GGUF IQ4 follow the same pipeline (`27_kernels_moe_lut.inc`); the GGUF↔engine tensor transform semantics reference its reference.cpp (`src/gguf_map.h`). Prefill optimizations such as HC gate fusion and producer epilogues writing the next GEMM's input directly also borrow from gufo's approach (comparison analysis in [GUFO-GAP.md](GUFO-GAP.md), Chinese).

The ngram speculative drafting approach and the two-tier prompt cache borrow ideas from the open-source [llama.cpp](https://github.com/ggml-org/llama.cpp) (MIT license); GGUF weights and the vision tower (mmproj) reuse the very same files as llama.cpp. See the "Attribution" section of [NGRAM_EN.md](NGRAM_EN.md).

## License

[AGPL-3.0](LICENSE)
