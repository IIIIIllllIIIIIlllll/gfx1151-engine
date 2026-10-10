# YaRN 512K

This branch adds configurable YaRN RoPE to Qwen3.8-Flash-Next, targeting an
extension from the native 262144-token context to 524288 tokens.

## Reported hardware validation (2026-09-30)

See `YARN-512K-RESULTS.md` for the real-machine report (Radeon 8060S / gfx1151,
122 GB UMA, HIP 7.14.60850). The engine/API build and kernel tests passed.
With a 524288-token configuration, BF16 paged KV + WMMA + BTV and one sequence:

- Six prompt lengths from 34794 to 500191 tokens completed prefill and decode.
- The 331K, 441K and 500K needle probes retrieved all three codewords each.
- The longest request took 418.8 s wall time, including decode; it is not a pure prefill timing measurement.
- Reported peaks were 45.8 GiB device memory and 76.6 GB process RSS, with no reported arena fallback, OOM or illegal access. These UMA measurements must not be added as independent allocations.
- Same-configuration SSD restore hit 500184/500191 prompt tokens; factor 1 correctly missed those snapshots.
- The 2048-token regression sample had factor-1 mean KLD rounded to zero, p999 KLD 0.000049 and 100% same-top. This supports numerical agreement on that sample, not a proof of bit-exact logits.
- Factor 2 changed the short-context distribution (mean KLD 0.023384, same-top 93.529%); native factor 1 remains the default.

The follow-up tests also covered the exact 524288 context boundary: full and
over-limit requests were rejected cleanly, while 524287-token prompts were
trimmed to exactly the available output budget. A 352K-token PPL probe had
only +0.4% tail PPL relative to its shallow-position comparison, with no
NaN/Inf or spike. A further PARALLEL=2 test with snapshots disabled covered
two approximately 24K prompts and pool over-subscription by two approximately
311K prompts. After adding the admission gate, the later over-subscribing
request was rejected in 0.3 s instead of spending about 404 s prefilling; the
earlier request completed in 241 s instead of 444 s. The small concurrent
prompts retained identical output text and 3/3 needle hits. This does not
establish that two full 512K sequences fit in the default shared pool.
Windows and YaRN M-RoPE parity remain separate deployment-mode checks.
The reported `/cache` timeout is a pre-existing CSTAT protocol issue and is
outside this change.

The earlier hardware results predate the admission synchronization/accounting
repair below. Section 15 of the report now covers the repaired version:
Linux engine/API rebuild, kernel tests, g++ host regression and host
ThreadSanitizer all passed. The five hardware retest groups also passed
with BF16 paged KV + WMMA + BTV, including PARALLEL=2 full-pool boundaries,
non-prefix slot reuse, RAM-checkpoint COW/eviction, SSD restore, and the
decode-overflow/cancellation suite. Acceptance applies to this tested
Linux configuration, not every platform or kernel combination.

## Integration with main (2026-09-30)

The merge combines feature commit `cb4950f` with main at `f64bdb4`, retaining
API disconnect cancellation (`f9cae14`), configurable loopback-default engine
binding (`52e0aad`), and Windows PLE prefetch (`f64bdb4`). The four overlapping
header/startup/config sections were resolved additively, not by replacing
either side wholesale.

Post-merge checks passed on the Windows host: KV admission regression,
PLE prefetch host regression, engine address/socket and launcher dry-run
regressions, API rebuild plus disconnect suites with one/two slots, YaRN
formula factors 1/2, and native/YaRN API health metadata with matching fake
engine INFO. The fake engine's default INFO advertises 256K; the API correctly
uses engine INFO as authoritative rather than blindly trusting --context.

No local HIP toolchain is available, so the merged GPU engine has not been
rebuilt or run here. The section 15 hardware results cover the pre-merge
feature implementation. Rebuild on the gfx1151 host and run a short native
and factor-2 smoke test before deploying the merged commit.

## Recommended configuration

Use YaRN factor 2 for 512K on the Linux HGN service:

```bash
MAX_CONTEXT=524288 \
ROPE_FACTOR=2 \
ROPE_ORIGINAL_CTX=262144 \
ROPE_BETA_FAST=32 \
ROPE_BETA_SLOW=1 \
PARALLEL=1 \
KV_POOL_TOKENS=0 \
bash start_hgn.sh --check
```

Remove `--check` after the configuration check passes. The Windows launcher
uses the same environment variables.

The default `ROPE_FACTOR=1` preserves native RoPE. Do not enable YaRN
statically for short-context service traffic.

RoPE settings are global to an engine instance, not selected per request.
Use separate instances or restart with the appropriate configuration when
switching between native and YaRN workloads.

## Shared KV pool and admission

`MAX_CONTEXT` limits each sequence; `KV_POOL_TOKENS` controls the aggregate
shared pool. With `KV_POOL_TOKENS=0` and `MAX_CONTEXT=524288`, all concurrent
slots share one 524288-token pool, not a separate 512K allocation per slot.
Increasing the pool requires a separate memory check.

In parallel paged mode, admission reserves a total sequence target:

```text
target = ceil(min(MAX_CONTEXT, prompt_tokens + capped_decode) / 256)
need = target for reset; max(target, mapped_pages) for live continuation
available = pool_pages - sum(max(target, mapped_pages) for other active slots)
```

`QWENOX_KV_RESERVE_DECODE` caps decode reservation at 4096 tokens by default;
it does not cap generation. It accepts integers from 0 through INT_MAX;
invalid values fail engine serve startup. Decode beyond the cap counts its
actual mapped pages at subsequent admissions, and still uses the existing
eviction/abort fallback on pool starvation. PARALLEL=1 bypasses this gate.

The synchronization/accounting repair in `src/gpu/parts/51_host_cfg.inc`:

- Obtains the GPU turn before reading mapped counts or claiming a slot. It releases `g_sm` before waiting for `g_turn`, then rechecks cancellation and slot selection under `g_sm`. Successful admission hands the owned GPU turn directly to generation.
- Removes free-deque reads and the old `pages0` baseline. Each slot stores its total target; reset cannot leave the reused slot's released pages counted as outstanding demand.
- Removes fixed `+2` page slack and clamps reservations at MAX_CONTEXT, so a single active full-pool request is not falsely rejected in parallel mode.
- Counts each active sequence independently, conservatively allowing for prefix sharing/COW. Idle-slot and checkpoint-only pages are reclaimable by the existing pressure handlers rather than charged as active demand; slots already aborted are excluded.

Admission now waits for a GPU-safe scheduling boundary before deciding.
The historical 0.3 s rejection time is not a latency guarantee for this
repair. Conservative counting may reject an otherwise physically fitting
shared-prefix request, and capped decode reservation is not an OOM-proof
promise for unlimited generation.

Host regression passed with Windows LLVM Clang 22.1.8: page arithmetic,
non-prefix reset, GPU-owner synchronization, rejection/cancellation cleanup,
continuation and changed mappings, parallel full-pool edge, single-slot
bypass, and four-thread admission/reset churn (400 requests). The report's
section 15 confirms g++ and host ThreadSanitizer regression passes, plus
hardware coverage of these admission fixes. The oversubscribing request
was rejected after 5.3 s at a GPU scheduling boundary, while the earlier
request completed in 242.1 s. The 90-page reset plus 1800-page peer both
completed in the 2048-page pool, and parallel single-active budget-1/25
requests generated exactly their remaining context budgets. See
`HANDOFF-YARN-512K.md` for the completed retest checklist.

## Implementation scope

- Main attention, decode, prefill, MTP, and M-RoPE share the YaRN inverse-frequency table.
- The default YaRN attention factor is `1 + 0.1 * ln(factor)`; it is applied to main-attention Q/K, not the indexer's partial RoPE.
- `QWENOX_ROPE_*` environment variables and `--rope-*` engine options are supported. Launchers pass the same settings to the engine and API.
- The KV snapshot fingerprint includes all RoPE parameters, so switching YaRN cannot reuse SSD KV pages created with a different position encoding.
- `QWENOX_QSA_UNION` is disabled automatically above 256K because its block IDs are `uint16_t`.

## Memory expectations

For 512K, use `QWENOX_QSA_KV_BF16=1`. With 12 trunk QSA layers plus the MTP layer,
the row-major BF16 K/V arrays use about 13 GiB, comparable to 256K FP32 K/V.
The tested BTV configuration additionally stores about 6 GiB of transposed V;
indexer keys, guard pages, weights and workspaces are extra. At 1M, row-major
BF16 K/V alone doubles to about 26 GiB (BTV adds about 12 GiB), so the 13/26 GiB
figures must not be used as total device-memory budgets. BF16 pagination has
WMMA constraints; production configuration should keep
`QWENOX_QSA_WMMA=1` and `QWENOX_QSA_WMMA_BTV=1`, and should not enable
`QWENOX_QSA_UNION`.

The 256K offline prefill benchmark measured 1374.4 tok/s with native RoPE and
1370.3 tok/s with YaRN factor 2, about 0.3% difference at noise level; both
runs produced the same generated token.

## Verification

The formula can be checked without a GPU:

```bash
python tools/yarn_verify.py --factor 2
python tools/yarn_verify.py --factor 1
```

The repaired admission path has a host-only regression that compiles the
actual production admission code against a fake model; no HIP or weights
are needed:

```bash
python tools/kv_admission_test.py --cxx clang++
# Optional on Linux with ThreadSanitizer support:
python tools/kv_admission_test.py --cxx clang++ --tsan
```

Both `build.sh test` and `build_win.sh test` run this regression before
kernel tests. Linux g++ host ThreadSanitizer and the full GPU build passed
on the test machine per report section 15. ThreadSanitizer covers the
extracted admission code with a fake model, not the entire HIP engine.

The reported Linux single-sequence YaRN acceptance checks and repaired
parallel admission regressions are complete for the tested configuration.
Optional follow-up coverage includes more
long-context PPL/retrieval samples, Windows, and YaRN M-RoPE parity if those
deployment modes are needed.

The existing kernel suite passed, but it does not explicitly select factor 2.
A dedicated factor-2 kernel regression remains useful in addition to the
successful end-to-end factor-2 tests.
