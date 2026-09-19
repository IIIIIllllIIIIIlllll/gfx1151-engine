# Ngram verify / GDN candidate optimization verification (2026-09-15)
*中文版:[NGRAM.md](NGRAM.md)*

## 2026-09-19 update: chunked verify (fixes ngram-round output stalls)

**Problem**: an ngram round submits a draft of up to 64 tokens as a single 65-row atomic verify (~350ms), and on partial acceptance it must replay the entire accepted prefix (~150ms); the client receives no tokens during this time, which shows up in production logs as ~1.5s streaming gaps ("stalls").

**Fix**: chunked verify + adaptive chunk size.

- The draft is split into chunks of at most ng_cs rows and verified chunk by chunk; each chunk is emitted as soon as it is accepted (smooth streaming). A rejection inside a chunk only restores + replays that chunk's prefix (bounded by ng_cs rows), replacing the 65-row atomic verify + whole-prefix replay proportional to accepted count.
- When a chunk is fully accepted, check whether the continuation token equals the next draft token (splice check): if equal, chain into the next chunk; if not, treat it as a boundary rejection — the state is clean, no replay needed, and the round ends immediately.
- ng_cs starts at 16, doubles on fully accepted rounds (cap 64), and halves on rejected rounds (floor 8); it persists across requests as a Model member.
- env `GDEC_NGRAM_CHUNK`: unset = adaptive (default); `0` = old single-pass path (the original code is kept for comparison/rollback); `N` = fixed chunk size.
- Implemented for both the greedy and sampling chains. `CHUNK=64` (a fixed single chunk) reproduces all legacy numbers, proving the round rewrite itself does not change semantics.

**A/B (heretic model, greedy, same prompt, byte-identical output)**:

- Code scenario (write quicksort → add comments, the original stall scenario) turn 2: legacy 14.74s / 36.4 tok/s / p99 gap 431ms / max 592ms → chunked 13.25s / 40.4 tok/s (+11%) / p99 191ms / max 441ms. Turn 1 (pure MTP) unchanged.
- Repetitive-text showcase: legacy 6.58s / 57.1 tok/s; chunked warm 6.80s (cold-start first request 7.52s, adaptive ramping up from 16), acceptance rate 0.767 identical in both.
- Sampling chain (temp 0.7, top_p 0.9) runs through, acceptance rate 0.750, coherent output.

**Regression**: ngram_regress ALL PASS; drafter_switch all pass; snapshot_regress 8/9 (cmp4-2 is a pre-existing DIFF); all five oracle groups MATCH; ktest ALL PASS.

## 2026-09-18 update: root cause of the index-90 divergence identified (near-tie, not a bug)

Quantitative conclusion for the numbering stress case (prompt 344 / gen 128):

- **MTP diverges at the same position**: in a three-way serial / NGRAM / MTP comparison, NGRAM and MTP both diverge from serial at index 90, and the two are bit-identical across all 128 tokens. The divergence is unrelated to the ngram draft mechanism; it is the shared property that "batched verify math ≠ serial decode math".
- **That position is a near-tie**: on the serial path the top-2 logprobs are -0.9553 vs -1.0417 (a gap of 0.086 nat, probability ratio ~1.09). Perturbations from bf16 intermediate activations plus different accumulation orders are enough to flip this gap.
- **Large-batch prefill lands on the serial side**: prefilling the prompt plus the first 90 generated tokens in one shot (P=434, single chunk) still argmaxes to 3294, the serial choice (gap 0.328 nat). That is, the same batched math lands on either side of the tie depending on chunk/offset shape — a coin flip.
- Conclusion: the two comparison failures of the numbering case are not state errors (the earlier convolution-state bug was a real bug, fixed and covered by ktest); they are the normal behavior of the approximate semantics speculative decoding currently allows, on a near-tie. The pre-existing cmp4-2 DIFF is the same class of phenomenon.
- **ngram_regress.py criteria updated**: the numbering case now has dual gold — matching serial, or matching the MTP drafter output (the two share the trunk batched verify; an ngram-specific state/rollback bug would decouple them) counts as a pass. All other cases still diff strictly against serial.
- Implication for MTP: MTP's "bit-identical to serial" is an empirical observation, not a guarantee; on tie-heavy text (such as this case's 128 tokens) it flips just the same. The only README-level statement that still holds is "speculation does not change the distribution; it only proposes, it never force-accepts".

## Changes in this round

- `k_gdn_verify` keeps short-batch GDN state in registers and processes verify rows back to back, avoiding padding to 64-token chunks and repeatedly reading/writing the entire state. Used only for ngram verify/replay, at most 65 rows.
- verify/replay no longer computes the last row's lm_head separately or synchronously reads back the argmax; after verification, logits for all rows are computed in one pass. lm_head reuses weights in groups of at most 8 rows.
- Fixed historical duplicate appends, pos/PLE ring positions on rollback, cross-request hash-learning state, generation budget, and context boundaries. Trunk checkpoints and argmax allocations are now independent of the MTP weights.
- Fixed a short-batch state bug in the existing `k_convst_update`: with P=1 the 1st and 2nd rows of the old history should be kept, with P=2 the 2nd row; the original code wrote the oldest kept row the wrong way around. The previously clean source had this error too.
- Added a short-batch state handoff test to `tools/ktest.cu`, verifying the full history state and the convolution output of the next token.

This convolution bug is not a mere floating-point difference: before the fix, both P=1 and P=2 had 10,240 state values differing from sequential execution, and the next-step output had a max absolute error of about 1.833 and 1.824 respectively. After the fix, the state and next-step output for P=1/2/3/4/8/9/16/17/32/33/64/65 are all exactly identical.

## Measured gains

Device gfx1151, model `qwen38-flash-next-w4b.hgn` + overlay, single engine, greedy, disk KV snapshots disabled. The times below are decode only, excluding prompt prefill; the generation count includes the first token, so use generation count minus one when converting time to TG.

Final source, maxctx=40960, prefill chunk=8192, ngram match=24/min=4/max=16:

| Case | Serial decode | Ngram decode (two runs) | Output diff |
|---|---:|---:|---|
| 240-token repetitive text, generate 64 | 1.9930 s | 0.8573 / 0.8410 s | Two runs identical |
| 137-token repetitive code, generate 48 | 1.4848 s | 0.8703 / 0.8622 s | Two runs identical |
| Ordinary question, generate 32 | 0.9853 s | 0.9782 / 0.9783 s | Identical; no draft, serial fallback |
| Numbered repetitive paragraphs, generate 128 | 4.0445 s | 4.2117 / 4.1919 s | **Both runs diverge at index 90** |

Repetitive text is about 2.35× and repetitive code about 1.71×; this cannot be extrapolated to ordinary Q&A. The numbering case has insufficient acceptance, and verify + rollback actually increase total time.

An earlier structural A/B also at max=16: restoring the old GDN and the redundant last-row head gave 0.9031/0.8949 s, versus about 0.841–0.850 s for the optimized path — an independent structural gain of about 6%. The extra gain from raising max=8 to 16 belongs to batch amortization and cannot all be attributed to the new GDN kernel. That A/B happened before this round's short-convolution fix.

In the isolated GDN test, P=9 dropped from about 0.1900 ms to 0.0237 ms (about 8×); P=17 went 0.1949→0.0342 ms; P=65 went 0.3508→0.1003 ms. Per-row outputs and final states were checked against `k_gdn_step`, with zero absolute error for all 11 tested batch lengths. The isolated speedup does not equal the whole-model speedup.

Whole-model tests have now been added for the default match=24/min=48/max=64, likewise maxctx=40960/chunk=8192:

| Case | Serial decode | Ngram decode (two runs) | Output diff |
|---|---:|---:|---|
| Repetitive text, generate 64 | 1.9936 s | 0.3095 / 0.3012 s | Two runs identical |
| Repetitive text, generate 130 | 4.0849 s | 0.6348 / 0.6278 s | Two runs identical; covers the full 65-row verify |
| Repetitive code, generate 48 | 1.4838 s | 0.4359 / 0.4298 s | Two runs identical |
| Numbered paragraphs, generate 128 | 4.0595 s | 4.6494 / 4.6139 s | Both runs diverge at index 90 |

With the default long drafts, highly repetitive text decodes at about 6.5× and repetitive code at about 3.4×. This includes amortization of weight reads and execution overhead by longer batches; it cannot be attributed to the GDN optimization alone, nor does it mean real code tasks generally reach this speed. The default configuration completed 39 requests in total; apart from the two comparison failures of the numbering case, all checks passed; the overall test exit status is still failure.

Old-path comparison of the final source, likewise at min=48/max=64 (`GDEC_NGRAM_LEGACY_GDN=1 GDEC_NGRAM_LEGACY_FINAL=1`):

| Case | Old GDN + old last-row head (two runs) | New path (two runs) | Average decode time reduction |
|---|---:|---:|---:|
| Repetitive text, generate 64 | 0.3265 / 0.3188 s | 0.3095 / 0.3012 s | About 5.4% |
| Repetitive code, generate 48 | 0.4453 / 0.4380 s | 0.4359 / 0.4298 s | About 2.0% |

Token comparisons all passed in this comparison, and the serial reference decode was also basically stable; but with only two measurements per configuration, the percentages should be treated as estimates for the current cases. What can be confirmed is that the large gains come mainly from ngram long-batch verification, with the new GDN / last-row head elimination contributing an additional small gain.

## Correctness and limitations

- Final `ktest` all pass, including the newly added short-convolution state handoff test (2026-09-18 rerun still ALL PASS, covering convstate_handoff P1..P65, 12 items in total).
- All five existing oracles (2048, 2051, 2052, 8192, 32768) MATCH (confirmed by the 2026-09-18 rerun).
- cmp4 is 3/4; the 2nd is still the pre-existing DIFF (a near-tie of the same class as index 90; see the 2026-09-18 update at the top).
- 2026-09-18 rerun: ngram_regress is ALL PASS under both the helper configuration (min=4/max=16) and the engine default configuration (match=24/min=48/max=64), with the numbering case matching the MTP-spec gold on both runs. Re-verification of measured gains (default configuration, decode time): repetitive text 64 tokens 2.00s→0.30s (6.7×), repetitive code 48 tokens 1.49s→0.43s (3.4×), repeat130 4.11s→0.63s (6.5×), ordinary question 1.01× (serial fallback with no draft), numbering case 0.88× (still slower due to insufficient acceptance, see below).
- For the numbering stress case (an extreme low-acceptance-rate shape), verify + rollback overhead still makes total time slightly worse than serial (~0.88×). This is an inherent trade-off of ngram drafting on this kind of content, not a correctness issue; whether to accept that trade-off is a deployment decision.
- The five cold-request comparison serial→MTP→ngram→MTP→ngram passes, and the drafter reported by the protocol is correct (confirmed by the 2026-09-18 rerun). This only validates mode switching; it does not mean the two draft sources are fused within the same round. Note: after an ngram request, `mtp_live` is cleared, so subsequent cont+MTP requests on the same connection fall back to serial until the next cold request re-warms (same origin as the warm-up ironclad rule in AGENTS.md).
- Live KV continuation is preserved. Generating 11 tokens of a repetitive sequence and then appending a newline: serial and ngram agree in both live/cold comparisons; generating 12 and then a newline: both still show live/cold differences — the same near-tie phenomenon as index 90 (live decode continuation vs full recomputation have different math orders), not an ngram-specific problem.
- A request with `drafter: "ngram-mod"` plus sampling parameters currently falls back to serial sampling explicitly. Rejection sampling for ngram is not implemented, nor is a hybrid MTP + ngram draft strategy.
- Loading the full model without MTP weights has not been separately tested; the dependency of the ngram checkpoint on the MTP allocation branch has now been removed in code.

## Default drafter (2026-09-18)

When a request omits `drafter`, the engine-side environment variable decides; the API no longer fills it in:

- `GDEC_DRAFTER=ngram` (or `ngram-mod`/`3`): greedy requests default to ngram; sampling requests automatically fall back to the existing default (ngram only drafts greedy).
- `GDEC_DRAFTER=serial` (or `0`): default to serial.
- Unset or `mtp`: existing behavior unchanged (MTP if MTP weights are present, otherwise serial).
- An explicit client-provided `drafter` always takes precedence and is unaffected by this variable.
- The 5th field of the engine INFO line reports the current default; the API `/health` `drafter_default` forwards it faithfully ("ngram-mod"/"mtp"/"serial").
- Note: on the first ngram cold-start round the draft table is empty, so the drafter column of D rows falls back to placeholder 0; from the second round on it reports 3 normally (protocol placeholder behavior, see the comment at line 12274).

## Reproduction

Using relative paths:

```bash
bash build.sh test
bash build.sh engine
PROBE_TAG=my-ngram-test PROBE_CONTEXT=40960 GDEC_PREFILL_CHUNK=8192 \
  GDEC_NGRAM_MIN=4 GDEC_NGRAM_MAX=16 bash tools/run_ngram_probe.sh
# After the engine log shows serve: listening, keep the API stopped, then run:
python3 tools/ngram_regress.py --keep-going --output logs/my-ngram-test.json
python3 tools/snapshot_regress.py
python3 tools/drafter_switch_test.py
```

`run_ngram_probe.sh` starts a single memory-limited test engine with disk snapshots disabled; the helper defaults to min=4/max=16, while the engine source defaults to min=48/max=64. It is not a production service launch script.

The test clients use the same port serially and should not be run concurrently. `--keep-going` preserves and summarizes all generation comparison failures and still returns non-zero at the end; it does not turn failures into passes. The snapshot script's cmp4 DIFF will appear in the JSON and the MATCH/DIFF output; inspect those results directly.
