# MTP Parameters and Comparison Methodology

*中文版:[MTP.md](MTP.md)*

The service defaults to `MTP_GAMMA=0` (auto): greedy (temperature=0) uses a fixed 4; sampling adapts per round, picking 3–7 from running per-depth acceptance estimates (see `Model::GammaCtl`). Setting 1–8 forces that fixed length for every request. For manual comparisons, change only the length each time:

```bash
MTP_GAMMA=1 bash start_hgn.sh    # start_gguf.sh for GGUF weights
```

After stopping that round of the service, replace 1 with 2 or 3. You can also change the default value of `MTP_GAMMA` in `service.conf`.
**Changing the draft length requires restarting the engine, not recompiling.** This round did not restart or alter any running service on your behalf.

Script fix note: service mode actually reads `GDEC_SPEC_GAMMA`; the old script only passed `--gamma`, which applies solely to offline `--spec-gen`. The launchers (start_hgn.sh / start_gguf.sh) export `MTP_GAMMA` as `GDEC_SPEC_GAMMA`. When launching the engine manually you can write `GDEC_SPEC_GAMMA=2 bash tools/run_capped.sh ...`. Do not use the gamma field in the request JSON; the current API does not support per-request draft length changes.

## Request Parameters

The MTP weights are set via `MTP_FILE` in `service.conf` (by default it points to the standalone 8-bit sidecar `qwen38-flash-next-mtp.hgn`; set `MTP_FILE=""` to fall back to the overlay's built-in 4-bit draft head). The default drafter is chain (ngram first, MTP as fallback; effective under both greedy and sampling; under sampling, ngram rounds resample from the target distribution following llama.cpp semantics and accept by comparing token ids, while MTP rounds keep exact rejection sampling, so the output distribution matches serial sampling); `GDEC_DRAFTER=mtp` is pure MTP, `GDEC_DRAFTER=ngram` is pure ngram, and `GDEC_DRAFTER=serial` is the serial baseline. Drafter selection is not controlled by requests: the HTTP API does not accept a `drafter` field (it is silently ignored). When temperature is not provided, the API defaults to `temperature=1.0, top_k=20, top_p=0.95, min_p=0`, which is sampling mode.

For controlled comparisons, fix these fields on the same prompt and compare different gammas (change `MTP_GAMMA` and restart):

```json
{
  "temperature": 0,
  "presence_penalty": 0,
  "frequency_penalty": 0,
  "max_tokens": 512
}
```

temperature=0 is a greedy control and is not guaranteed to raise the acceptance rate. Then repeat the comparison with your everyday sampling parameters; run each group 3 times and set aside the first warm-up run. Changing temperature/top_k/top_p/min_p or the penalties changes the output distribution, so you cannot treat an acceptance-rate change alone as a lossless speedup. Fixing the seed during sampling can aid comparison, but different gammas do not guarantee token-by-token agreement under the same seed.

This implementation has no separate draft_temperature or tunable acceptance threshold: greedy mode verifies whether tokens match, and sampling mode performs rejection sampling by p/q. GDEC_MTP_NORM/HAGG are legacy structure-debug switches; keep their default values.

## Reading the Logs

```bash
grep -E 'decode-live:|spec(-sample)?:|depth acc:|spec(-sample)? time:' logs/*/engine.log | tail -30
```

Focus on tok/s, commit/round, depth acc, and draft/verify/rollback times. The `1:x 2:y 3:z` in depth acc are the cumulative proportions of accepting the first 1/2/3 drafts, not independent per-layer hit rates.

The overall acceptance rate in the logs is accepted drafts / proposed drafts. Shortening gamma often makes this percentage look better without necessarily making generation faster. For example, at gamma=3 with a 30% acceptance rate, about 0.9 drafts are accepted on average; adding the verification token for each round gives roughly 1.9 tokens/round (ignoring truncation at the end).

If the first draft has a high acceptance rate and the next two drop off quickly, try gamma=1/2 first; if even the first item is very low, shortening the length mainly reduces waste, and you should also compare whether serial is faster. In the end, choose based on stable actual generation tok/s and output quality, not on acceptance rate alone.

## Decode speed-up switches and offline testing

All on by default; turn off only for troubleshooting:

| Switch | Effect |
|---|---|
| `GDEC_SMS_GPU=0` | Disable GPU sampling prep (bias/penalty corrections + top-k on the GPU, only candidates copied back; active for top_k 1..64, otherwise the host full-vocab path is used). Output distribution unchanged; the exact tokens for a given seed may differ |
| `GDEC_DRAFT_LM_Q4=0` | Drafts use the production lm_head again (default: the base file's 4-bit copy; affects drafts only, never committed tokens; ~350 MB extra VRAM) |
| `GDEC_ARGMAX_OLD=1` | Single-block argmax (differs only on exact ties) |

Offline reproduction of the API decode paths (`--spec-gen N`, or `tools/pp_prod.sh <tok> LABEL SPEC=256 ...`):

- `GDEC_SPEC_CHAIN=1`: chain drafter (API default); unset = pure MTP.
- `GDEC_SPEC_SAMPLE=temp,top_k,top_p,seed[,presence,frequency]`: rejection-sampling mode; unset = greedy.
- `GDEC_SMS_CHECK=1`: compare the GPU sparse distribution with the host full-vocab `prepare()` on every row, printing `[sms-check] n=… bad=…` (slow; verification only).

One-shot check: `bash tools/tg_verify.sh` (kernel unit test + greedy token identity + sampling distribution self-check + sampling speed, ends in PASS/FAIL).

## Adaptive γ (auto)

The controller keeps two sets of per-depth acceptance estimates, split by whether the previous round was fully accepted. A round that follows a full accept is clearly more likely to be accepted (real-text sampling: .93/.86/.83 vs .77/.71/.73). A deeper γ makes full accepts rarer, so the effective acceptance falls as γ grows. A single estimate can't see this and drifts deep. The controller combines the two sets through their stationary probabilities into an expected commit, divides by the cost `1 + r·γ`, and picks γ.

| Switch | Effect |
|---|---|
| `GDEC_SPEC_ADAPT=0` | Sampling auto falls back to fixed 3 |
| `GDEC_SPEC_ADAPT_GREEDY=1` | Greedy auto adapts too (default fixed 4; paired tests put the two within ±1%, and fixed is reproducible) |
| `GDEC_SPEC_ADAPT_R` | Slope/intercept ratio of the round cost model `1 + r·γ` (measured ≈ 42 + 10.7·γ ms → 0.25; default 0.3 offsets the model's optimism about deep γ) |
| `GDEC_SPEC_ADAPT_ALPHA` | EMA step of the per-depth acceptance estimates (default 0.05) |
| `GDEC_SPEC_ADAPT_HYST` / `_MIN` / `_MAX` | Switch margin (default 0.02) / γ floor 3 (runs stuck at γ2 measured ~20% slower than γ3) / cap 7 (γ=8 = 9 verify rows, past the P≤8 fast paths) |

Offline: `--gamma 0` (pp_prod.sh `GAMMA=0`) runs adaptive. Tuning tools:

- `tools/gamma_trace.sh` + `tools/gamma_sim.py`: `GDEC_SPEC_TRACE=1` prints per-round `γ,accepted,ms`; replay controllers offline. `MODEL=markov` (two-state generator; fixed-γ commit/round within ~2% of measured) is the most accurate; the default pos/round replays underpredict shallow γ by 7–9% — use them for ranking only.
- `tools/gamma_force_check.sh`: `GDEC_SPEC_FORCE=<ids file>` makes greedy spec verify against a reference sequence, so every γ config commits identical tokens — a paired comparison with only timing noise (~±1%). Comparing γ values directly, greedy trajectories diverge at near-ties and a single prompt varies ±15–27%.
- `tools/gamma_sample_check.sh`: sampling mode pooled over offsets × seeds (sampling trajectories can't be pinned).
