# MTP Parameters and Comparison Methodology

*中文版:[MTP.md](MTP.md)*

The service currently defaults to a draft length of 3, with a range of 1–8. Start by comparing 1, 2, and 3, changing only the length each time:

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
