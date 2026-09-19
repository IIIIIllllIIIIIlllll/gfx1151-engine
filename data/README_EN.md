# data/qsa-oracle — Numerical Regression Baselines

*中文版:[README.md](README.md)*

This directory is the engine's **output diffing baseline**: greedy reference
outputs with fixed prompts and fixed generation lengths, used to verify that
code changes have not altered the model's numerical behavior. Any engine
change should pass this set of baselines before being considered for merge.

## File Layout

- `<N>.json` — 5 groups of oracle baselines (N = 2048 / 2051 / 2052 / 8192 / 32768).
  Fields: `prompt_ids` (input token sequence), `gen` (generation length),
  `response` (reference greedy output, OpenAI response shape).
- `<N>.tokens` — raw token lists of the corresponding prompts (plain text,
  one per line), convenient for manual inspection and reuse; the `65536` /
  `131072` files are extra-long prompt material for generating future
  long-context baselines.
- `<N>.gdec.log` — per-step top-5 logit records captured engine-side when
  collecting the baselines, used to locate the first divergence point when
  outputs differ.
- `server-health.json` — reference sample of the API `/health` response shape.

## Tools That Consume These Files

- `tools/snapshot_regress.py` — reads the 5 groups of `.json` files, replays
  each against a running engine and compares the output text, printing
  MATCH/DIFF (plus 4 short cmp4 cases whose baselines live in
  `tools/server_greedy24.json`). The "Reproduction" section of NGRAM_EN.md
  gives the full procedure.
- `tools/oracle8192_one.py` — replays the 8192 group alone.
- `tools/accept_checks.py` — uses the 32768 group's prompt to assemble a 64K
  long-context check.

The reference outputs are produced by greedy decoding; on near-tied tokens,
different batch shapes may legitimately land on the other side of the tie
(NGRAM_EN.md has a quantitative analysis), so a DIFF on individual cases
requires manual judgment — it is not an automatic failure signal.
