#!/usr/bin/env python3
"""Direct token comparisons; run with the API stopped and GDEC_KVSNAP=0."""
import argparse
import json
import socket
import time
from pathlib import Path
from qwentok import Tokenizer

p = argparse.ArgumentParser()
p.add_argument('--output', default='logs/ngram-regress.json')
p.add_argument('--quick', action='store_true')
p.add_argument('--case', help='run one named generation case')
p.add_argument('--keep-going', action='store_true', help='finish other cases, but exit nonzero on any comparison failure')
a = p.parse_args()
tk = Tokenizer()
sock = socket.create_connection(('127.0.0.1', 8732), timeout=55)
f = sock.makefile('rb')
results = []
failures = []

def gen(ids, count, drafter, eos=(), cancel=0, suffix=''):
    req = len(results) + 1
    fields = ['GEN', req, count, len(eos), *eos, len(ids), *ids, drafter]
    sock.sendall((' '.join(map(str, fields)) + ' ' + suffix + '\n').encode())
    out, sent = [], False
    while True:
        line = f.readline().decode().strip()
        if not line:
            raise RuntimeError('engine disconnected')
        v = line.split()
        if v[0] == 'T':
            out.append(int(v[2]))
            if cancel and len(out) >= cancel and not sent:
                sock.sendall(f'X {req}\n'.encode())
                sent = True
        elif v[0] == 'D':
            record = dict(req=req, drafter=drafter, prompt=len(ids), tokens=out,
                          done=line, ms=float(v[6]))
            results.append(record)
            Path(a.output).write_text(json.dumps(results, indent=2))
            print(json.dumps({k:v for k,v in record.items() if k!='tokens'}), flush=True)
            return out, v

repeat = tk.encode(' '.join(['alpha beta gamma'] * 80))
code = tk.encode(('def add(a, b):\n    return a + b\n\n' * 10) + 'def add(a, b):\n')
natural = tk.encode('<|im_start|>user\nExplain why rainbows have different colors.\n<|im_end|>\n<|im_start|>assistant\n')
cases = [('repeat', repeat, 16), ('repeat64', repeat, 64), ('code', code, 48), ('natural', natural, 32)]
numbered = tk.encode(''.join(f'{i}. The quick brown fox jumps over the lazy dog and then runs back to the green field beside the river. It pauses beneath the tall old tree to watch the leaves drifting slowly in the warm evening breeze.\n' for i in range(1,9)))
cases.append(('numbered-rejections', numbered, 128))
cases.append(('repeat130', repeat, 130))  # exercises full 65-row verify at max=64
if a.quick:
    cases = cases[:2]
if a.case:
    cases = [c for c in cases if c[0] == a.case]
    assert cases, 'unknown case'
for name, ids, count in cases:
    gold, _ = gen(ids, count, 0)
    spec_gold = None
    if name == 'numbered-rejections':
        # Near-tie prompt: at generation index 90 the serial top-2 logprobs are
        # 0.086 nat apart, and the batched verify path legitimately lands on
        # the other side of the tie (MTP spec diverges from serial at the same
        # index, then matches ngram token-for-token). The ngram machinery is
        # therefore judged against the MTP drafter's output, which shares the
        # trunk's batch-verify math; any ngram-specific state/rollback bug
        # would desync the two. See NGRAM.md.
        spec_gold, _ = gen(ids, count, 1)
    for run in range(2):
        got, _ = gen(ids, count, 'NGRAM')
        if got != gold and not (spec_gold is not None and got == spec_gold):
            failure = (name, run, next((i for i,(x,y) in enumerate(zip(gold,got)) if x!=y), min(len(got),len(gold))))
            failures.append(failure)
            print('FAIL token comparison', failure, flush=True)
            if not a.keep_going:
                raise AssertionError(failure)
        elif spec_gold is not None:
            basis = 'serial' if got == gold else 'mtp-spec'
            print(f'PASS {name} run {run} (matches {basis} gold)', flush=True)
    if not any(f[0] == name for f in failures):
        print(f'PASS {name} repeated token comparison', flush=True)

if not a.quick and not a.case:
    for count in [1, 2, 8, 9, 10, 18, 19]:
        gold, _ = gen(repeat, count, 0)
        got, _ = gen(repeat, count, 'NGRAM')
        assert got == gold, ('length', count)
    gold, _ = gen(repeat, 20, 0)
    got, done = gen(repeat, 20, 'NGRAM', eos=(gold[2],))
    assert got == gold[:2] and done[2] == 'done', ('eos', got, done)
    got, done = gen(repeat, 64, 'NGRAM', cancel=5)
    assert done[2] == 'cancel', done
    # Keep the continuation decisive. Appending a newline to the synthetic
    # repetition induces near-tie divergent prose even with serial live KV
    # versus full prefill (recorded by ngram_cont_probe.py).
    continuation = repeat + got + repeat
    resumed, done = gen(continuation, 16, 'NGRAM')
    assert int(done[10]) > 0, ('no continuation cache', done)
    fresh, _ = gen(continuation, 16, 0)
    assert resumed == fresh, ('resume', resumed, fresh)
    # Sampled ngram must fall back to serial sampling, without using MTP.
    sampled, _ = gen(natural, 16, 0, suffix='SAMPLE 0.8 20 0.95 0 777')
    got, done = gen(natural, 16, 'NGRAM', suffix='SAMPLE 0.8 20 0.95 0 777')
    assert sampled == got and done[7] == '0', ('sampling fallback', sampled, got, done)
if failures:
    print('FAILED comparisons:', failures, flush=True)
    raise SystemExit(1)
print('ALL PASS', flush=True)
