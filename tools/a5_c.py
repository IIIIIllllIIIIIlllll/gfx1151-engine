#!/usr/bin/env python3
"""A5 C-scenario isolation: run only the multi-turn conversation T1 -> T2 -> T3
of tools/a5_ab.py (same prompts) against the engine on :8732, with configurable
drafters / SNAPS, and store tokens + D-line stats for off-vs-paged comparison.

  python3 tools/a5_c.py run --tag off-base --drafters 4,4,1 --snaps 1
  python3 tools/a5_c.py compare --a off-base --b p1-base
"""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from qwentok import Tokenizer
from a5_ab import Conn, first_diff

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'logs' / 'a5c'


def pre_cases(tk):
    """The F / S requests a5_ab runs before C, in the same order and form."""
    from a5_ab import prompts, SAMPLE
    o8, o32, natural, code, numbered = prompts(tk)
    cases = []
    for name, ids, n in [('nat', natural, 96), ('code', code, 96), ('num', numbered, 128),
                         ('L8', o8[:8100], 64), ('L32', o32, 48)]:
        for d in (0, 1, 3, 4):
            cases.append((f'F-{name}-d{d}', tk.encode(f'[a5 {name} drafter {d}]\n') + ids, n, d, ''))
    for name, ids in (('nat', natural), ('code', code)):
        for d in (0, 1, 4):
            cases.append((f'S-{name}-d{d}', tk.encode(f'[a5 sample {name} drafter {d}]\n') + ids,
                          64, d, SAMPLE))
    return cases


def run(tag, drafters, snaps, n, pre=''):
    tk = Tokenizer()
    o8 = json.loads((ROOT / 'data/qsa-oracle/8192.json').read_text())['prompt_ids']
    head = tk.encode('<|im_start|>user\n')
    ask = lambda q: tk.encode('<|im_end|>\n<|im_start|>user\n' + q +
                              '<|im_end|>\n<|im_start|>assistant\n')
    t = head + o8[:6000] + tk.encode('\n\nSummarize the text above.<|im_end|>\n<|im_start|>assistant\n')
    qs = [None, 'List three keywords for the text.', 'Now answer in one short sentence.']
    c = Conn()
    res = {}
    if pre:
        # --pre all | F | S | comma list of case names / prefixes (e.g. F-L32,S-code-d4)
        want = pre.split(',')
        for key, ids, pn, d, sfx in pre_cases(tk):
            if pre == 'all' or any(key.startswith(w) for w in want):
                r = c.gen(ids, pn, d, sfx)
                print(f'{tag} pre {key} n={len(r["tokens"])} stats={r["stats"]}', flush=True)
    for i, d in enumerate(drafters):
        if i:
            t = t + res[f'T{i}']['tokens'] + ask(qs[i] if i < len(qs) else f'Question {i}?')
        sfx = f'SNAPS 1 {len(t) - 4}' if snaps else ''
        r = c.gen(t, n, d, sfx)
        res[f'T{i + 1}'] = r
        print(f'{tag} T{i + 1} |p|={len(t)} d={d} n={len(r["tokens"])} stats={r["stats"]}', flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f'{tag}.json').write_text(json.dumps(res))


def hist(tag, drafters, n):
    """History dependence: fresh T1, then an unrelated 32K prompt, then T1 again.
    Run with rckpt/kvsnap off so the second T1 is a fresh prefill too."""
    tk = Tokenizer()
    o8 = json.loads((ROOT / 'data/qsa-oracle/8192.json').read_text())['prompt_ids']
    o32 = json.loads((ROOT / 'data/qsa-oracle/32768.json').read_text())['prompt_ids']
    head = tk.encode('<|im_start|>user\n')
    t1 = head + o8[:6000] + tk.encode('\n\nSummarize the text above.<|im_end|>\n<|im_start|>assistant\n')
    pol = head + o32 + tk.encode('\n\nWhat is this about?<|im_end|>\n<|im_start|>assistant\n')
    c = Conn()
    bad = 0
    for d in drafters:
        a = c.gen(t1, n, d)
        p = c.gen(pol, 16, d)
        b = c.gen(t1, n, d)
        same = a['tokens'] == b['tokens'] and a['stats'] == b['stats']
        bad += not same
        print(f'{tag} drafter {d}: T1 before vs after 32K polluter: '
              f'{"SAME" if same else "DIFF@%d" % first_diff(a["tokens"], b["tokens"])} '
              f'{a["stats"]} vs {b["stats"]} (cached {a["stats"][3]},{p["stats"][3]},{b["stats"][3]})',
              flush=True)
    return bad == 0


def compare(a, b):
    A = json.loads((OUT / f'{a}.json').read_text())
    B = json.loads((OUT / f'{b}.json').read_text())
    ok = True
    parts = []
    for k in sorted(A):
        ta, tb = A[k]['tokens'], B.get(k, {}).get('tokens', [])
        if ta == tb and A[k]['stats'] == B[k]['stats']:
            parts.append(f'{k}=MATCH')
        else:
            ok = False
            parts.append(f'{k}=DIFF@{first_diff(ta, tb)} {A[k]["stats"]} vs {B[k]["stats"]}')
            break
    print(' '.join(parts))
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['run', 'compare', 'hist'])
    ap.add_argument('--tag')
    ap.add_argument('--drafters', default='4,4,1')
    ap.add_argument('--snaps', type=int, default=1)
    ap.add_argument('--n', type=int, default=48)
    ap.add_argument('--pre', default='')
    ap.add_argument('--a')
    ap.add_argument('--b')
    x = ap.parse_args()
    if x.cmd == 'hist':
        sys.exit(0 if hist(x.tag, [int(v) for v in x.drafters.split(',')], x.n) else 1)
    elif x.cmd == 'run':
        run(x.tag, [int(v) for v in x.drafters.split(',')], x.snaps, x.n, x.pre)
    else:
        sys.exit(0 if compare(x.a, x.b) else 1)


if __name__ == '__main__':
    main()
