"""A5: paged KV under the full production engine (start.sh env, MTP weights, vision
tower, production maxctx, kvsnap + rckpt on).

Every earlier paged test (a1..a3) decoded serially (drafter 0). Production runs the
chain drafter (ngram + MTP) with spec verify / rollback through prefill_chunk, so this
script covers what those tests did not:
  F-*  fresh prompts x drafters 0 (serial) / 1 (MTP) / 3 (ngram) / 4 (chain), greedy;
       short chat, code, numbered list, 8100-token and 32768-token (two 16K chunks)
       prompts. Each drafter gets a unique prefix, so every request is a real prefill
       (no rckpt / kvsnap hit) with MTP live.
  S-*  sampled requests (SAMPLE 0.8 20 0.95 0 777) with drafters 0 / 1 / 4.
  C-*  one connection, like the API: a multi-turn conversation A (live continuation,
       SNAPS cuts -> mid-prompt checkpoints + copy-on-write), then conversation B,
       then back to A. After the switch a paged engine restores A from its RAM
       checkpoint; an unpaged one cannot (kvsnap / full prefill) -> C-T4/C-T5 are
       compared paged-vs-paged only (p1 == p2) and must report n_cached > 0.
Everything else must equal the unpaged engine bit-exactly: tokens, finish reason and
the spec stats (drafter used, rounds, commits, cached, proposed) -- identical stats
prove the MTP/ngram drafts and the verify batches were identical too.

Usage (repo root, one engine on 127.0.0.1:8732 per tag; tools/a5_verify.sh drives it):
  python3 tools/a5_ab.py run --tag off|p1|p2
  python3 tools/a5_ab.py compare          # also compares logs/a5-<tag>-ngram.json
State: logs/a5-ab.json
"""
import argparse, json, socket, sys
from pathlib import Path
from qwentok import Tokenizer

ROOT = Path(__file__).resolve().parent.parent
STATE = ROOT / 'logs' / 'a5-ab.json'
SAMPLE = 'SAMPLE 0.8 20 0.95 0 777'
PAGED_ONLY = ('C-T4', 'C-T5')   # after the switch back: restore path differs from off
REF = 'off'


class Conn:
    def __init__(self):
        self.sock = socket.create_connection(('127.0.0.1', 8732), timeout=1800)
        self.f = self.sock.makefile('rb')
        self.req = 0

    def line(self):
        v = self.f.readline().decode().split()
        if not v:
            raise RuntimeError('engine disconnected')
        return v

    def gen(self, ids, n, drafter, suffix=''):
        self.req += 1
        fields = ['GEN', self.req, n, 0, len(ids), *ids, drafter]
        self.sock.sendall((' '.join(map(str, fields)) + (' ' + suffix if suffix else '') + '\n').encode())
        out = []
        while True:
            v = self.line()
            if v[0] == 'T':
                out.append(int(v[2]))
            elif v[0] == 'D':
                num = lambda i, t=int: t(v[i]) if len(v) > i else None
                return dict(tokens=out, reason=v[2], prompt=len(ids),
                            stats=[num(7), num(8), num(9), num(10), num(11)],
                            pre_ms=num(5, float), dec_ms=num(6, float), done=' '.join(v))

    def mem(self):
        self.sock.sendall(b'MEM\n')
        v = self.line()
        return dict(device_cur=int(v[2]), device_peak=int(v[3])) if v[0] == 'M' else {}


def prompts(tk):
    o8 = json.loads((ROOT / 'data/qsa-oracle/8192.json').read_text())['prompt_ids']
    o32 = json.loads((ROOT / 'data/qsa-oracle/32768.json').read_text())['prompt_ids']
    chat = lambda u: tk.encode('<|im_start|>user\n' + u + '<|im_end|>\n<|im_start|>assistant\n')
    natural = chat('Explain why rainbows have different colors.')
    code = tk.encode(('def add(a, b):\n    return a + b\n\n' * 10) + 'def add(a, b):\n')
    numbered = tk.encode(''.join(
        f'{i}. The quick brown fox jumps over the lazy dog and then runs back to the green '
        f'field beside the river. It pauses beneath the tall old tree to watch the leaves '
        f'drifting slowly in the warm evening breeze.\n' for i in range(1, 9)))
    return o8, o32, natural, code, numbered


def run(tag):
    tk = Tokenizer()
    o8, o32, natural, code, numbered = prompts(tk)
    c = Conn()
    res = {}

    def do(key, ids, n, drafter, suffix=''):
        r = c.gen(ids, n, drafter, suffix)
        res[key] = r
        print(f'{tag} {key:<16} |p|={len(ids):<6} {r["reason"]:<6} n={len(r["tokens"]):<4} '
              f'stats(drafter,rounds,commit,cached,proposed)={r["stats"]} '
              f'prompt {r["pre_ms"]}ms decode {r["dec_ms"]}ms', flush=True)
        if r['reason'] == 'error':
            print(f'  !! engine returned error: {r["done"]}', flush=True)
        return r

    # ---- F: fresh prompts x drafters (greedy) ----
    fresh = [('nat', natural, 96), ('code', code, 96), ('num', numbered, 128),
             ('L8', o8[:8100], 64), ('L32', o32, 48)]
    for name, ids, n in fresh:
        for d in (0, 1, 3, 4):
            pfx = tk.encode(f'[a5 {name} drafter {d}]\n')
            do(f'F-{name}-d{d}', pfx + ids, n, d)
    # ---- S: sampled ----
    for name, ids in (('nat', natural), ('code', code)):
        for d in (0, 1, 4):
            pfx = tk.encode(f'[a5 sample {name} drafter {d}]\n')
            do(f'S-{name}-d{d}', pfx + ids, 64, d, SAMPLE)
    # ---- C: multi-turn conversation on this one connection ----
    head = tk.encode('<|im_start|>user\n')
    ask = lambda q: tk.encode('<|im_end|>\n<|im_start|>user\n' + q +
                              '<|im_end|>\n<|im_start|>assistant\n')
    doc = o8[:6000]
    t1 = head + doc + tk.encode('\n\nSummarize the text above.<|im_end|>\n<|im_start|>assistant\n')
    r1 = do('C-T1', t1, 48, 4, f'SNAPS 1 {len(t1) - 4}')
    t2 = t1 + r1['tokens'] + ask('List three keywords for the text.')
    r2 = do('C-T2', t2, 48, 4, f'SNAPS 1 {len(t2) - 4}')          # live continuation, chain
    t3 = t2 + r2['tokens'] + ask('Now answer in one short sentence.')
    r3 = do('C-T3', t3, 48, 1, f'SNAPS 1 {len(t3) - 4}')          # live continuation, MTP
    tb = head + o32[20000:28000] + tk.encode('\n\nWhat is this about?<|im_end|>\n<|im_start|>assistant\n')
    do('C-B', tb, 32, 4)                                           # other conversation
    t4 = t3 + r3['tokens'] + ask('Thanks. One more keyword?')
    r4 = do('C-T4', t4, 48, 4, f'SNAPS 1 {len(t4) - 4}')          # back to A (restore)
    t5 = t4 + r4['tokens'] + ask('And a title?')
    do('C-T5', t5, 32, 4)                                          # live again after restore
    mem = c.mem()
    st = json.loads(STATE.read_text()) if STATE.exists() else {}
    st[tag] = dict(res=res, mem=mem)
    STATE.parent.mkdir(exist_ok=True)
    STATE.write_text(json.dumps(st))
    if mem:
        print(f'{tag} MEM device_cur {mem["device_cur"] / 2**30:.2f} GiB, '
              f'device_peak {mem["device_peak"] / 2**30:.2f} GiB', flush=True)
    bad = [k for k, r in res.items() if r['reason'] == 'error']
    if bad:
        print(f'{tag}: engine errors on {bad}', flush=True)
        sys.exit(1)


def first_diff(a, b):
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))


def same(a, b):
    return a is not None and b is not None and a['tokens'] == b['tokens'] and \
        a['reason'] == b['reason'] and a['stats'] == b['stats']


def ngram_records(tag):
    p = ROOT / 'logs' / f'a5-{tag}-ngram.json'
    if not p.exists():
        return None
    out = []
    for r in json.loads(p.read_text()):
        d = r['done'].split()
        out.append((r['drafter'], r['prompt'], r['tokens'], d[:5] + d[7:]))  # drop ms
    return out


def ngram_diff(a, b):
    """First differing record index, or None. ngram_regress's cancel case (X after 5
    tokens) is timing dependent: how many tokens arrive before the cancel lands varies
    run to run. That record is skipped; the records after it build on the cancelled
    output, so they are compared only when the cancelled tokens happen to agree (their
    own internal check, resumed == fresh, is covered by ngram_regress's exit code)."""
    if len(a) != len(b):
        return min(len(a), len(b))
    skip_rest = False
    for i, (x, y) in enumerate(zip(a, b)):
        if x[3][2] == 'cancel' or y[3][2] == 'cancel':
            if x[3][2] != y[3][2] or x[:2] != y[:2]:
                return i
            skip_rest = x[2] != y[2]
            continue
        if skip_rest:
            continue
        if x != y:
            return i
    return None


def compare():
    st = json.loads(STATE.read_text()) if STATE.exists() else {}
    tags = [t for t in ('off', 'p1', 'p2') if t in st]
    bad = 0
    if REF not in st or len(tags) < 2:
        print(f'需要 off 和至少一个分页引擎的结果，现有: {tags}')
        sys.exit(1)
    ref = st[REF]['res']
    keys = list(ref)
    paged = [t for t in tags if t != REF]
    print(f'{"用例":<16}' + ''.join(f'{t:>10}' for t in paged) + '   说明')
    for k in keys:
        row, note = [], ''
        for t in paged:
            r = st[t]['res'].get(k)
            if k in PAGED_ONLY:
                other = st[paged[0]]['res'].get(k)
                ok = r is not None and r['reason'] != 'error' and (r['stats'][3] or 0) > 0 \
                    and same(r, other)
                note = f'分页之间比较（{paged[0]}），且必须命中缓存 n_cached>0'
            else:
                ok = same(r, ref.get(k))
                if not ok and r is not None and ref.get(k) is not None:
                    note = (f'{t}: 第 {first_diff(r["tokens"], ref[k]["tokens"])} 个 token 起不同；'
                            f'stats {r["stats"]} vs off {ref[k]["stats"]}')
            bad += not ok
            row.append('MATCH' if ok else 'DIFF')
        print(f'{k:<16}' + ''.join(f'{c:>10}' for c in row) + (f'   {note}' if note else ''))
    # spec decoding must actually have happened (otherwise the test proves little)
    used = {r['stats'][0] for r in ref.values() if r['stats'][0] and r['stats'][1]}
    print(f'\noff 实际用到的投机 drafter（1=MTP 3=ngram 4=chain）: {sorted(used)}')
    if not {1, 3, 4} <= used:
        print('  !! 缺少 MTP/ngram/chain 投机轮次（MTP_FILE 没加载？），这项检查不成立')
        bad += 1
    # ngram_regress outputs
    ng = {t: ngram_records(t) for t in tags}
    if ng[REF] is None:
        print('ngram_regress: 没有 off 的结果'); bad += 1
    else:
        for t in paged:
            if ng[t] is None:
                print(f'ngram_regress {t}: 没有结果'); bad += 1
            elif (i := ngram_diff(ng[t], ng[REF])) is not None:
                print(f'ngram_regress {t}: DIFF（第 {i + 1} 个请求起不同，共 {len(ng[t])} vs {len(ng[REF])}）')
                bad += 1
            else:
                print(f'ngram_regress {t}: MATCH（{len(ng[t])} 个请求，tokens + D 行字段逐一相同；'
                      f'取消用例与时序有关，不比较）')
    # perf / memory (informational)
    print('\n性能（仅供参考）：prompt ms / decode ms')
    for k in ('F-L8-d0', 'F-L32-d0', 'F-L32-d4', 'C-T2', 'C-T4'):
        cells = []
        for t in tags:
            r = st[t]['res'].get(k)
            if r:
                cells.append(f'{t} {r["pre_ms"]:.0f}/{r["dec_ms"]:.0f}')
        print(f'  {k:<10} ' + '   '.join(cells))
    for k in ('F-L32-d0', 'F-L8-d0'):
        a, b = ref.get(k), st.get('p1', {}).get('res', {}).get(k)
        if a and b and a['pre_ms']:
            print(f'  {k} prefill p1/off = {b["pre_ms"] / a["pre_ms"]:.3f}')
    for t in tags:
        m = st[t].get('mem') or {}
        if m:
            print(f'  MEM {t:<4} device_cur {m["device_cur"] / 2**30:.2f} GiB  '
                  f'device_peak {m["device_peak"] / 2**30:.2f} GiB')
    print('a5_ab', 'ALL MATCH' if bad == 0 else f'{bad} DIFF', flush=True)
    sys.exit(1 if bad else 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['run', 'compare', 'reset'])
    ap.add_argument('--tag', default='off')
    a = ap.parse_args()
    if a.cmd == 'reset':
        STATE.unlink(missing_ok=True)
        for t in ('off', 'p1', 'p2'):
            (ROOT / 'logs' / f'a5-{t}-ngram.json').unlink(missing_ok=True)
    elif a.cmd == 'compare':
        compare()
    else:
        run(a.tag)


if __name__ == '__main__':
    main()
