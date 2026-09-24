#!/usr/bin/env python3
"""Concurrent serving (GDEC_PARALLEL) checks against the engine on 127.0.0.1:8732.
tools/conc_verify.sh drives it; every subcommand exits 0 on PASS.

  seq  --tag T   run FLOWS one after another on ONE connection  -> logs/conc/T.json
  conc --tag T   --slots N client threads, each flow on its own connection -> logs/conc/T.json
  cmp  --a A --b B   tokens + finish reason + spec stats must match bit-exactly
  ctl  --slots N     INFO kv_slots, PING latency under load, cancel (running/queued)
  ovf                shared-pool overflow (engine: N=2, small --maxctx, rckpt off):
                     the LATER request fails, the earlier one is untouched

A flow is a list of turns on one connection; conversation turns extend the
previous turn's prompt + output, so they exercise live-state continuation
(n_cached > 0) while other sequences interleave on the GPU. Each flow has a
unique prefix, so no request can hit another's cache.
"""
import argparse, json, socket, sys, threading, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from qwentok import Tokenizer
from a5_ab import prompts, first_diff, SAMPLE

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'logs' / 'conc'
PORT = 8732


class Conn:
    def __init__(self, timeout=1800):
        self.sock = socket.create_connection(('127.0.0.1', PORT), timeout=timeout)
        self.f = self.sock.makefile('rb')
        self.req = 0

    def close(self):
        try:
            self.sock.close()
        except OSError:
            pass

    def line(self):
        v = self.f.readline().decode().split()
        if not v:
            raise RuntimeError('engine disconnected')
        return v

    def send_gen(self, ids, n, drafter, suffix=''):
        self.req += 1
        fields = ['GEN', self.req, n, 0, len(ids), *ids, drafter]
        self.sock.sendall((' '.join(map(str, fields)) + (' ' + suffix if suffix else '') + '\n').encode())
        return self.req

    def read_gen(self, on_tok=None):
        out = []
        while True:
            v = self.line()
            if v[0] == 'T':
                out.append(int(v[2]))
                if on_tok:
                    on_tok(len(out))
            elif v[0] == 'D':
                num = lambda i, t=int: t(v[i]) if len(v) > i else None
                return dict(tokens=out, reason=v[2], stats=[num(7), num(8), num(9), num(10), num(11)],
                            done=' '.join(v))

    def gen(self, ids, n, drafter, suffix=''):
        self.send_gen(ids, n, drafter, suffix)
        return self.read_gen()

    def cmd(self, verb):
        self.sock.sendall((verb + '\n').encode())
        return self.line()


def flows(tk):
    o8, o32, natural, code, numbered = prompts(tk)
    tag = lambda s: tk.encode(f'[conc {s}]\n')
    F = [
        [('nat-d0', tag('nat-d0') + natural, 96, 0, '')],
        [('code-d3', tag('code-d3') + code, 96, 3, '')],
        [('num-d4', tag('num-d4') + numbered, 128, 4, '')],
        [('nat-d1', tag('nat-d1') + natural, 96, 1, '')],
        [('S-code-d4', tag('S-code-d4') + code, 64, 4, SAMPLE)],
        [('S-nat-d1', tag('S-nat-d1') + natural, 64, 1, SAMPLE)],
        [('L3k-d0', tag('L3k-d0') + o8[:3000], 64, 0, '')],
    ]
    head = tk.encode('<|im_start|>user\n')
    t1 = tag('conv') + head + o8[3000:6000] + tk.encode(
        '\n\nSummarize the text above.<|im_end|>\n<|im_start|>assistant\n')
    asks = ['List three keywords for the text.', 'Now answer in one short sentence.']
    F.append([('conv-T1', t1, 64, 4, ''),
              ('conv-T2', asks[0], 48, 4, ''),
              ('conv-T3', asks[1], 48, 1, '')])
    return F


def run_flow(tk, flow, res, conn=None):
    c = conn or Conn()
    ids = None
    for key, p, n, d, sfx in flow:
        if isinstance(p, str):  # conversation turn: previous prompt + output + question
            prev = res[prev_key]
            p = ids + prev['tokens'] + tk.encode(
                '<|im_end|>\n<|im_start|>user\n' + p + '<|im_end|>\n<|im_start|>assistant\n')
        ids = p
        t0 = time.time()
        r = c.gen(p, n, d, sfx)
        r['wall'] = time.time() - t0
        r['prompt'] = len(p)
        res[key] = r
        prev_key = key
        print(f'  {key:10s} |p|={len(p):5d} d={d} n={len(r["tokens"]):3d} {r["reason"]:6s} '
              f'stats={r["stats"]} {r["wall"]:.1f}s', flush=True)
    if conn is None:
        c.close()


def cmd_run(tag, concurrent, par=4):
    tk = Tokenizer()
    F = flows(tk)
    res = {}
    t0 = time.time()
    if concurrent:
        # `par` client threads (= engine slots) pull flows from a queue, so a
        # conversation's slot is never the LRU pick of an oversubscribed queue
        # (that is legal behaviour, but it drops live state -> n_cached differs)
        errs = []
        todo = [F[-1]] + F[:-1]  # conversation first
        lock = threading.Lock()

        def worker():
            while True:
                with lock:
                    if not todo:
                        return
                    fl = todo.pop(0)
                try:
                    run_flow(tk, fl, res)
                except Exception as e:  # noqa: BLE001
                    errs.append(f'{fl[0][0]}: {e}')
        th = [threading.Thread(target=worker) for _ in range(par)]
        for t in th:
            t.start()
        for t in th:
            t.join()
        if errs:
            print('FAIL:', errs)
            return False
    else:
        c = Conn()
        for fl in F:
            run_flow(tk, fl, res, c)
        c.close()
    wall = time.time() - t0
    ntok = sum(len(r['tokens']) for r in res.values())
    print(f'{tag}: {len(res)} requests, {ntok} tokens, wall {wall:.1f}s '
          f'({ntok / wall:.1f} tok/s aggregate)', flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f'{tag}.json').write_text(json.dumps(dict(res=res, wall=wall)))
    bad = [k for k, r in res.items() if r['reason'] not in ('length', 'done')]
    if bad:
        print('FAIL: bad finish', {k: res[k]['done'] for k in bad})
    return not bad


def cmd_cmp(a, b):
    A = json.loads((OUT / f'{a}.json').read_text())['res']
    B = json.loads((OUT / f'{b}.json').read_text())['res']
    ok = True
    for k in A:
        ra, rb = A[k], B.get(k)
        if rb and ra['tokens'] == rb['tokens'] and ra['reason'] == rb['reason'] and ra['stats'] == rb['stats']:
            continue
        ok = False
        if not rb:
            print(f'  {k}: missing in {b}')
        else:
            print(f'  {k}: DIFF@{first_diff(ra["tokens"], rb["tokens"])} '
                  f'{ra["reason"]} {ra["stats"]} vs {rb["reason"]} {rb["stats"]}')
    conv = A.get('conv-T2', {}).get('stats', [0, 0, 0, 0])[3]
    print(f'{a} vs {b}: {"MATCH" if ok else "DIFF"} ({len(A)} requests; conv-T2 cached {conv})')
    return ok


def cmd_ctl(slots):
    tk = Tokenizer()
    o8 = prompts(tk)[0]
    ok = True
    c = Conn()
    v = c.cmd('INFO')
    got = int(v[10]) if len(v) > 10 else -1
    print(f'  INFO kv_slots = {got} (want {slots})')
    ok &= got == slots

    # busy load: `slots` long serial decodes, each on its own connection
    stop = {}
    busy = []

    def hog(i):
        h = Conn()
        busy.append(h)
        h.send_gen(tk.encode(f'[conc hog {i}]\n') + o8[:1500], 400, 0)
        stop[i] = h.read_gen()
        h.close()
    th = [threading.Thread(target=hog, args=(i,)) for i in range(slots)]
    for t in th:
        t.start()
    time.sleep(8)  # all slots prefilled and decoding

    lat = []
    for _ in range(5):
        t0 = time.time()
        r = c.cmd('PING')
        lat.append(time.time() - t0)
        ok &= r[0] == 'PONG'
    print(f'  PING under load: max {max(lat) * 1000:.0f} ms')
    ok &= max(lat) < 0.5

    # a request queued behind full slots, cancelled before it gets one
    q = Conn()
    t0 = time.time()
    rq = q.send_gen(tk.encode('[conc queued]\n') + o8[:500], 64, 0)
    time.sleep(1.0)
    q.sock.sendall(f'X {rq}\n'.encode())
    r = q.read_gen()
    dt = time.time() - t0
    qdone = len(stop)
    print(f'  cancel while queued: {r["done"]!r} after {dt:.1f}s ({qdone}/{slots} hogs done)')
    ok &= r['reason'] == 'cancel' and not r['tokens'] and qdone == 0
    q.close()
    for t in th:
        t.join()
    hogs_ok = all(r['reason'] == 'length' and len(r['tokens']) == 400 for r in stop.values())
    print(f'  hogs finished: {[len(r["tokens"]) for r in stop.values()]} {"OK" if hogs_ok else "BAD"}')
    ok &= hogs_ok and len(stop) == slots

    # cancel a running request mid-decode
    seen = []
    rc = c.send_gen(tk.encode('[conc cancel-run]\n') + o8[:500], 2000, 0)

    def on_tok(n):
        if n == 5:
            seen.append(time.time())
            c.sock.sendall(f'X {rc}\n'.encode())
    r = c.read_gen(on_tok)
    dt = time.time() - seen[0] if seen else -1
    print(f'  cancel while running: {r["reason"]} after {len(r["tokens"])} tokens, {dt * 1000:.0f} ms')
    ok &= r['reason'] == 'cancel' and len(r['tokens']) < 100

    # one GEN per connection: a second GEN while the first is running is refused
    rp = c.send_gen(tk.encode('[conc pipe]\n') + o8[:300], 200, 0)
    c.send_gen(tk.encode('[conc pipe2]\n') + o8[:300], 8, 0)
    got = {}
    while len(got) < 2:
        v = c.line()
        if v[0] == 'D':
            got[int(v[1])] = v[2]
    print(f'  pipelined GEN: first={got.get(rp)} second={got.get(rp + 1)}')
    ok &= got.get(rp) == 'length' and got.get(rp + 1) == 'error'

    # a disconnect mid-request frees the slot (the next request still runs)
    d = Conn()
    d.send_gen(tk.encode('[conc drop]\n') + o8[:300], 2000, 0)
    d.line()
    d.close()
    time.sleep(1)
    r = c.gen(tk.encode('[conc after-drop]\n') + o8[:300], 16, 0)
    print(f'  after client drop: {r["reason"]} n={len(r["tokens"])}')
    ok &= r['reason'] == 'length'
    c.close()
    print(f'ctl: {"PASS" if ok else "FAIL"}')
    return ok


def cmd_ovf():
    """Engine: GDEC_PARALLEL=2, --maxctx 16384 (pool = 16384 tokens), rckpt off."""
    tk = Tokenizer()
    o8, o32 = prompts(tk)[:2]
    ok = True
    A = tk.encode('[conc ovf A]\n') + o8[:7000]
    B = tk.encode('[conc ovf B]\n') + o8[:7000]
    NA = 3000

    # O1: A (earlier) grows while B (later) holds pages -> B is aborted mid-decode
    ref = Conn().gen(A, NA, 0)
    print(f'  O1 A solo: {ref["reason"]} n={len(ref["tokens"])}')
    ca, cb = Conn(), Conn()
    res = {}
    started = threading.Event()

    def run_a():
        ca.send_gen(A, NA, 0)
        res['A'] = ca.read_gen(lambda n: n == 20 and started.set())
    ta = threading.Thread(target=run_a)
    ta.start()
    started.wait(600)
    res['B'] = cb.gen(B, 3000, 0)
    ta.join()
    a, b = res['A'], res['B']
    same = a['tokens'] == ref['tokens'] and a['reason'] == ref['reason']
    print(f'  O1 A concurrent: {a["reason"]} n={len(a["tokens"])} '
          f'{"== solo" if same else "DIFF@%d" % first_diff(a["tokens"], ref["tokens"])}')
    print(f'  O1 B (later): {b["reason"]} after {len(b["tokens"])} tokens')
    ok &= same and b['reason'] == 'error' and 0 < len(b['tokens']) < 3000

    # O2: B (later) cannot even prefill next to a big A -> B fails, A untouched
    A2 = tk.encode('[conc ovf A2]\n') + o32[:12000]
    B2 = tk.encode('[conc ovf B2]\n') + o8[:6000]
    ref2 = Conn().gen(A2, 1500, 0)
    started.clear()

    def run_a2():
        ca.send_gen(A2, 1500, 0)
        res['A2'] = ca.read_gen(lambda n: n == 5 and started.set())
    ta = threading.Thread(target=run_a2)
    ta.start()
    started.wait(600)
    res['B2'] = cb.gen(B2, 64, 0)
    ta.join()
    a, b = res['A2'], res['B2']
    same = a['tokens'] == ref2['tokens'] and a['reason'] == ref2['reason']
    print(f'  O2 A concurrent: {a["reason"]} n={len(a["tokens"])} '
          f'{"== solo" if same else "DIFF@%d" % first_diff(a["tokens"], ref2["tokens"])}')
    print(f'  O2 B (later): {b["reason"]} n={len(b["tokens"])}')
    ok &= same and b['reason'] == 'error' and not b['tokens']

    # the pool is whole again: B alone now succeeds on the same connection
    r = cb.gen(B2, 64, 0)
    print(f'  O3 B retried alone: {r["reason"]} n={len(r["tokens"])}')
    ok &= r['reason'] == 'length' and len(r['tokens']) == 64
    ca.close()
    cb.close()
    OUT.mkdir(parents=True, exist_ok=True)  # solo references, for cross-run debugging
    (OUT / 'ovf.json').write_text(json.dumps(dict(ref=ref['tokens'], ref2=ref2['tokens'],
                                                  A=res['A']['tokens'], A2=res['A2']['tokens'])))
    print(f'ovf: {"PASS" if ok else "FAIL"}')
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['seq', 'conc', 'cmp', 'ctl', 'ovf'])
    ap.add_argument('--tag')
    ap.add_argument('--a')
    ap.add_argument('--b')
    ap.add_argument('--slots', type=int, default=4)
    x = ap.parse_args()
    if x.cmd in ('seq', 'conc'):
        ok = cmd_run(x.tag, x.cmd == 'conc', x.slots)
    elif x.cmd == 'cmp':
        ok = cmd_cmp(x.a, x.b)
    elif x.cmd == 'ctl':
        ok = cmd_ctl(x.slots)
    else:
        ok = cmd_ovf()
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
