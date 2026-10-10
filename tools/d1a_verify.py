#!/usr/bin/env python3
"""D1a 验收的客户端（由 tools/d1a_verify.sh 调用，引擎已在 127.0.0.1:8732 以 PARALLEL=2 起好）。

  run  --tag T   依次：A 单独 decode（参照）→ B 单独 32K prefill（PP）→ A decode 期间提交 B（停顿）
                 结果写 logs/d1a/T.json
  gate --a 16K --b 8K   读两份结果，打印对比，最后一行 PASS/FAIL

A = 短 prompt、chain drafter、greedy、A_TOKENS 个 token；B = 32K prompt（qsa-oracle），只要首 token。
B 两次用不同的首 token（引擎开 QWENOX_RCKPT=0），保证每次都从头 prefill；D 行 cached 必须为 0。
停顿只统计 B 提交到 B 首 token 这段窗口内 A 的 token 间隔。
"""
import argparse
import json
import socket
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from qwentok import Tokenizer
from a5_ab import prompts

PORT = 8732
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'logs/d1a'


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

    def gen(self, ids, n, drafter, on_tok=None):
        self.req += 1
        fields = ['GEN', self.req, n, 0, len(ids), *ids, drafter]
        self.sock.sendall((' '.join(map(str, fields)) + '\n').encode())
        out = []
        while True:
            v = self.f.readline().decode().split()
            if not v:
                raise RuntimeError('engine disconnected')
            if v[0] == 'T':
                out.append(int(v[2]))
                if on_tok:
                    on_tok(len(out))
            elif v[0] == 'D':
                num = lambda i, t=int: t(v[i]) if len(v) > i else None
                return dict(tokens=out, reason=v[2], pre_ms=num(5, float), dec_ms=num(6, float),
                            cached=num(10), done=' '.join(v))


def run_a(ids, n, times, res, err):
    try:
        c = Conn()
        res.update(c.gen(ids, n, 4, on_tok=lambda i: times.append(time.monotonic())))
        c.close()
    except Exception as e:  # noqa: BLE001
        err.append(repr(e))


def cmd_run(args):
    tk = Tokenizer()
    _, o32, natural, _, _ = prompts(tk)
    b_solo = [220] + o32[1:]   # 两个 B 首 token 不同 → 互不命中前缀
    b_conc = [198] + o32[1:]
    r = dict(tag=args.tag, a_tokens=args.a_tokens, b_len=len(o32))

    # 1) A 单独：参照 token 与单独时的 token 间隔
    times, a1, err = [], {}, []
    run_a(natural, args.a_tokens, times, a1, err)
    if err:
        print('A 单独异常:', err[0]); sys.exit(1)
    g = sorted(b - a for a, b in zip(times, times[1:]))
    r['a_solo'] = dict(tokens=a1['tokens'], reason=a1['reason'],
                       tps=len(times) / (times[-1] - times[0]) if len(times) > 1 else 0,
                       max_gap=g[-1] if g else 0)
    print(f'[{args.tag}] A 单独: n={len(a1["tokens"])} {a1["reason"]} '
          f'{r["a_solo"]["tps"]:.1f} tok/s  最大间隔 {r["a_solo"]["max_gap"] * 1000:.0f} ms', flush=True)

    # 2) B 单独：32K prefill 耗时（PP）
    c = Conn()
    t0 = time.monotonic()
    b1 = c.gen(b_solo, 1, 0)
    c.close()
    r['b_solo'] = dict(pre_ms=b1['pre_ms'], cached=b1['cached'], wall=time.monotonic() - t0,
                       reason=b1['reason'])
    print(f'[{args.tag}] B 单独: prefill {b1["pre_ms"]:.0f} ms（{len(o32) / b1["pre_ms"] * 1000:.0f} tok/s）'
          f' cached={b1["cached"]} {b1["reason"]}', flush=True)

    # 3) 并发：A decode 3 s 后提交 B，量 B prefill 窗口内 A 的最大间隔
    times, a2, err = [], {}, []
    ta = threading.Thread(target=run_a, args=(natural, args.a_tokens, times, a2, err), daemon=True)
    ta.start()
    time.sleep(3)
    c = Conn()
    tb0 = time.monotonic()
    tb1 = []
    b2 = c.gen(b_conc, 1, 0, on_tok=lambda i: tb1.append(time.monotonic()))
    c.close()
    ta.join(timeout=900)
    if err or ta.is_alive():
        print('A 并发异常:', err[0] if err else '超时'); sys.exit(1)
    tb1 = tb1[0] if tb1 else time.monotonic()
    # 窗口 [tb0, tb1]：包含跨过 tb0 的那一个间隔
    win = [(b - a, a - tb0) for a, b in zip(times, times[1:]) if b > tb0 and a < tb1]
    win.sort(reverse=True)
    r['conc'] = dict(tokens=a2['tokens'], reason=a2['reason'], b_pre_ms=b2['pre_ms'],
                     b_cached=b2['cached'], b_ttft=tb1 - tb0, b_reason=b2['reason'],
                     a_outlived_b=bool(times) and times[-1] > tb1,
                     a_tokens_in_window=sum(1 for t in times if tb0 <= t <= tb1),
                     max_gap=win[0][0] if win else -1,
                     top_gaps=[(round(d, 3), round(at, 2)) for d, at in win[:6]],
                     gaps_over_1s=sum(1 for d, _ in win if d > 1.0))
    cc = r['conc']
    print(f'[{args.tag}] 并发: B TTFT {cc["b_ttft"]:.1f} s（prefill {b2["pre_ms"]:.0f} ms, cached={b2["cached"]}）；'
          f'窗口内 A {cc["a_tokens_in_window"]} token，最大间隔 {cc["max_gap"] * 1000:.0f} ms，'
          f'>1 s 的间隔 {cc["gaps_over_1s"]} 个', flush=True)
    print(f'[{args.tag}]   最大几个间隔（秒, 距 B 提交的秒数）: {cc["top_gaps"]}', flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f'{args.tag}.json').write_text(json.dumps(r))


def cmd_gate(args):
    A = json.loads((OUT / f'{args.a}.json').read_text())
    B = json.loads((OUT / f'{args.b}.json').read_text())
    res = []

    def note(name, ok, detail=''):
        res.append(ok)
        print(f'  {"PASS" if ok else "FAIL"}  {name}{"：" + detail if detail else ""}')

    print(f'==== D1a 验收（{args.a} = 旧分段，{args.b} = 并发分段）====')
    for R in (A, B):
        t = R['tag']
        note(f'{t} 测量有效（B 从头 prefill，A 覆盖整个 B prefill）',
             R['b_solo']['cached'] == 0 and R['conc']['b_cached'] == 0 and R['conc']['a_outlived_b']
             and R['a_solo']['reason'] == 'length' and R['conc']['reason'] == 'length',
             f'cached {R["b_solo"]["cached"]}/{R["conc"]["b_cached"]}，A 活过 B={R["conc"]["a_outlived_b"]}')
        same = R['conc']['tokens'] == R['a_solo']['tokens']
        note(f'{t} A 并发 == A 单独（逐 token）', same)
    note('两个引擎的 A 单独输出相同（A prompt < 8K，分段不影响）',
         A['a_solo']['tokens'] == B['a_solo']['tokens'])
    ga, gb = A['conc']['max_gap'], B['conc']['max_gap']
    note(f'{args.b} 最大停顿 < {args.stall_max:.2f} s', gb < args.stall_max,
         f'{args.a} {ga * 1000:.0f} ms → {args.b} {gb * 1000:.0f} ms（{gb / ga:.2f}×）')
    pa, pb = A['b_solo']['pre_ms'], B['b_solo']['pre_ms']
    cost = (pb / pa - 1) * 100
    note(f'32K 单独 prefill 变慢 ≤ {args.pp_max:.0f}%', cost <= args.pp_max,
         f'{pa:.0f} → {pb:.0f} ms（{cost:+.1f}%）')
    print(f'  信息：并发时 B TTFT {A["conc"]["b_ttft"]:.1f} → {B["conc"]["b_ttft"]:.1f} s；'
          f'A 单独 {A["a_solo"]["tps"]:.1f} / {B["a_solo"]["tps"]:.1f} tok/s')
    ok = all(res)
    print('D1A VERIFY: ' + ('PASS' if ok else 'FAIL'))
    sys.exit(0 if ok else 1)


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest='cmd', required=True)
    p = sp.add_parser('run'); p.add_argument('--tag', required=True)
    p.add_argument('--a-tokens', type=int, default=1536)
    p = sp.add_parser('gate'); p.add_argument('--a', required=True); p.add_argument('--b', required=True)
    p.add_argument('--stall-max', type=float, default=1.0)
    p.add_argument('--pp-max', type=float, default=10.0)
    a = ap.parse_args()
    cmd_run(a) if a.cmd == 'run' else cmd_gate(a)


if __name__ == '__main__':
    main()
