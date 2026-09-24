#!/usr/bin/env python3
"""A4: per-page content-addressed SSD KV cache (kvsnap) — restores across restarts.

Usage (repo root, one engine on 127.0.0.1:8732 per step, GDEC_KVSNAP_DIR=DIR,
GDEC_KVSNAP_MIN=1024 — tools/a4_verify.sh drives all of this):
  python3 tools/a4_ab.py save   --dir DIR        # E1 (empty dir, rckpt on): gold runs
  python3 tools/a4_ab.py run    --tag T [--only a2,b2,c2]   # fresh engine: restores
  python3 tools/a4_ab.py compare --tag T [--only ...] [--ref gold|TAG] [--miss|--hit]
  python3 tools/a4_ab.py predict --dir DIR --keep 2   # LRU cap keeping the 2 newest
  python3 tools/a4_ab.py check-lru --dir DIR          # disk == predicted survivors
  python3 tools/a4_ab.py check-disk --dir DIR         # KVSYNC counts/bytes == files
  python3 tools/a4_ab.py corrupt --dir DIR            # flip a byte in every page file
  python3 tools/a4_ab.py sync                         # KVSYNC

E1 scenario (drafter 1 = MTP speculation, greedy):
  conn 1: a1 A(8100, SNAPS 4096) 16 tok -> a2 QA = A+out+t2 64 tok (live cont)
  conn 2: b1 B = A[:6000]+other text 16 tok -> b2 QB = B+out+t3 64 tok (live cont)
          B shares A's first 23 pages: its end-of-turn save adds <= 10 pages
  conn 3: c1 C(6000, SNAPS 4096) 8 tok -> c2 C2 = C[:4096]+t2 64 tok (rckpt cut)
The golds a2/c2 continue from exactly the states the SSD tier saved (a1-end,
the C cut), so after a restart the SSD restores must reproduce them token for
token, with the same spec statistics (the MTP layer is restored live).
b1-end's checkpoint references pages 16-22 as A computed them (dedupe): A and
B prefilled with different segmentations, the GEMMs pick kernels by M, so
those pages are not bitwise B's live KV and b2 may drift from its gold on a
near-tie (inherent to prefix caching). b2 is checked with --hit, and a second
restart must reproduce the first restore exactly (--ref restart).
State: logs/a4-ab.json
"""
import argparse, json, os, struct, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from qwentok import Tokenizer
from conc_verify import Conn

ROOT = Path(__file__).resolve().parent.parent
STATE = ROOT / 'logs' / 'a4-ab.json'
CK = struct.Struct('<8s4Q10i6Q')  # CkHdr (51_host_cfg.inc)
KV_PAGE = 256


def load():
    return json.loads(STATE.read_text()) if STATE.exists() else {}


def store(st):
    STATE.parent.mkdir(exist_ok=True)
    STATE.write_text(json.dumps(st))


def prompts():
    o8 = json.loads((ROOT / 'data/qsa-oracle/8192.json').read_text())['prompt_ids']
    o32 = json.loads((ROOT / 'data/qsa-oracle/32768.json').read_text())['prompt_ids']
    tok = Tokenizer()
    t2 = tok.encode('\n\nSummarize the text above in one sentence.\n')
    t3 = tok.encode('\n\nList three keywords for the text above.\n')
    A = o8[:8100]
    B = A[:6000] + o32[20000:22500]
    C = o32[0:6000]
    return A, B, C, t2, t3


def sync():
    c = Conn()
    v = c.cmd('KVSYNC')
    c.close()
    if v[0] != 'K':
        raise RuntimeError(f'KVSYNC: {v}')
    return dict(ckpts=int(v[1]), pages=int(v[2]), mtps=int(v[3]), bytes=int(v[4]))


def show(tag, key, ids, r):
    print(f'a4/{tag} {key:<3} |p|={len(ids):<5} n={len(r["tokens"]):3d} {r["reason"]:6s} '
          f'stats={r["stats"]}', flush=True)


def cmd_save(a):
    A, B, C, t2, t3 = prompts()
    st = {'gold': {}, 'prompts': {}, 'sync': {}}
    g, P = st['gold'], st['prompts']

    c = Conn()
    r = c.gen(A, 16, 1, 'SNAPS 1 4096'); show('e1', 'a1', A, r)
    st['sync']['a1'] = sync()
    P['a2'] = A + r['tokens'] + t2
    g['a2'] = c.gen(P['a2'], 64, 1); show('e1', 'a2', P['a2'], g['a2'])
    c.close()
    st['sync']['a2'] = sync()

    c = Conn()
    r = c.gen(B, 16, 1); show('e1', 'b1', B, r)
    st['sync']['b1'] = sync()
    P['b2'] = B + r['tokens'] + t3
    g['b2'] = c.gen(P['b2'], 64, 1); show('e1', 'b2', P['b2'], g['b2'])
    c.close()
    st['sync']['b2'] = sync()

    c = Conn()
    r = c.gen(C, 8, 1, 'SNAPS 1 4096'); show('e1', 'c1', C, r)
    st['sync']['c1'] = sync()
    P['c2'] = C[:4096] + t2
    g['c2'] = c.gen(P['c2'], 64, 1); show('e1', 'c2', P['c2'], g['c2'])
    c.close()
    st['sync']['c2'] = sync()
    for k, v in st['sync'].items():
        print(f'  sync after {k}: {v}', flush=True)
    store(st)

    # checks that belong to E1 itself
    bad = 0
    s = st['sync']
    npb = (len(B) + 15 - 1) // KV_PAGE     # full pages of B's end-of-turn state
    shared = 6000 // KV_PAGE                # pages whose prefix equals A's
    dp, dm = s['b1']['pages'] - s['a2']['pages'], s['b1']['mtps'] - s['a2']['mtps']
    ok = 1 <= dp <= npb - shared and dm <= npb - shared
    bad += not ok
    print(f'  {"PASS" if ok else "FAIL"}  B 与 A 共享前缀页：B 结束态 {npb} 页，新增 {dp} 页 / '
          f'{dm} 个 MTP 段（应 1..{npb - shared}）', flush=True)
    ok = all(g[k]['stats'][3] > 0 for k in g)
    bad += not ok
    print(f'  {"PASS" if ok else "FAIL"}  E1 gold 都是续写/检查点（n_cached: '
          f'{[g[k]["stats"][3] for k in g]}）', flush=True)
    ok = all(g[k]['stats'][1] > 0 for k in g)
    bad += not ok
    print(f'  {"PASS" if ok else "FAIL"}  E1 gold 都在投机（rounds: {[g[k]["stats"][1] for k in g]}）',
          flush=True)
    sys.exit(1 if bad else 0)


def keys(a):
    return a.only.split(',') if a.only else ['a2', 'b2', 'c2']


def cmd_run(a):
    st = load()
    if 'gold' not in st:
        sys.exit('run `save` first')
    runs = st.setdefault('runs', {}).setdefault(a.tag, {})
    for k in keys(a):
        c = Conn()  # fresh connection: no live state to continue
        r = c.gen(st['prompts'][k], 64, 1)
        c.close()
        runs[k] = r
        show(a.tag, k, st['prompts'][k], r)
    st.setdefault('sync', {})[a.tag] = sync()
    print(f'  sync: {st["sync"][a.tag]}', flush=True)
    store(st)


def cmd_compare(a):
    st = load()
    runs = st.get('runs', {}).get(a.tag)
    if not runs:
        sys.exit(f'no run {a.tag}')
    bad = 0

    def chk(ok, msg):
        nonlocal bad
        bad += not ok
        print(f'  {"PASS" if ok else "FAIL"}  {msg}', flush=True)

    ref = st['gold'] if a.ref == 'gold' else st['runs'][a.ref]
    for k in keys(a):
        r, g = runs[k], ref[k]
        first = next((i for i, (x, y) in enumerate(zip(r['tokens'], g['tokens'])) if x != y), None)
        if a.hit:  # restore must hit and speculate; differences vs the reference are reported only
            chk(r['stats'][3] > 0, f'{a.tag}/{k}: 命中 SSD 检查点 n_cached={r["stats"][3]}')
            chk(r['stats'][1] > 0, f'{a.tag}/{k}: 恢复后仍在投机 rounds={r["stats"][1]}')
            print(f'  INFO  {a.tag}/{k} vs {a.ref}: ' +
                  ('逐 token 相同' if r['tokens'] == g['tokens'] else f'首个不同 @{first}') +
                  f'，spec {r["stats"]} vs {g["stats"]}', flush=True)
            continue
        if a.miss:  # damaged cache: a plain recompute that still completes
            chk(r['stats'][3] == 0, f'{a.tag}/{k}: 未命中（坏页）→ 整段重算 n_cached={r["stats"][3]}')
            chk(len(r['tokens']) > 0, f'{a.tag}/{k}: 请求正常完成（{len(r["tokens"])} tok, {r["reason"]}）')
            continue
        chk(r['stats'][3] > 0, f'{a.tag}/{k}: 命中 SSD 检查点 n_cached={r["stats"][3]}')
        chk(r['stats'][1] > 0, f'{a.tag}/{k}: 恢复后仍在投机 rounds={r["stats"][1]}')
        chk(r['tokens'] == g['tokens'],
            f'{a.tag}/{k}: 逐 token == {a.ref}' + ('' if first is None else f'（首个不同 @{first}）'))
        chk(r['stats'] == g['stats'], f'{a.tag}/{k}: spec 统计 == {a.ref}（{r["stats"]} vs {g["stats"]}）')
    print('a4_ab', 'ALL PASS' if bad == 0 else f'{bad} FAIL', flush=True)
    sys.exit(1 if bad else 0)


def inventory(d):
    d = Path(d)
    cks, files = [], {}
    for p in d.iterdir():
        files[p.name] = p.stat().st_size
        if p.name.startswith('k_') and p.name.endswith('.kck'):
            b = p.read_bytes()
            h = CK.unpack_from(b)
            nc = h[6]
            ch = struct.unpack_from(f'<{nc}Q', b, CK.size)
            mt = struct.unpack_from(f'<{nc}Q', b, CK.size + 8 * nc)
            cks.append(dict(name=p.name, n=h[5], t=p.stat().st_mtime_ns, size=len(b),
                            chunks=[f'c_{k:016x}.kvc' for k in ch],
                            mtps=[f'm_{k:016x}.kvm' for k in mt if k]))
    cks.sort(key=lambda e: -e['t'])
    return cks, files


def closure(cks, files):
    names = set()
    for e in cks:
        names.add(e['name'])
        names.update(x for x in e['chunks'] + e['mtps'] if x in files)
    return names


def cmd_predict(a):
    cks, files = inventory(a.dir)
    if len(cks) <= a.keep:
        sys.exit(f'only {len(cks)} checkpoints on disk, need > {a.keep}')
    keep = closure(cks[:a.keep], files)
    size = sum(files[x] for x in keep)
    nxt = sum(files[x] for x in closure(cks[:a.keep + 1], files))
    cap = (size + nxt) / 2  # strictly between: the newest `keep` fit, one more does not
    st = load()
    st['lru'] = dict(keep=sorted(keep), cap=cap, ck=[e['name'] for e in cks[:a.keep]])
    store(st)
    total = sum(files.values())
    kept = ', '.join('%s n=%d' % (e['name'], e['n']) for e in cks[:a.keep])
    print(f'  disk: {len(cks)} ckpts, {len(files)} files, {total / 2**30:.2f} GiB; keep newest '
          f'{a.keep} ({kept}) '
          f'= {size / 2**30:.3f} GiB, +1 = {nxt / 2**30:.3f} GiB', flush=True)
    print(f'{cap / 2**30:.4f}')  # last line: GDEC_KVSNAP_MAX_GB


def cmd_check_lru(a):
    st = load()
    want = set(st['lru']['keep'])
    _, files = inventory(a.dir)
    have = set(files)
    extra, miss = sorted(have - want), sorted(want - have)
    ok = not extra and not miss
    print(f'  {"PASS" if ok else "FAIL"}  LRU 后磁盘文件 == 预测（保留 {len(want)} 个文件，'
          f'多出 {extra[:4]}{"…" if len(extra) > 4 else ""}，缺少 {miss[:4]}）', flush=True)
    sys.exit(0 if ok else 1)


def cmd_check_disk(a):
    s = sync()
    _, files = inventory(a.dir)
    cnt = lambda p, x: sum(1 for f in files if f.startswith(p) and f.endswith(x))
    disk = dict(ckpts=cnt('k_', '.kck'), pages=cnt('c_', '.kvc'), mtps=cnt('m_', '.kvm'),
                bytes=sum(files.values()))
    ok = disk == s and len(files) == disk['ckpts'] + disk['pages'] + disk['mtps']
    print(f'  {"PASS" if ok else "FAIL"}  索引 == 磁盘（KVSYNC {s}，磁盘 {disk}，'
          f'其它文件 {sorted(f for f in files if f[:2] not in ("k_", "c_", "m_"))[:4]}）',
          flush=True)
    sys.exit(0 if ok else 1)


def cmd_corrupt(a):
    n = 0
    for p in Path(a.dir).glob('c_*.kvc'):
        with open(p, 'r+b') as f:
            f.seek(1072 + 4096)  # inside layer 0's K rows
            b = f.read(1)
            f.seek(1072 + 4096)
            f.write(bytes([b[0] ^ 0x5a]))
        n += 1
    print(f'  corrupted {n} page files', flush=True)
    sys.exit(0 if n else 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['save', 'run', 'compare', 'predict', 'check-lru', 'check-disk',
                                    'corrupt', 'sync', 'reset'])
    ap.add_argument('--tag', default='restart')
    ap.add_argument('--only', default='')
    ap.add_argument('--dir', default=str(ROOT / 'data' / 'kvsnap-a4'))
    ap.add_argument('--keep', type=int, default=2)
    ap.add_argument('--miss', action='store_true')
    ap.add_argument('--ref', default='gold')  # compare against gold or another run tag
    ap.add_argument('--hit', action='store_true')
    a = ap.parse_args()
    if a.cmd == 'reset':
        STATE.unlink(missing_ok=True)
    elif a.cmd == 'sync':
        print(sync())
    else:
        {'save': cmd_save, 'run': cmd_run, 'compare': cmd_compare, 'predict': cmd_predict,
         'check-lru': cmd_check_lru, 'check-disk': cmd_check_disk,
         'corrupt': cmd_corrupt}[a.cmd](a)


if __name__ == '__main__':
    main()
