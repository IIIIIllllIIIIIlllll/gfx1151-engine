"""A3: RAM checkpoints that survive conversation switches (GDEC_KV_PAGED pins KV pages).

Usage (repo root, one fresh engine on 127.0.0.1:8732 per run; GDEC_KVSNAP=0,
GDEC_RCKPT_MIN=1024 — tools/a3_verify.sh drives all of this):
  python3 tools/rckpt_ab.py switch --tag off|p1|p2|p1big
  python3 tools/rckpt_ab.py evict  --tag off|p2         # engine with --maxctx 16384
  python3 tools/rckpt_ab.py compare

switch: A(8100 tok, not page aligned) -> A+t2 -> A+t3 -> B -> A+t2 -> A+t3 -> C -> A+t2
  r2/r3 restore A's checkpoint in every config (the live stream still starts
  with A). After B, unpaged engines have pruned A (full prefill); paged engines
  must restore A again, so r5 == r2, r6 == r3, r8 == r2 bit-exactly — that
  also proves copy-on-write kept A's partial tail page intact.
evict: A(8100) -> B2(6000) -> C2(12000) -> A+t2 in a 64-page pool: C2 forces
  the LRU checkpoints out; every output must equal the unpaged engine's.
State: logs/rckpt-ab.json
"""
import argparse, json, sys
from pathlib import Path
from qwentok import Tokenizer
from kvsnap_ab import gen

ROOT = Path(__file__).resolve().parent.parent
STATE = ROOT / 'logs' / 'rckpt-ab.json'


def prompts():
    o8 = json.loads((ROOT / 'data/qsa-oracle/8192.json').read_text())['prompt_ids']
    o32 = json.loads((ROOT / 'data/qsa-oracle/32768.json').read_text())['prompt_ids']
    tok = Tokenizer()
    A = o8[:8100]  # 8100 % 256 = 164: the checkpoint's last page is partial
    t2 = tok.encode('\n\nSummarize the text above in one sentence.\n')
    t3 = tok.encode('\n\nList three keywords for the text above.\n')
    return dict(A=A, Q2=A + t2, Q3=A + t3,
                B=o32[20000:23000], C=o32[24000:26500],
                B2=o32[9000:15000], C2=o32[15000:27000])


SCN = {
    'switch': [('r1', 'A', 8), ('r2', 'Q2', 24), ('r3', 'Q3', 24), ('r4', 'B', 8),
               ('r5', 'Q2', 24), ('r6', 'Q3', 24), ('r7', 'C', 8), ('r8', 'Q2', 24)],
    'evict': [('e1', 'A', 8), ('e2', 'B2', 8), ('e3', 'C2', 8), ('e4', 'Q2', 24)],
}
# paged engines: these later requests must reproduce the earlier restore exactly
SAME = {'switch': [('r5', 'r2'), ('r6', 'r3'), ('r8', 'r2')], 'evict': []}


def run(scn, tag):
    p = prompts()
    st = json.loads(STATE.read_text()) if STATE.exists() else {}
    res = {}
    for i, (k, name, n) in enumerate(SCN[scn]):
        res[k] = gen(p[name], n, i + 1)
        print(f'{scn}/{tag} {k} {name:<3} |prompt|={len(p[name]):<6} -> {res[k][:8]}...', flush=True)
    st.setdefault(scn, {})[tag] = res
    STATE.parent.mkdir(exist_ok=True)
    STATE.write_text(json.dumps(st))


def compare():
    st = json.loads(STATE.read_text()) if STATE.exists() else {}
    bad = 0
    for scn, runs in st.items():
        if 'off' not in runs:
            print(f'{scn}: no "off" reference run'); bad += 1; continue
        ref = runs['off']
        keys = [k for k, _, _ in SCN[scn]]
        print(f'== {scn} ==  ' + ' '.join(f'{k:>6}' for k in keys))
        self_ref = dict(SAME[scn])
        for tag, r in runs.items():
            if tag == 'off':
                print(f'{tag:<8}' + ' '.join(f'{"ref":>6}' for _ in keys))
                continue
            row = []
            for k in keys:
                # after the switch an unpaged engine has pruned A and re-prefills,
                # a paged one restores A: compare against this engine's own r2/r3
                ok = r.get(k) is not None and r.get(k) == (r.get(self_ref[k]) if k in self_ref else ref.get(k))
                bad += not ok
                row.append('MATCH' if ok else 'DIFF')
            print(f'{tag:<8}' + ' '.join(f'{c:>6}' for c in row))
        if scn == 'switch':
            same = all(ref[a] == ref[b] for a, b in SAME[scn])
            print(f'(off 的 r5/r6/r8 是整段重算，与 r2/r3 是否相同仅供参考: {same})')
            print('说明: r1-r4、r7 与 off 比；r5/r6/r8 与同一引擎自己的 r2/r3/r2 比（分页时必须命中检查点）')
    print('rckpt_ab', 'ALL MATCH' if bad == 0 else f'{bad} DIFF', flush=True)
    sys.exit(1 if bad else 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['switch', 'evict', 'compare', 'reset'])
    ap.add_argument('--tag', default='off')
    a = ap.parse_args()
    if a.cmd == 'reset':
        STATE.unlink(missing_ok=True)
    elif a.cmd == 'compare':
        compare()
    else:
        run(a.cmd, a.tag)


if __name__ == '__main__':
    main()
