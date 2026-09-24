#!/usr/bin/env python3
"""B0: MTP KV in the page pool — speculation survives rckpt restores.

Usage (repo root, one fresh engine on 127.0.0.1:8732 per run; GDEC_KVSNAP=0,
GDEC_RCKPT_MIN=1024 — tools/b0_verify.sh drives all of this):
  python3 tools/b0_ab.py run --tag ck|nock|ref
  python3 tools/b0_ab.py compare

Scenario (MTP drafter 1 / chain 4, greedy unless noted):
  s1 A(8100) -> s2 A+t2 -> s3 B -> s4 A+t2 -> s5 A+t3 (chain) -> s6 A+t3 (sampled)
  -> s7 C(6000, SNAPS 4096) -> s8 C[:4096]+t2 -> s9 B -> s10 C[:4096]+t2
  s2/s4/s5/s6 restore A's prompt-end checkpoint, s8/s10 restore the SNAPS-cut
  checkpoint taken mid-prefill. With B0 those restores keep the MTP layer live:
  spec rounds > 0, and the repeat restores (s4 vs s2, s10 vs s8) match exactly.
  `nock` (GDEC_RCKPT=0) recomputes every prompt and runs spec from a fresh MTP
  ingest: the restored runs' draft acceptance must be close to it (a stale or
  misaddressed MTP KV collapses acceptance). `ref` (the pre-B0 binary) restores
  with plain decode (rounds == 0): from the same restored trunk state, greedy
  speculation must reproduce its tokens exactly.
State: logs/b0-ab.json
"""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from qwentok import Tokenizer
from conc_verify import Conn
from a5_ab import SAMPLE

ROOT = Path(__file__).resolve().parent.parent
STATE = ROOT / 'logs' / 'b0-ab.json'
RESTORED = ['s2', 's4', 's5', 's6', 's8', 's10']
SAME = [('s4', 's2'), ('s10', 's8')]


def prompts():
    o8 = json.loads((ROOT / 'data/qsa-oracle/8192.json').read_text())['prompt_ids']
    o32 = json.loads((ROOT / 'data/qsa-oracle/32768.json').read_text())['prompt_ids']
    tok = Tokenizer()
    A = o8[:8100]  # 8100 % 256 = 164: the checkpoint's last page is partial
    t2 = tok.encode('\n\nSummarize the text above in one sentence.\n')
    t3 = tok.encode('\n\nList three keywords for the text above.\n')
    C = o32[0:6000]
    return dict(A=A, Q2=A + t2, Q3=A + t3, B=o32[20000:23000], C=C, C2=C[:4096] + t2)


# key, prompt, max tokens, drafter, suffix
SCN = [('s1', 'A', 8, 1, ''), ('s2', 'Q2', 96, 1, ''), ('s3', 'B', 8, 1, ''),
       ('s4', 'Q2', 96, 1, ''), ('s5', 'Q3', 96, 4, ''), ('s6', 'Q3', 64, 1, SAMPLE),
       ('s7', 'C', 8, 1, 'SNAPS 1 4096'), ('s8', 'C2', 96, 1, ''), ('s9', 'B', 8, 1, ''),
       ('s10', 'C2', 96, 1, '')]


def acc(r):
    # stats = [drafter, rounds, commit, n_cached, proposed]; committed drafts =
    # commit - rounds (each round also emits the verify/bonus token)
    d, rounds, commit, _, prop = r['stats']
    if not rounds or not prop:
        return None
    return (commit - rounds) / prop


def run(tag):
    p = prompts()
    st = json.loads(STATE.read_text()) if STATE.exists() else {}
    res = {}
    for key, name, n, d, sfx in SCN:
        c = Conn()
        r = c.gen(p[name], n, d, sfx)
        c.close()
        res[key] = r
        a = acc(r)
        print(f'b0/{tag} {key:<3} {name:<3} |p|={len(p[name]):<5} d={d} n={len(r["tokens"]):3d} '
              f'{r["reason"]:6s} stats={r["stats"]} acc={"-" if a is None else f"{a:.2f}"}', flush=True)
    st[tag] = res
    STATE.parent.mkdir(exist_ok=True)
    STATE.write_text(json.dumps(st))


def compare():
    st = json.loads(STATE.read_text()) if STATE.exists() else {}
    ck, nock, ref = st.get('ck'), st.get('nock'), st.get('ref')
    if not ck or not nock:
        print('b0_ab: missing ck/nock runs'); sys.exit(1)
    bad = 0

    def chk(ok, msg):
        nonlocal bad
        bad += not ok
        print(f'  {"PASS" if ok else "FAIL"}  {msg}')

    for k in RESTORED:
        r = ck[k]
        chk(r['stats'][3] > 0, f'{k}: ck 命中检查点 n_cached={r["stats"][3]}')
        chk(r['stats'][1] > 0, f'{k}: ck 恢复后仍在投机 rounds={r["stats"][1]}')
    for a, b in SAME:
        ok = ck[a]['tokens'] == ck[b]['tokens'] and ck[a]['stats'] == ck[b]['stats']
        chk(ok, f'{a} == {b}（同一检查点两次恢复，逐 token + spec 统计）')
    # acceptance: restored MTP state vs a fresh prefill ingest of the same prompt
    for k in RESTORED:
        if k == 's6':
            continue  # sampled: acceptance is noisy over 64 tokens
        a, b = acc(ck[k]), acc(nock[k])
        ok = a is not None and b is not None and a >= b - 0.15
        chk(ok, f'{k}: 接受率 ck {a if a is None else round(a, 2)} vs nock（整段重算）'
                f' {b if b is None else round(b, 2)}（允许低 0.15）')
    same = sum(ck[k]['tokens'] == nock[k]['tokens'] for k in RESTORED if k != 's6')
    print(f'  info  greedy 恢复 vs 整段重算 token 完全相同: {same}/{len(RESTORED) - 1}'
          '（分块不同，末位数值可能不同，仅供参考）')
    if ref:
        rr = [ref[k]['stats'][1] for k in RESTORED]
        print(f'  info  ref（B0 之前）恢复后 rounds: {rr}（应全为 0：旧版恢复后关投机）')
        # same restored trunk state: greedy MTP speculation must reproduce the
        # pre-B0 plain decode token for token
        for k in RESTORED:
            if k != 's6':
                chk(ck[k]['tokens'] == ref[k]['tokens'],
                    f'{k}: ck 恢复后投机 == ref 恢复后串行解码（逐 token）')
    print('b0_ab', 'ALL PASS' if bad == 0 else f'{bad} FAIL', flush=True)
    sys.exit(1 if bad else 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['run', 'compare', 'reset'])
    ap.add_argument('--tag', default='ck')
    a = ap.parse_args()
    if a.cmd == 'reset':
        STATE.unlink(missing_ok=True)
    elif a.cmd == 'compare':
        compare()
    else:
        run(a.tag)


if __name__ == '__main__':
    main()
